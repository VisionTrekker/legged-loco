# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PreTrainedPolicyAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: PreTrainedPolicyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: PreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)  # low level pre-trained policy
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        # (num_envs, 3)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_act_pos_term: ActionTerm = cfg.low_level_act_pos.class_type(cfg.low_level_act_pos, env)  # joint_actions.JointPositionAction
        if cfg.low_level_act_vel is not None:  # wheel velocity action
            self._low_level_act_vel_term: ActionTerm = cfg.low_level_act_vel.class_type(cfg.low_level_act_vel, env)  # joint_actions.JointVelocityAction

        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_act_pos_term.action_dim, device=self.device)  # (num_envs, num_joints)
        if cfg.low_level_act_vel is not None:
            self.low_level_actions = torch.zeros(self.num_envs, self._low_level_act_pos_term.action_dim + self._low_level_act_vel_term.action_dim, device=self.device)

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # remap some of the low level observations to internal observations，修改 低阶policy的 观测
        #   1. actions 更新为 预训练policy推理出的 low_level_actions (num_envs, num_joints)
        cfg.low_level_obs_policy.actions.func = lambda dummy_env: last_action()
        cfg.low_level_obs_policy.actions.params = dict()  # 与原有保持一致
        #   2. velocity_commands 更新为 navigation给出的 commands
        cfg.low_level_obs_policy.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_obs_policy.velocity_commands.params = dict()

        # add the low level observations to the observation manager
        self._low_level_obs_policy_manager = ObservationManager({"lowLevel_policy": cfg.low_level_obs_policy}, env)
        self._low_level_obs_policy_dim = self._low_level_obs_policy_manager.compute_group("lowLevel_policy").shape[1]
        print(f"[INFO] _low_level_obs_policy_manager: {self._low_level_obs_policy_manager}")

        self._low_level_obs_proprio_dim = 0
        if cfg.low_level_obs_proprio is not None:
            cfg.low_level_obs_proprio.actions.func = lambda dummy_env: last_action()
            cfg.low_level_obs_proprio.actions.params = dict()
            cfg.low_level_obs_proprio.velocity_commands.func = lambda dummy_env: self._raw_actions
            cfg.low_level_obs_proprio.velocity_commands.params = dict()

            self._low_level_obs_proprio_manager = ObservationManager({"lowLevel_proprio": cfg.low_level_obs_proprio}, env)
            self._low_level_obs_proprio_dim = self._low_level_obs_proprio_manager.compute_group("lowLevel_proprio").shape[1]
            print(f"[INFO] _low_level_obs_proprio_manager: {self._low_level_obs_proprio_manager}")

        self._counter = 0

        # for history observation
        self._low_level_obs_his_len = cfg.low_level_obs_his_len
        if self._low_level_obs_proprio_dim > 0:
            # (num_envs, 5, 45)
            self._low_level_obs_proprio_buf = torch.zeros(self.num_envs, self._low_level_obs_his_len, self._low_level_obs_proprio_dim, dtype=torch.float, device=self.device)
        else:
            # (num_envs, 6, 45)
            self._low_level_obs_his_buf = torch.zeros(self.num_envs, self._low_level_obs_his_len + 1, self._low_level_obs_policy_dim, dtype=torch.float, device=self.device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs_policy = self._low_level_obs_policy_manager.compute_group("lowLevel_policy")

            # history observations
            if self._low_level_obs_proprio_dim > 0:
                low_level_obs_proprio = self._low_level_obs_proprio_manager.compute_group("lowLevel_proprio")

                self._low_level_obs_proprio_buf = torch.cat([
                    low_level_obs_proprio.unsqueeze(1),
                    self._low_level_obs_proprio_buf[:, :-1],
                ], dim=1)
                low_level_obs_proprio_his = self._low_level_obs_proprio_buf.view(self.num_envs, -1)
                low_level_obs_his = torch.cat([low_level_obs_policy, low_level_obs_proprio_his], dim=1)
            else:
                self._low_level_obs_his_buf = torch.cat([
                    low_level_obs_policy.unsqueeze(1),
                    self._low_level_obs_his_buf[:, :-1],
                ], dim=1)
                low_level_obs_his = self._low_level_obs_his_buf.view(self.num_envs, -1)

            # 预训练policy推理出的 low-level-actions
            self.low_level_actions[:] = self.policy(low_level_obs_his)
            # 预处理: actions * scale + offset ==> clip
            self._low_level_act_pos_term.process_actions(self.low_level_actions[:, :self._low_level_act_pos_term.action_dim])
            self._low_level_act_vel_term.process_actions(self.low_level_actions[:, -self._low_level_act_vel_term.action_dim:])
            self._counter = 0
        # 应用 low-level-actions 到 robot
        self._low_level_act_pos_term.apply_actions()
        self._low_level_act_vel_term.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class PreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = PreTrainedPolicyAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_act_pos: ActionTermCfg = MISSING
    """Low level action position configuration."""
    low_level_act_vel: ActionTermCfg = MISSING
    """Low level action velocity configuration."""
    low_level_obs_policy: ObservationGroupCfg = MISSING
    """Low level policy observation configuration."""
    low_level_obs_proprio: ObservationGroupCfg = MISSING
    """Low level proprioception observation configuration."""
    low_level_obs_his_len: int = 0
    """Number of history steps to include in the low level observation. if 0, no proprioception history is included."""
    debug_vis: bool = True
    """Whether to visualize debug information. Defaults to False."""
