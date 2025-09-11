# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import eilab_tasks.manager_based.navigation.mdp as mdp


##
# Scene definition
##


from eilab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
LOW_LEVEL_ENV_CFG = LocomotionVelocityRoughEnvCfg()


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    base_pose = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-5.0, 5.0), pos_y=(-5.0, 5.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pretrained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_act_pos=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_obs_policy=LOW_LEVEL_ENV_CFG.observations.policy,
        low_level_obs_proprio=LOW_LEVEL_ENV_CFG.observations.proprio if LOW_LEVEL_ENV_CFG.observations.proprio is not None else None,
        low_level_obs_his_len=5,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_pose"},
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    randomize_reset_base = EventTerm(  # 随机更改env的base 位置、速度（ ± ）
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    is_terminated = RewTerm(
        func=mdp.is_terminated, weight=-400.0
    )
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 2.0, "command_name": "base_pose"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={"std": 0.2, "command_name": "base_pose"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.2,
        params={"command_name": "base_pose"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # MDP terminations
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Contact sensor
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 1.0},
    )

    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )



##
# Environment configuration
##


@configclass
class NavigationRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the navigation environment."""

    # Scene settings
    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""

        # general settings
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 10  # = 40
        self.episode_length_s = self.commands.base_pose.resampling_time_range[1]  # = 10
        # simulation settings (sim : isaaclab/sim/simlation_cfg.py/SimulationCfg())
        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt  # = 0.005
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation  # = 4
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period =  self.sim.dt * self.actions.pretrained_policy_action.low_level_decimation
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.update_period = self.sim.dt * self.actions.pretrained_policy_action.low_level_decimation
        if self.scene.lidar_scanner is not None:
            self.scene.lidar_scanner.update_period = self.sim.dt * self.actions.pretrained_policy_action.low_level_decimation
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)