# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import leggedloco_tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class LocomotionVelocityRoughRecoverEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        """Post initialization."""

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings (sim : isaaclab/sim/simlation_cfg.py/SimulationCfg())
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2 ** 15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.lidar_scanner is not None:
            self.scene.lidar_scanner.update_period = self.sim.dt * self.decimation
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

        # --- Change the rewards functions to recover mode ---
        # Root penalties
        self.rewards.lin_vel_z_l2.func = mdp.lin_vel_z_l2_up
        self.rewards.ang_vel_xy_l2.func = mdp.ang_vel_xy_l2_up
        self.rewards.flat_orientation_l2.func = mdp.flat_orientation_l2_up
        self.rewards.base_height_l2.func = mdp.base_height_l2_up
        # Joint penalties
        self.rewards.joint_pos_deviation.func = mdp.joint_pos_deviation_up
        self.rewards.stand_still_without_cmd.func = mdp.stand_still_without_cmd_up
        self.rewards.joint_mirror.func = mdp.joint_mirror_up
        # Action penalties
        self.rewards.action_mirror.func = mdp.action_mirror_up
        self.rewards.action_sync.func = mdp.action_sync_up
        # Contact sensor
        self.rewards.undesired_contacts.func = mdp.undesired_contacts_up
        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_exp_up
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp_up
        # Others
        self.rewards.feet_air_time.func = mdp.feet_air_time_up
        self.rewards.feet_contact.func = mdp.feet_contact_up
        self.rewards.feet_contact_without_cmd.func = mdp.feet_contact_without_cmd_up
        self.rewards.feet_stumble.func = mdp.feet_stumble_up
        self.rewards.feet_slide.func = mdp.feet_slide_up
        self.rewards.feet_height.func = mdp.feet_height_up
        self.rewards.feet_height_body.func = mdp.feet_height_body_up
        self.rewards.feet_distance_y_exp.func = mdp.feet_distance_y_exp_up

        # --- Change the terminations to recover mode ---
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact_delay,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
                "contact_threshold": 1.0,
                "episode_length_threshold": 100.0,  # 2.0s / 0.02
            }
        )

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)
