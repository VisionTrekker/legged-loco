import math
from isaaclab.utils import configclass

from .rough_env_cfg import AlienGoRoughEnvCfg

##
# Pre-defined configs
##
# use local assets
from leggedloco_tasks.assets.robots.unitree import UNITREE_ALIENGO_CFG


@configclass
class AlienGoRoughRecoverEnvCfg(AlienGoRoughEnvCfg):
    """Configuration for the AlienGo locomotion velocity-tracking environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -0.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0  # -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.0
        self.rewards.base_height_l2.weight = -0.0
        self.rewards.body_lin_acc_l2.weight = -0.0

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -5.0e-5
        self.rewards.joint_vel_l2.weight = -0.0
        self.rewards.joint_acc_l2.weight = -5e-8  # -2.5e-7

        self.rewards.joint_pos_deviation.weight = -0.1  # -1.0
        self.rewards.stand_still_without_cmd.weight = -1.0  # -2.0
        self.rewards.joint_pos_limits.weight = -1.0  # -5.0

        self.rewards.joint_vel_limits.weight = -0.0
        self.rewards.wheel_vel_lowCmd.weight = -0.0

        self.rewards.joint_power.weight = -1e-5  # -2e-5
        self.rewards.joint_mirror.weight = -0.1  # -0.05

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness.weight = -0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.contact_forces.weight = -1.5e-4

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_contact.weight = -0.0
        self.rewards.feet_contact_without_cmd.weight = 0.2  # 0.1
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_height.weight = -0.0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height_body.weight = -1.0  # -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.32
        self.rewards.feet_distance_y_exp.weight = 0.0

        # Upward
        self.rewards.upward.weight = 0.5  # 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AlienGoRoughRecoverEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # --- first 2.0s don't activate illegal_contact when recover-mode ---
        # self.terminations.illegal_contact = DoneTerm(
        #     func=mdp.illegal_contact_delay,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        #         "contact_threshold": 1.0,
        #         "episode_length_threshold": 100.0,  # 2.0s / 0.02
        #     }
        # )
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)

        # ------------------------------Curriculums------------------------------
        # self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.lin_vel_cmd_levels.params["vel_range_multiplier"] = (1.0, 1.5)


@configclass
class AlienGoRoughRecoverPlayEnvCfg(AlienGoRoughRecoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 10.0

        # ------------------------------Sence------------------------------
        # make a smaller scene for play
        self.scene.num_envs = 50
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = 5
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.3
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.16, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.3
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.16, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.1
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.1
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1


        # ------------------------------Observations------------------------------
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        if self.observations.proprio is not None:
            self.observations.proprio.enable_corruption = False

        # ------------------------------Events------------------------------
        # remove random pushing event
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (1.57, 3.14),
                "pitch": (-1.57, 1.57),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }
        self.events.randomize_push_robot = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)