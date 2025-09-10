
from isaaclab.utils import configclass

from .rough_lidar_env_cfg import AlienGoRoughLidarEnvCfg
from eilab_tasks.assets.terrains import FLAT_TERRAINS_CFG


@configclass
class AlienGoFlatLidarEnvCfg(AlienGoRoughLidarEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG

        # ------------------------------Events------------------------------
        # startup
        self.events.physics_material.params["static_friction_range"] = (0.1, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.1, 1.5)

        # reset
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # ------------------------------Rewards------------------------------
        # Root penalties
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.base_height_l2.params["target_height"] = 0.43

        # Joint penalties
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.4, [".*_hip_joint"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_thigh_l1", -0.04, [".*_thigh_joint"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_calf_l1", -0.04, [".*_calf_joint"])

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness.weight = -0.02

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_slide.weight = -0.05

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AlienGoFlatLidarEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class AlienGoFlatLidarPlayEnvCfg(AlienGoFlatLidarEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

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

        # ------------------------------Observations------------------------------
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False

        # ------------------------------Events------------------------------
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.actuator_gains = None
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                # "roll": (-3.14, 3.14),
                # "pitch": (-3.14, 3.14),
                # "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.push_robot = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)