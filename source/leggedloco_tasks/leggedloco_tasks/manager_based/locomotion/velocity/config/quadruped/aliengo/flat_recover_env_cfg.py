
from isaaclab.utils import configclass

from .rough_recover_env_cfg import AlienGoRoughRecoverEnvCfg
from leggedloco_tasks.assets.terrains import FLAT_TERRAINS_CFG


@configclass
class AlienGoFlatRecoverEnvCfg(AlienGoRoughRecoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.terrain.terrain_generator = FLAT_TERRAINS_CFG
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.50)
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": -0.1,
            ".*R_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }

        # ------------------------------Events------------------------------
        # startup
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 1.2)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.5, 1.0)

        # reset
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-10.0, 10.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-5.0, 5.0)
        self.events.randomize_reset_joints.params["position_range"] = (-0.1, 0.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.1),
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
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.body_lin_acc_l2.weight = -0.0

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -5e-5
        self.rewards.joint_vel_l2.weight = -0.0
        self.rewards.joint_acc_l2.weight = -5e-8
        # self.rewards.create_joint_deviation_l1_rewTerm("joint_deviation_hip_l1", -0.4, [".*_hip_joint"])
        # self.rewards.create_joint_deviation_l1_rewTerm("joint_deviation_thigh_l1", -0.04, [".*_thigh_joint"])
        # self.rewards.create_joint_deviation_l1_rewTerm("joint_deviation_calf_l1", -0.04, [".*_calf_joint"])
        self.rewards.joint_deviation_lowCmd.weight = -0.1  # -1.0
        self.rewards.joint_pos_limits.weight = -0.5  # -5.0
        self.rewards.joint_vel_limits.weight = -0.0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_without_cmd.weight = -1.0  # -2.0
        self.rewards.joint_mirror.weight = -0.25  # -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
            ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.action_smoothness.weight = -0.02

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_hip", ".*_thigh", ".*_calf"]
        self.rewards.contact_forces.weight = -1.5e-4

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time_variance.weight = -0.5
        self.rewards.feet_gait.weight = 0.05
        self.rewards.feet_contact.weight = -0.0
        self.rewards.feet_contact_without_cmd.weight = 0.5  # 0.1
        self.rewards.feet_stumble.weight = -0.0
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_height.weight = -0.0
        self.rewards.feet_height_body.weight = -5.0  # -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.27
        self.rewards.feet_distance_y_exp.weight = 0.0

        # Upward
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AlienGoFlatRecoverEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class AlienGoFlatRecoverPlayEnvCfg(AlienGoFlatRecoverEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.episode_length_s = 5.0

        # ------------------------------Sence------------------------------
        # make a smaller scene for play
        self.scene.num_envs = 50
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = 5
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 4
            self.scene.terrain.terrain_generator.curriculum = False

        # ------------------------------Observations------------------------------
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.observations.proprio.enable_corruption = False

        # ------------------------------Events------------------------------
        # remove random pushing event
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.8, 0.8),
                "y": (-0.8, 0.8),
                "z": (0.0, 0.1),
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
        self.events.randomize_push_robot = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)