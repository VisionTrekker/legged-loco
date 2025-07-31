import math
from isaaclab.utils import configclass

from leggedloco_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# use local assets
from leggedloco_tasks.assets.robots.unitree import UNITREE_ALIENGO_DCMOTOR_CFG


@configclass
class AlienGoRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the AlienGo locomotion velocity-tracking environment."""
    base_link_name = "trunk"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_ALIENGO_DCMOTOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        }
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # no lidar sensor
        self.scene.lidar_scanner = None

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.lidar_scan = None

        self.observations.proprio = None

        self.observations.critic.base_lin_vel.scale = 2.0
        self.observations.critic.base_ang_vel.scale = 0.25
        self.observations.critic.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.critic.joint_pos.scale = 1.0
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 0.5
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        # startup
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.5, 0.9)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.5, 0.7)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.3)
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.0, 3.0)
        self.events.randomize_rigid_body_inertia.params["inertia_distribution_params"] = (0.75, 1.25)
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = [self.base_link_name]

        # reset
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-5.0, 5.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-2.0, 2.0)
        self.events.randomize_reset_joints.params["position_range"] = (-0.1, 0.1)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                # "roll": (-3.14, 3.14),
                # "pitch": (-3.14, 3.14),
                # "yaw": (-3.14, 3.14),
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

        # interval
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -10.0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = -5.0
        self.rewards.base_height_l2.params["target_height"] = 0.43
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = -0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1e-4
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -1e-7
        # self.rewards.create_joint_deviation_l1_rewTerm("joint_deviation_hip_l1", -0.1, [".*_hip_joint"])
        # self.rewards.create_joint_deviation_l1_rewTerm("joint_deviation_thigh_l1", -0.1, [".*_thigh_joint"])
        # self.rewards.create_joint_deviation_l1_rewTerm("joint_deviation_calf_l1", -0.1, [".*_calf_joint"])
        self.rewards.joint_deviation_lowCmd.weight = -0.1  # -1.0
        self.rewards.stand_still_without_cmd.weight = -2.0  # -2.0
        self.rewards.joint_pos_limits.weight = -3.0  # -5.0
        self.rewards.joint_vel_limits.weight = -0.0
        self.rewards.joint_power.weight = -1e-5
        self.rewards.joint_mirror.weight = -0.1  # -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
            ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness.weight = -0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight= -0.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.1
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.feet_contact.weight = -0.0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 1.0  # 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.02
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = -0.0
        self.rewards.feet_height.params["target_height"] = 0.15
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -1.0  # -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.25
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AlienGoRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)
        self.commands.base_velocity.ranges_limits.lin_vel_x = (-1.0, 1.5)
        self.commands.base_velocity.ranges_limits.lin_vel_y = (-1.0, 1.0)


@configclass
class AlienGoRoughPlayEnvCfg(AlienGoRoughEnvCfg):
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
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.2
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.05, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.2
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.05, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.15
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.15
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.15
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.15


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
                # "roll": (-3.14, 3.14),
                # "pitch": (-3.14, 3.14),
                # "yaw": (-3.14, 3.14),
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
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)