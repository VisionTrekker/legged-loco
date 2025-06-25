
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

        self.observations.proprio.base_lin_vel.scale = 2.0
        self.observations.proprio.base_lin_vel = None
        self.observations.proprio.base_ang_vel.scale = 0.25
        self.observations.proprio.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.proprio.joint_pos.scale = 1.0
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.proprio.joint_vel.scale = 0.05
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.proprio.lidar_scan = None

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
        self.events.add_base_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 3.0)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]

        # reset
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.base_external_force_torque.params["force_range"] = (-5.0, 5.0)
        self.events.base_external_force_torque.params["torque_range"] = (-5.0, 5.0)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
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

        # interval
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = 0.0

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -0.0002
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*_hip_joint"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_thigh_l1", -0.1, [".*_thigh_joint"])
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_calf_l1", -0.1, [".*_calf_joint"])
        self.rewards.joint_pos_limits.weight = -0.0002
        self.rewards.joint_power.weight = -2e-5

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -5.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.02
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.05
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "AlienGoRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


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
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.13, 0.23)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.13, 0.23)

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