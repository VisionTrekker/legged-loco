# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import eilab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from eilab_tasks.assets.terrains import ROUGH_BLIND_TERRAINS_CFG, ROUGH_TERRAINS_CFG


##
# Scene definition
##


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_BLIND_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",  # 发生碰撞时的 摩擦系数的结合方式，默认为average
            restitution_combine_mode="multiply",  # 发生碰撞时的 弹性系数的结合方式，默认为average
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',  #  isaacsim 4.5: attach_yaw_only=True/False ==> isaacsim 5.0: ray_alignment='yaw'/'base'
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.4, 0.2]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    lidar_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.10, 0.0, 0.082), rot=(1.0, 0.0, 0.0, 0.0)),
        ray_alignment='base',
        pattern_cfg=patterns.LidarPatternCfg(
            channels=32, vertical_fov_range=(-7.0, 52.0), horizontal_fov_range=(-180, 180.0), horizontal_res=1.3
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
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
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        lidar_scan = ObsTerm(
            func=mdp.height_map_lidar,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner"), "offset": 0.0},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-10.0, 10.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ProprioCfg(ObsGroup):
        """Observations for proprioceptive group."""

        # observation terms
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_body_material = EventTerm(  # 随机更改env所有部位的 摩擦系数、弹性系数
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.0),
            "dynamic_friction_range": (0.1, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass = EventTerm(  # 随机更改env的指定部位（一般为base）的质量（ ± ）
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_rigid_body_inertia = EventTerm(  # 随机改变env所有部位的惯性张量（ * ）
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )

    randomize_rigid_body_com = EventTerm(  # 随机更改env的指定部位（一般为base）的质心偏移（xyz方向）
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(  # 随机给env的指定部位（一般为base）施加随机扰动力 和 扭矩
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_reset_joints = EventTerm(  # 随机更改env的 默认关节位置、速度
        func=mdp.reset_joints_by_scale,  # （ * ）
        # func=mdp.reset_joints_by_offset,  # （ ± ）
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # by scale
            "velocity_range": (0.0, 0.0),
            # "position_range": (-0.2, 0.2),  # by offset
            # "velocity_range": (-0.5, 0.5),
        },
    )

    randomize_actuator_gains = EventTerm(  # 随机更改env所有关节的 刚度和阻尼系数（ * ）
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_reset_base = EventTerm(  # 随机更改env的base 位置、速度（ ± ）
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(  # 随机给base在水平方向上施加一个线速度
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # General
    is_terminated = RewTerm(
        func=mdp.is_terminated, weight=0.0
    )

    # Root penalties
    lin_vel_z_l2 = RewTerm(  # (upward)
        func=mdp.lin_vel_z_l2, weight=-0.0
    )
    ang_vel_xy_l2 = RewTerm(  # (upward)
        func=mdp.ang_vel_xy_l2, weight=-0.0
    )
    flat_orientation_l2 = RewTerm(  # (upward)
        func=mdp.flat_orientation_l2, weight=-0.0
    )
    base_height_l2 = RewTerm(  # (upward)
        func=mdp.base_height_l2, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewTerm(  # base 的线加速度
        func=mdp.body_lin_acc_l2, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="")},
    )

    # Joint penalties
    joint_torques_l2 = RewTerm(  # 关节实际扭矩的 平方和
        func=mdp.joint_torques_l2, weight=-0.0,
        params = {"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_l2 = RewTerm(  # 关节速度的 平方和
        func=mdp.joint_vel_l2, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_acc_l2 = RewTerm(  # 关节加速度的 平方和
        func=mdp.joint_acc_l2, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )

    # 关节位置 与 默认关节位置的 L1偏差
    def create_joint_deviation_l1_rewTerm(self, attr_name, weight, joint_names_pattern):
        rew_term = RewTerm(
            func=mdp.joint_deviation_l1, weight=weight,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=joint_names_pattern)},
        )
        setattr(self, attr_name, rew_term)
    # 惩罚 关节位置 与 默认位置的 偏差（低cmd时权重大）(upward)
    #   1. commands > 0.1 || base前进线速度 > 0.5的情况下，关节位置与默认位置的偏差 * 1.0
    #   2. commands <= 0.1 && base前进线速度 <= 0.5的情况下，关节位置与默认位置的偏差 * 5.0
    joint_pos_deviation = RewTerm(
        func=mdp.joint_pos_deviation, weight=-0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "velocity_threshold": 0.5,
            "stand_still_scale": 5.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    stand_still_without_cmd = RewTerm(  # commands < 0.1 的情况下，关节位置与默认位置的偏差 (upward)
        func=mdp.stand_still_without_cmd, weight=-0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")}
    )
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"), "soft_ratio": 1.0},
    )
    # 惩罚四轮在空中或者静立时的关节速度：
    #   1. command > 0.1 && vel > 0.5时，惩罚空中轮子的关节速度；
    #   2. command <= 0.1 || vel <= 0.5时，即静立时，惩罚四轮的关节速度
    wheel_vel_lowCmd = RewTerm(
        func=mdp.wheel_vel_lowCmd, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=""),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    joint_power = RewTerm(  # 关节实际扭矩 * 关节速度 之和
        func=mdp.joint_power, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_mirror = RewTerm(  # 对称关节的位置偏差 (upward)
        func=mdp.joint_mirror, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )

    # Action penalties
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits, weight=-0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_rate_l2 = RewTerm(  # 惩罚 相邻 actions 的变化（差值的平方和）
        func=mdp.action_rate_l2, weight=-0.0
    )
    action_smoothness = RewTerm(  # 惩罚 相邻 actions 的变化（差值的L2范数）
        func=mdp.action_smoothness, weight=-0.0,
    )
    action_mirror = RewTerm(  # 对称关节的 actions 偏差 (upward)
        func=mdp.action_mirror, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [["FR.*", "RL.*"], ["FL.*", "RR.*"]],
        },
    )
    action_sync = RewTerm(  # 相同部位关节的 actions 方差之和，即让相同部位的关节同步 (upward)
        func=mdp.action_sync, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_groups": [
                ["FR_hip_joint", "FL_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                ["FR_thigh_joint", "FL_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"],
                ["FR_calf_joint", "FL_calf_joint", "RL_calf_joint", "RR_calf_joint"],
            ],
        },
    )

    # Contact sensor
    undesired_contacts = RewTerm(  # 部位接触力 > 1.0 的数量 (upward)
        func=mdp.undesired_contacts, weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(  # 部位接触力 与 100N 的偏差，即鼓励 < 100N
        func=mdp.contact_forces, weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 100.0,
        },
    )

    # Velocity-tracking rewards
    track_lin_vel_xy_exp = RewTerm(  # commands >= 0.1 情况下，commands 与 base线速度 的偏差 (upward)
        func=mdp.track_lin_vel_xy_exp, weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(  # commands 与 base角速度 的偏差 (upward)
        func=mdp.track_ang_vel_z_exp, weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    # Others
    feet_air_time = RewTerm(  # commands_vel > 0.1情况下，接触时间和空中时间接近 0.5s (upward)
        func=mdp.feet_air_time, weight=0.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 0.5,
        }
    )
    feet_air_time_variance = RewTerm(  # (upward)
        func=mdp.feet_air_time_variance, weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        }
    )
    feet_gait = RewTerm(  # commands > 0.1 且 base的xy线速度 > 0.5情况下，同步脚 和 异步脚的奖励值（考虑 base 的 z方向）
        func=mdp.GaitReward, weight=0.0,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("", ""), ("", "")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )
    feet_contact = RewTerm(  # 有 commands 情况下，惩罚足部触地的个数 != 2 (upward)
        func=mdp.feet_contact, weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )
    feet_contact_without_cmd = RewTerm(  # commands < 0.1 情况下，奖励足部触地的个数 (upward)
        func=mdp.feet_contact_without_cmd, weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "command_name": "base_velocity",
        },
    )
    feet_stumble = RewTerm(  # xy方向接触力 > 4 * z方向接触力，惩罚接触到竖直面 (upward)
        func=mdp.feet_stumble, weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )
    feet_slide = RewTerm(  # 足部触地时，足部相对base的xy方向线速度 的范数 (upward)
        func=mdp.feet_slide, weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
        },
    )
    feet_height = RewTerm(  # commands > 0.1 情况下，足部xy线速度 * (足部高度 - 目标高度) (upward)
        func=mdp.feet_height, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )
    feet_height_body = RewTerm(  # commands > 0.1 情况下，足部xy线速度 * (足部相对base高度 - 目标高度) (upward)
        func=mdp.feet_height_body, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "tanh_mult": 2.0,
            "target_height": -0.3,
            "command_name": "base_velocity",
        },
    )
    feet_distance_y_exp = RewTerm(  # 足部相对base的y方向位置 与 期望位置 的误差的exp (upward)
        func=mdp.feet_distance_y_exp, weight=0.0,
        params={
            "std": math.sqrt(0.25),
            "asset_cfg": SceneEntityCfg("robot", body_names=""),
            "stance_width": float,  # 期望的足部相对base的y方向位置
        },
    )
    # 1 - 重力投影z轴，鼓励base正置
    upward = RewTerm(
        func=mdp.upward, weight=0.0
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

    # drop
    # base_drop = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": -20},
    # )

    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    lin_vel_cmd_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={
            "reward_term_name": "track_lin_vel_xy_exp",
            "vel_range_multiplier": (0.2, 1.0),
        }
    )



##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings (sim : isaaclab/sim/simlation_cfg.py/SimulationCfg())
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        if self.scene.height_scanner_base is not None:
            self.scene.height_scanner_base.update_period = self.sim.dt * self.decimation
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

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)