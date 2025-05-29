# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg, CurriculumCfg, RewardsCfg, EventCfg, TerminationsCfg, CommandsCfg

import omni.isaac.leggedloco.leggedloco.mdp as mdp
##
# Pre-defined configs
##
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AlienGoRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):#定义基于近端策略优化（PPO）算法的运行器配置
    num_steps_per_env = 24#每个环境的步数
    max_iterations = 5000# 最大训练迭代次数
    save_interval = 50# 保存模型的间隔迭代次数
    experiment_name = "aliengo_base"# 实验名称
    empirical_normalization = False# 是否使用经验归一化
    policy = RslRlPpoActorCriticCfg(# 策略网络的配置，包括初始噪声标准差、隐藏层维度和激活函数
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(# PPO算法的配置，如价值损失系数、裁剪参数、熵系数、学习率等
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

import os
UNITREE_AlienGo_CFG = ArticulationCfg(# 定义Unitree Go2机器人的配置，包括模型文件路径、初始状态、刚体和关节属性以及执行器设置
    spawn=sim_utils.UsdFileCfg(# 包含机器人模型的USD文件路径、接触传感器设置、刚性和关节属性
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Aliengo/aliengo.usd",
        usd_path=f"{os.getenv('USER_PATH_TO_USD')}/robots/aliengo/aliengo.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(# 机器人的初始位置、关节位置和关节速度
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,# 软关节位置限制因子
    actuators={# 机器人腿部执行器的配置，包括关节名称表达式、力限制、速度限制、刚度、阻尼等
        "base_legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=40.0,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            min_delay=4,
            max_delay=4,
        )
    }
)


##
# Configuration for custom terrains.
##
AlienGo_BASE_TERRAINS_CFG = TerrainGeneratorCfg(# 定义了地形生成器的配置，包括地形大小、边界宽度、行数、列数、缩放比例、斜率阈值以及各种子地形的配置。
    size=(8.0, 8.0),# 地形的尺寸
    border_width=20.0,# 边界宽度
    num_rows=10,# 地形行数和列数
    num_cols=20,
    horizontal_scale=0.1,# 水平和垂直缩放比例
    vertical_scale=0.005,
    slope_threshold=0.75,# 斜率阈值
    use_cache=False,
    sub_terrains={# 包含多种子地形的配置，如金字塔楼梯、倒金字塔楼梯、随机网格、平面、随机粗糙和离散障碍物地形。每个子地形都有其特定的比例、尺寸范围等参数。
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.2),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.2),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.1), platform_width=2.0
        # ),
        ### 只开了这4行
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
        # "gaps": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.1, gap_width_range=(0.5, 1.0), platform_width=2.0
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.2, slope_range=(0.0, 0.02), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.2, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25 ),
        ###新增
        # "init_pos": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.6,
        #     num_obstacles=10,
        #     obstacle_height_mode="fixed",
        #     obstacle_height_range=(0.3, 2.0), obstacle_width_range=(0.4, 1.0),
        #
        #     platform_width=0.0
        # ),
    },
)


##
# Scene definition
##
@configclass
class AlienGoSceneCfg(InteractiveSceneCfg):# 定义了场景的配置，包括地形、机器人、传感器和灯光
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(# 地形导入配置，指定地形类型为生成器，并使用Go2_BASE_TERRAINS_CFG进行地形生成。还包括物理材质视觉材质等设置。
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=AlienGo_BASE_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
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
    robot: ArticulationCfg = UNITREE_AlienGo_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")# 机器人配置，基于UNITREE_GO2_CFG并指定其在场景中的路径

    # sensors
    height_scanner = RayCasterCfg(# 高度扫描传感器的配置，包括位置、偏移、扫描模式等
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.0, 2.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)# 接触力传感器的配置，包括路径、历史长度和是否跟踪空中时间

    # lights
    sky_light = AssetBaseCfg(# 天空光的配置，包括强度和纹理文件
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Rewards
##
@configclass 
class CustomAlienGoRewardsCfg(RewardsCfg):#  定义了自定义的奖励配置
    hip_deviation = RewTerm(# 髋关节偏差奖励项，使用mdp.joint_deviation_l1函数计算，权重为-0.4，针对髋关节
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation = RewTerm(# 其他关节偏差奖励项，同样使用mdp.joint_deviation_l1函数，权重 -0.04，针对大腿和小腿关节。
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )
    base_height = RewTerm(# 基于机器人底盘高度的奖励项，使用mdp.base_height_l2函数，权重 -5.0，目标高度为 0.32。
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.32},
    )
    action_smoothness = RewTerm(# 动作平滑度奖励项，使用mdp.action_smoothness_penalty函数，权重 -0.02。
        func=mdp.action_smoothness_penalty,
        weight=-0.02,
    )
    joint_power = RewTerm(# 关节功率奖励项，使用mdp.power_penalty函数，权重 -2e - 5，针对所有关节。
        func=mdp.power_penalty,
        weight=-2e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.05,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )


##
# Observations
##
@configclass
class ObservationsCfg:# 定义了观测配置，分为策略组（PolicyCfg）、本体感受组（ProprioCfg）和评论家观测组（CriticObsCfg）。
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):# 用于策略网络的观测配置，包括底盘角速度、底盘翻滚俯仰偏航角、速度命令等观测项，并添加噪声。__post_init__方法中设置启用噪声干扰和连接观测项。
        """Observations for policy group."""

        # observation terms (order preserved)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    
    @configclass
    class ProprioCfg(ObsGroup):# 本体感受相关的观测配置，与PolicyCfg类似，但无噪声干扰设置。
        """Observations for proprioceptive group."""

        # observation terms
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # projected_gravity = ObsTerm(
        #     func=mdp.projected_gravity,
        #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # )
        base_rpy = ObsTerm(func=mdp.base_rpy, noise=Unoise(n_min=-0.1, n_max=0.1))
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True
    
    @configclass
    class CriticObsCfg(ObsGroup):# 用于评论家网络的观测配置，包括底盘线速度、角速度、投影重力等观测项，添加噪声，__post_init__方法中设置不启用噪声干扰和连接观测项。
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioCfg = ProprioCfg()
    critic: CriticObsCfg = CriticObsCfg()


##
# Events (for domain randomization)
##
@configclass
class EventCfg:# 定义了用于领域随机化的事件配置。
    """Configuration for events."""

    # startup
    physics_material = EventTerm(# 在启动时随机化机器人刚体物理材质的事件，包括静态摩擦、动态摩擦和恢复系数的范围。
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 4.0),
            "dynamic_friction_range": (0.4, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(# 在启动时随机化机器人底盘质量的事件，通过添加一定范围内的质量。
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    actuator_gains = EventTerm(# 在重置时随机化执行器增益的事件，通过缩放刚度。
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # reset 还包括在重置和间隔时间触发的其他事件，如施加外部力和扭矩、重置底盘和关节状态等。
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class AlienGoBaseRoughEnvCfg(ManagerBasedRLEnvCfg):# 定义了 Go2 机器人在粗糙地形上的速度跟踪强化学习环境的配置。  包含场景、观测、动作、命令、奖励、终止条件、事件和课程学习等配置。 __post_init__方法中设置了模拟的通用参数，如抽取率、渲染间隔、 episode 长度、模拟时间步长等。还对地形、动作、事件、奖励、命令和终止条件等进行了进一步的调整和配置。根据课程学习是否启用，设置地形生成器的课程学习模式。
    """Configuration for the Go2 locomotion velocity-tracking environment."""

    scene: AlienGoSceneCfg = AlienGoSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = CustomAlienGoRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # self.sim.physics_material.static_friction = 1.0
        # self.sim.physics_material.dynamic_friction = 1.0
        # self.sim.physics_material.friction_combine_mode = "multiply"
        # self.sim.physics_material.restitution_combine_mode = "multiply"

        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.04)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # reduce action scale
        self.actions.joint_pos.scale = 0.25

        # event
        # self.events.push_robot = None
        self.events.actuator_gains = None
        self.events.add_base_mass.params["mass_distribution_params"] = (-3.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_calf"# ".*_foot"
        # self.rewards.feet_air_time.weight = 0.01
        self.rewards.undesired_contacts = None
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.dof_pos_limits.weight = -0.0001

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0#0.1 将 10% 的环境配置为相对站立模式，其中速度命令设置为零

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"
        self.events.add_base_mass.params["asset_cfg"].body_names = "base"
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training

        # flat terrain
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # self.curriculum.terrain_levels = None
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.feet_air_time.weight = 0.2

        # update sensor period
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class AlienGoBaseRoughEnvCfg_PLAY(AlienGoBaseRoughEnvCfg):#继承自Go2BaseRoughEnvCfg，为游戏模式定义了特定的配置。 在__post_init__方法中，调用父类的__post_init__方法后，对场景进行了调整，如减少环境数量、调整地形生成器参数以节省内存。 禁用了策略观测中的噪声干扰，移除了一些随机事件，并调整了命令的速度范围。
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.actuator_gains = None

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)