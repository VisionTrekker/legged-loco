# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from leggedloco_tasks.assets import ISAACLAB_ASSETS_DATA_DIR
from leggedloco_tasks.assets.robots.utils.usd_converter import (
    mjcf_to_usd,
    spawn_from_lazy_usd,
    urdf_to_usd,
    xacro_to_usd,
)


##
# Configuration
##


UNITREE_ALIENGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/aliengo_description/urdf/aliengo_bigAng.urdf",  # aliengo / aliengo_bigAng
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/aliengo_description/usd/aliengo_bigAng.usd",  # aliengo / aliengo_bigAng
            merge_joints=True,
            fix_base=False,
        ),
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.50),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=35.2,  # 44.0 / 35.2
            saturation_effort=35.2,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=35.2,  # 44.0 / 35.2
            saturation_effort=35.2,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=44.4,  # 55.0 / 44.4
            saturation_effort=44.4,
            velocity_limit=15.0,
            stiffness=40.0,
            damping=2.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree AlienGo using DC-Motor actuator model.
"""


UNITREE_GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        func=spawn_from_lazy_usd,
        usd_path=urdf_to_usd(  # type: ignore
            file_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/go2w_description/urdf/go2w_description.urdf",
            output_usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/go2w_description/usd/go2w_description.usd",
            merge_joints=True,
            fix_base=False,
        ),
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
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2W using DC motor.
"""