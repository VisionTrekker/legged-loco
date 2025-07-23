# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Boston Dynamics robot.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from leggedloco_tasks.assets import ISAACLAB_ASSETS_DATA_DIR


##
# Configuration - Actuators.
##

# DCMotor
ALIENGO_ACTUATOR_DCMOTOR_HIP_CFG = DCMotorCfg(
    joint_names_expr=[".*_hip_joint"],
    effort_limit=44.0,
    saturation_effort=44.0,
    velocity_limit=20.0,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
)
ALIENGO_ACTUATOR_DCMOTOR_THIGH_CFG = DCMotorCfg(
    joint_names_expr=[".*_thigh_joint"],
    effort_limit=44.0,
    saturation_effort=44.0,
    velocity_limit=20.0,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
)
ALIENGO_ACTUATOR_DCMOTOR_CALF_CFG = DCMotorCfg(
    joint_names_expr=[".*_calf_joint"],
    effort_limit=55.0,
    saturation_effort=55.0,
    velocity_limit=15.0,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
)

# DelayedPDActuator
ALIENGO_ACTUATOR_DELAYEDPD_HIP_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_hip_joint"],
    effort_limit=44.0,
    velocity_limit=20.0,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
    min_delay=4,
    max_delay=4,
)
ALIENGO_ACTUATOR_DELAYEDPD_THIGH_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_thigh_joint"],
    effort_limit=44.0,
    velocity_limit=20.0,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
    min_delay=4,
    max_delay=4,
)
ALIENGO_ACTUATOR_DELAYEDPD_CALF_CFG = DelayedPDActuatorCfg(
    joint_names_expr=[".*_calf_joint"],
    effort_limit=55.0,
    velocity_limit=15.0,
    stiffness=40.0,
    damping=2.0,
    friction=0.0,
    min_delay=4,
    max_delay=4,
)


##
# Configuration
##

UNITREE_ALIENGO_DCMOTOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.1/Isaac/Robots/Unitree/aliengo/aliengo.usd",
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/Aliengo/aliengo.usd",
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
        "hip": ALIENGO_ACTUATOR_DCMOTOR_HIP_CFG,
        "thigh": ALIENGO_ACTUATOR_DCMOTOR_THIGH_CFG,
        "calf": ALIENGO_ACTUATOR_DCMOTOR_CALF_CFG,
    },
)
"""Configuration of Unitree AlienGo using DC-Motor actuator model."""


UNITREE_ALIENGO_DELAYEDPD_CFG = UNITREE_ALIENGO_DCMOTOR_CFG.replace(
    actuators={
        "hip": ALIENGO_ACTUATOR_DELAYEDPD_HIP_CFG,
        "thigh": ALIENGO_ACTUATOR_DELAYEDPD_THIGH_CFG,
        "calf": ALIENGO_ACTUATOR_DELAYEDPD_CALF_CFG,
    }
)
"""Configuration of Unitree AlienGo using DelayedPD actuator model."""