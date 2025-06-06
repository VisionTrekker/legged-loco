# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg

ALIENGO_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
)