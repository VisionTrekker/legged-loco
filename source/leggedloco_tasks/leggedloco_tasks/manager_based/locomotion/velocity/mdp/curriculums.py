# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy_exp",
    vel_range_multiplier: tuple[float, float] = (0.2, 1.0),
) -> torch.Tensor:
    command_ranges = env.command_manager.get_term("base_velocity").cfg.ranges

    # compute the inital and final velocity range only on 1st episode
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(command_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(command_ranges.lin_vel_y, device=env.device)

        env._initial_vel_x = env._original_vel_x * vel_range_multiplier[0]
        env._initial_vel_y = env._original_vel_y * vel_range_multiplier[0]

        env._final_vel_x = env._original_vel_x * vel_range_multiplier[1]
        env._final_vel_y = env._original_vel_y * vel_range_multiplier[1]

        # Initialize command ranges to initial values
        command_ranges.lin_vel_x = env._initial_vel_x.tolist()
        command_ranges.lin_vel_y = env._initial_vel_y.tolist()

    if env.common_step_counter % env.max_episode_length == 0:
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        episode_sums = env.reward_manager._episode_sums[reward_term_name]  # 每个env在当前episode的所有奖励之和
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > reward_term_cfg.weight * 0.8:  # 平均奖励 > 80% * 该奖励的最大可能奖励
            new_vel_x = torch.tensor(command_ranges.lin_vel_x, device=env.device) + delta_command  # 线速度范围 +- 0.1
            new_vel_y = torch.tensor(command_ranges.lin_vel_y, device=env.device) + delta_command

            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            command_ranges.lin_vel_x = new_vel_x.tolist()
            command_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(command_ranges.lin_vel_x[1], device=env.device)