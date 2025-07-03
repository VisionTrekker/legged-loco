from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Contact sensor.
"""


def illegal_contact_delay(env: ManagerBasedRLEnv, contact_threshold: float, sensor_cfg: SceneEntityCfg, episode_length_threshold: float) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    any_force_exceeded = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > contact_threshold, dim=1
    )
    return torch.logical_and(any_force_exceeded, env.episode_length_buf >= episode_length_threshold)