from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from eilab_tasks.assets import ISAACLAB_ASSETS_DATA_DIR
from eilab_tasks.manager_based.navigation.navigation_env_cfg import (
    NavigationRoughEnvCfg,
    ActionsCfg
)
from .rough_env_cfg import Go2WNavigationRoughEnvCfg

import eilab_tasks.manager_based.navigation.mdp as mdp


##
# Scene definition
##


from eilab_tasks.manager_based.locomotion.velocity.config.wheeled.go2w.flat_env_cfg import Go2WFlatEnvCfg
LOW_LEVEL_ENV_CFG = Go2WFlatEnvCfg()


@configclass
class Go2WNavigationFlatEnvCfg(Go2WNavigationRoughEnvCfg):
    """Configuration for the Go2W navigation environment."""

    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


class Go2WNavigationFlatPlayEnvCfg(Go2WNavigationFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.terrain_generator.difficulty_range = (0.6, 1.0)

        # ------------------------------Observations------------------------------
        # disable randomization for play
        self.observations.policy.enable_corruption = False
