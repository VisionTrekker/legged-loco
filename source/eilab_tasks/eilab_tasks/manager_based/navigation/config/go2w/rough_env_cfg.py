from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from eilab_tasks.assets import ISAACLAB_ASSETS_DATA_DIR
from eilab_tasks.manager_based.navigation.navigation_env_cfg import (
    NavigationRoughEnvCfg,
    ActionsCfg
)

import eilab_tasks.manager_based.navigation.mdp as mdp


##
# Scene definition
##


from eilab_tasks.manager_based.locomotion.velocity.config.wheeled.go2w.rough_env_cfg import Go2WRoughEnvCfg
LOW_LEVEL_ENV_CFG = Go2WRoughEnvCfg()


##
# MDP settings
##


@configclass
class Go2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    pretrained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Policies/Unitree/go2w/locomotion/blind/policy_roughRecover.jit",
        low_level_decimation=LOW_LEVEL_ENV_CFG.decimation,
        low_level_act_pos=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_act_vel=LOW_LEVEL_ENV_CFG.actions.joint_vel,
        low_level_obs_policy=LOW_LEVEL_ENV_CFG.observations.policy,
        low_level_obs_proprio=LOW_LEVEL_ENV_CFG.observations.proprio if LOW_LEVEL_ENV_CFG.observations.proprio is not None else None,
        low_level_obs_his_len=5,
    )


@configclass
class Go2WNavigationRoughEnvCfg(NavigationRoughEnvCfg):
    """Configuration for the Go2W navigation environment."""

    scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # 因增加 轮足 关节，所以需更新新的 actions
    actions: Go2WActionsCfg = Go2WActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
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

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [LOW_LEVEL_ENV_CFG.base_link_name, ".*_hip"]


class Go2WNavigationRoughPlayEnvCfg(Go2WNavigationRoughEnvCfg):
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
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.2
            # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].step_height_range = (0.16, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.2
            # self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.16, 0.20)
            self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.2
            self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 0.2
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.1
            self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.1

        # ------------------------------Observations------------------------------
        # disable randomization for play
        self.observations.policy.enable_corruption = False
