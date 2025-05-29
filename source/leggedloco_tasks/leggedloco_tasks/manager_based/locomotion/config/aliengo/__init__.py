import gymnasium as gym

from .aliengo_low_base_cfg import AlienGoBaseRoughEnvCfg, AlienGoBaseRoughEnvCfg_PLAY, AlienGoRoughPPORunnerCfg
from .aliengo_low_vision_cfg import AlienGoVisionRoughEnvCfg, AlienGoVisionRoughEnvCfg_PLAY, AlienGoVisionRoughPPORunnerCfg

##
# Register Gym environments.
##

gym.register(
    id="aliengo_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AlienGoBaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": AlienGoRoughPPORunnerCfg,
    },
)


gym.register(
    id="aliengo_base_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AlienGoBaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AlienGoRoughPPORunnerCfg,
    },
)

gym.register(
    id="aliengo_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AlienGoVisionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": AlienGoVisionRoughPPORunnerCfg,
    },
)

gym.register(
    id="aliengo_vision_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AlienGoVisionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": AlienGoVisionRoughPPORunnerCfg,
    },
)