import gymnasium as gym

from .h1_low_base_cfg import H1BaseRoughEnvCfg, H1BaseRoughEnvCfg_PLAY, H1RoughPPORunnerCfg
from .h1_low_vision_cfg import H1VisionRoughEnvCfg, H1VisionRoughEnvCfg_PLAY, H1VisionRoughPPORunnerCfg

##
# Register Gym environments.
##


gym.register(
    id="h1_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1BaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_base_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1BaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1RoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)


gym.register(
    id="h1_vision_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": H1VisionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": H1VisionRoughPPORunnerCfg,
    },
)
