import gymnasium as gym

from .go1_low_base_cfg import Go1BaseRoughEnvCfg, Go1BaseRoughEnvCfg_PLAY, Go1RoughPPORunnerCfg
from .go1_low_vision_cfg import Go1VisionRoughEnvCfg, Go1VisionRoughEnvCfg_PLAY, Go1VisionRoughPPORunnerCfg

##
# Register Gym environments.
##


gym.register(
    id="go1_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go1BaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": Go1RoughPPORunnerCfg,
    },
)


gym.register(
    id="go1_base_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go1BaseRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": Go1RoughPPORunnerCfg,
    },
)


gym.register(
    id="go1_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go1VisionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": Go1VisionRoughPPORunnerCfg,
    },
)


gym.register(
    id="go1_vision_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go1VisionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": Go1VisionRoughPPORunnerCfg,
    },
)
