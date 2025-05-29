import gymnasium as gym

from .g1_low_base_cfg import G1BaseRoughEnvCfg, G1BaseRoughEnvCfg_PLAY, G1RoughPPORunnerCfg
from .g1_low_vision_cfg import G1VisionRoughEnvCfg, G1VisionRoughEnvCfg_PLAY, G1VisionRoughPPORunnerCfg

##
# Register Gym environments.
##


gym.register(
    id="g1_base",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1BaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": G1RoughPPORunnerCfg,
    },
)


gym.register(
    id="g1_base_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1BaseRoughEnvCfg,
        "rsl_rl_cfg_entry_point": G1RoughPPORunnerCfg,
    },
)


gym.register(
    id="g1_vision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1VisionRoughEnvCfg,
        "rsl_rl_cfg_entry_point": G1VisionRoughPPORunnerCfg,
    },
)


gym.register(
    id="g1_vision_play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1VisionRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1VisionRoughPPORunnerCfg,
    },
)
