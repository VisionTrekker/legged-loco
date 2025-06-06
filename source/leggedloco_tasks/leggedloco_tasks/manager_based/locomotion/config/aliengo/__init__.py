import gymnasium as gym

from . import agents


##
# Register Gym environments.
##

gym.register(
    id="LeggedLoco-AlienGo-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_low_base_cfg:AlienGoBaseRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_low_base_cfg:AlienGoBaseRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)


gym.register(
    id="LeggedLoco-AlienGo-Flat-Lidar",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_low_base_lidar_cfg:AlienGoBaseLidarRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Flat-Lidar-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_low_base_lidar_cfg:AlienGoBaseLidarRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)


gym.register(
    id="LeggedLoco-AlienGo-Rough-Lidar",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_low_vision_cfg:AlienGoVisionRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoVisionRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Rough-Lidar-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_low_vision_cfg:AlienGoVisionRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoVisionRoughPPORunnerCfg",
    },
)