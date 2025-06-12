import gymnasium as gym

from . import agents


##
# Register Gym environments.
##


# --- Flat
gym.register(
    id="LeggedLoco-AlienGo-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_flat_env_cfg:AlienGoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_flat_env_cfg:AlienGoFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)


gym.register(
    id="LeggedLoco-AlienGo-Flat-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_flat_lidar_env_cfg:AlienGoFlatLidarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Flat-Lidar-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_flat_lidar_env_cfg:AlienGoFlatLidarEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)


# --- Rough (stairs)
gym.register(
    id="LeggedLoco-AlienGo-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_rough_env_cfg:AlienGoRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_rough_env_cfg:AlienGoRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)


gym.register(
    id="LeggedLoco-AlienGo-Rough-Lidar-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_rough_lidar_env_cfg:AlienGoRoughLidarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-AlienGo-Rough-Lidar-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.aliengo_rough_lidar_env_cfg:AlienGoRoughLidarEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)