import gymnasium as gym

from . import agents


##
# Register Gym environments.
##


# --- Flat ---
# blind
gym.register(
    id="LeggedLoco-Isaac-Velocity-Flat-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AlienGoFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-Isaac-Velocity-Flat-Play-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AlienGoFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)

# with mid-360
gym.register(
    id="LeggedLoco-Isaac-Velocity-Flat-Lidar-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_lidar_env_cfg:AlienGoFlatLidarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-Isaac-Velocity-Flat-Lidar-Play-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_lidar_env_cfg:AlienGoFlatLidarPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatPPORunnerCfg",
    },
)


# --- Rough (stairs) ---
# blind
gym.register(
    id="LeggedLoco-Isaac-Velocity-Rough-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AlienGoRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-Isaac-Velocity-Rough-Play-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AlienGoRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

# with mid-360
gym.register(
    id="LeggedLoco-Isaac-Velocity-Rough-Lidar-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_lidar_env_cfg:AlienGoRoughLidarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-Isaac-Velocity-Rough-Lidar-Play-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_lidar_env_cfg:AlienGoRoughLidarPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughPPORunnerCfg",
    },
)




"""
Recover
"""


# --- Flat ---
# blind
gym.register(
    id="LeggedLoco-Isaac-Velocity-Flat-Recover-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_recover_env_cfg:AlienGoFlatRecoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatRecoverPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-Isaac-Velocity-Flat-Recover-Play-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_recover_env_cfg:AlienGoFlatRecoverPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoFlatRecoverPPORunnerCfg",
    },
)


# --- Rough (stairs) ---
# blind
gym.register(
    id="LeggedLoco-Isaac-Velocity-Rough-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_recover_env_cfg:AlienGoRoughRecoverEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughRecoverPPORunnerCfg",
    },
)

gym.register(
    id="LeggedLoco-Isaac-Velocity-Rough-Play-AlienGo-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_recover_env_cfg:AlienGoRoughRecoverPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AlienGoRoughRecoverPPORunnerCfg",
    },
)