import gymnasium as gym

from . import agents


##
# Register Gym environments.
##


"""
Navigation
"""


# --- Flat ---
# blind
gym.register(
    id="EiLab-Isaac-Navigation-Flat-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2WNavigationFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)

gym.register(
    id="EiLab-Isaac-Navigation-Flat-Go2W-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2WNavigationFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)


# --- Rough ---
# blind
gym.register(
    id="EiLab-Isaac-Navigation-Rough-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2WNavigationRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WRoughPPORunnerCfg",
    },
)

gym.register(
    id="EiLab-Isaac-Navigation-Rough-Go2W-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2WNavigationRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WRoughPPORunnerCfg",
    },
)
