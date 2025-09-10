import gymnasium as gym

from . import agents


##
# Register Gym environments.
##


"""
Recover
"""


# --- Flat ---
# blind
gym.register(
    id="EiLab-Isaac-Velocity-Flat-Recover-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2WFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)

gym.register(
    id="EiLab-Isaac-Velocity-Flat-Recover-Go2W-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2WFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WFlatPPORunnerCfg",
    },
)


# --- Rough (stairs) ---
# blind
gym.register(
    id="EiLab-Isaac-Velocity-Rough-Recover-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WRoughPPORunnerCfg",
    },
)

gym.register(
    id="EiLab-Isaac-Velocity-Rough-Recover-Go2W-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2WRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WRoughPPORunnerCfg",
    },
)