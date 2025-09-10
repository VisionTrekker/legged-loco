import os

from .wrappers import RslRlVecEnvHistoryWrapper

ASSETS_DIR = os.path.abspath("assets")

__all__ = [
    "ASSETS_DIR",
    "RslRlVecEnvHistoryWrapper",
]
