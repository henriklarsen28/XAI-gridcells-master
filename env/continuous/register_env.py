import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from gymnasium.envs.registration import register
import gymnasium as gym


register(
    id="SunburstMazeContinuous-v0",  # Unique ID for your environment
    entry_point="env.continuous.sunburstmaze_continuous:SunburstMazeContinuous",  # Path to the class
)

