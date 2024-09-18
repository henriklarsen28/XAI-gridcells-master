from gymnasium.envs.registration import register

register(
    id="SunburstMazeDiscrete-v1",
    entry_point="env.sunburstmaze_discrete:SunburstMazeDiscrete"
)