import os
import sys

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)


import math

import gymnasium as gym
import numpy as np
import torch

from ppo import PPO_agent
from env import SunburstMazeContinuous, SunburstMazeDiscrete
from utils.state_preprocess import state_preprocess

# Define the CSV file path relative to the project root
map_path_train = os.path.join(project_root, "env/map_v0/map_closed_doors_left.csv")
map_path_train_2 = os.path.join(project_root, "env/map_v0/map_open_doors_vertical.csv")
map_path_train_3 = os.path.join(project_root, "env/map_v0/map_no_doors.csv")
map_path_test = os.path.join(project_root, "env/map_v0/map_open_doors_90_degrees.csv")
map_path_test_2 = os.path.join(
    project_root, "env/map_v0/map_open_doors_horizontal_v2.csv"
)

# device = torch.device(
#    "mps" if torch.backends.mps.is_available() else "cpu"
# )  # Was faster with cpu??? Loading between cpu and mps is slow maybe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

print(f"Device: {device}", flush=True)

# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)

# For cuda seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def salt_and_pepper_noise(matrix, prob=0.1):

    noisy_matrix = matrix.clone()
    noise = torch.rand(*matrix.shape)
    noisy_matrix[noise < prob / 2] = 1  # Add "salt"
    noisy_matrix[noise > 1 - prob / 2] = 0  # Add "pepper"
    return noisy_matrix


class Model_TrainTest:
    def __init__(self, config):

        # Define RL parameters
        self.train_mode = config["train_mode"]
        self.policy_load_path = config["policy_load_path"]
        self.critic_load_path = config["critic_load_path"]
        self.save_path = config["save_path"]
        self.save_interval = config["save_interval"]

        self.clip_grad_normalization = config["clip_grad_normalization"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.max_steps = config["max_steps_per_episode"]
        self.render = config["render"]
        self.render_mode = config["render_mode"]
        self.render_fps = config["render_fps"]

        self.rewards = config["rewards"]
        self.random_start_position = config["random_start_position"]
        self.observation_space = config["observation_space"]

        self.fov = config["fov"]
        self.ray_length = config["ray_length"]
        self.number_of_rays = config["number_of_rays"]

        self.transformer = config["transformer"]
        self.sequence_length = self.transformer["sequence_length"]
        map_path = map_path_train
        if not self.train_mode:
            map_path = map_path_test

        # Define Env
        self.env = SunburstMazeContinuous(
            maze_file=map_path,
            render_mode=render_mode,
            max_steps_per_episode=self.max_steps,
            random_start_position=self.random_start_position,
            rewards=self.rewards,
            observation_space=self.observation_space,
            fov=self.fov,
            ray_length=self.ray_length,
            number_of_rays=self.number_of_rays,
        )

        self.env.metadata["render_fps"] = (
            self.render_fps
        )  # For max frame rate make it 0
        # self.env = gym.make('Pendulum-v1', render_mode=self.render_mode)

        self.agent = PPO_agent(
            env=self.env,
            device=device,
            config=config,
        )

    def train(self):
        """
        Reinforcement learning training loop.
        """
        self.agent.learn(20_000_000)

    def test(self, max_episodes=100):
        """
        Reinforcement learning testing loop.
        """
        self.agent.load_model(self.policy_load_path, self.critic_load_path)
        self.agent.rollout(max_episodes, render=True)


if __name__ == "__main__":
    # Parameters:

    train_mode = True

    render = True
    render_mode = "human"

    if train_mode:
        render_mode = "rgb_array" if render else None

    map_version = map_path_test.split("/")[-2]

    # Read the map file to find the number of states
    # num_states = get_num_states(map_path_train)

    fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 10,
        "number_of_rays": 100,
    }

    # Parameters
    config = {
        "train_mode": train_mode,
        "map_path_train": map_path_train,
        "render": render,
        "render_mode": render_mode,
        "model_name": "vivid-firebrand-872",
        "policy_load_path": f"./model/transformers/seq_len_45/model_vivid-firebrand-872/sunburst_maze_map_v0_5100.pth",
        "critic_load_path": "/model/transformers/ppo/model_vivid-firebrand-872/sunburst_maze_map_v0_5100.pth",
        "save_path": f"/sunburst_maze_{map_version}",
        "loss_function": "mse",
        "learning_rate": 3e-4,
        "batch_size": 2000,
        "mini_batch_size": 64,
        "optimizer": "adam",
        "gamma": 0.99,
        # "gae_lambda": 0.95,
        "map_path": map_path_train,
        "n_updates_per_iteration": 10,  # hard update of the target model
        "max_steps_per_episode": 500,
        "random_start_position": True,
        "rewards": {
            "is_goal": 2,
            "hit_wall": -0.01,
            "has_not_moved": -0.005,
            "new_square": 0.0025,
            "max_steps_reached": -0.025,
            "penalty_per_step": -0.0002,
            "number_of_squares_visible": 0,
            "goal_in_sight": 0.1,
            "is_false_goal": -0.01,
            # and the proportion of number of squares viewed (set in the env)
        },
        # TODO
        "observation_space": {
            "position": True,
            "orientation": True,
            "last_known_steps": 0,
            "salt_and_pepper_noise": 0,
        },
        "save_interval": 1,
        "render_fps": 5,
        "clip_grad_normalization": 3,
        "clip": 0.2,
        "fov": fov_config["fov"],
        "ray_length": fov_config["ray_length"],
        "number_of_rays": fov_config["number_of_rays"],
        "transformer": {
            "sequence_length": 45,
            "n_embd": 128,
            "n_head": 8,
            "n_layer": 2,
            "dropout": 0.3,
            "decouple_positional_embedding": False,
        },
    }

    # Run
    DRL = Model_TrainTest(config)
    # Train
    if train_mode:
        DRL.train()
        # DRL.train_from_model()
    else:
        # Test
        # DRL.test(max_episodes=config["total_episodes"])
        DRL.test(max_episodes=100)
