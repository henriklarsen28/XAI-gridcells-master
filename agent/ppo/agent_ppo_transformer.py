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

import env.continuous.register_env as register_env
from env import SunburstMazeContinuous, SunburstMazeDiscrete
from utils.state_preprocess import state_preprocess

# Define the CSV file path relative to the project root
map_path_train = os.path.join(
    project_root, "env/random_generated_maps/goal/large/map_circular_4_19.csv"
)
map_path_test = os.path.join(project_root, "env/map_v0/map_open_doors_90_degrees.csv")
map_path_test_2 = os.path.join(
    project_root, "env/map_v0/map_open_doors_horizontal_v2.csv"
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)  # Was faster with cpu??? Loading between cpu and mps is slow maybe
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
        # self.save_path = config["save_path"]
        self.save_interval = config["save_interval"]

        self.max_steps = config["max_steps_per_episode"]
        self.render_mode = config["render_mode"]
        self.render_fps = config["render_fps"]

        self.rewards = config["rewards"]
        self.random_start_position = config["random_start_position"]
        self.random_goal_position = config["random_goal_position"]

        self.fov = config["fov"]
        self.ray_length = config["ray_length"]
        self.number_of_rays = config["number_of_rays"]

        map_path = map_path_train
        if not self.train_mode:
            map_path = map_path_test

        # Define Env
        """self.env = SunburstMazeContinuous(
            maze_file=map_path,
            render_mode=render_mode,
            max_steps_per_episode=self.max_steps,
            random_start_position=self.random_start_position,
            rewards=self.rewards,
            fov=self.fov,
            ray_length=self.ray_length,
            number_of_rays=self.number_of_rays,
        )

        self.env.metadata["render_fps"] = (
            self.render_fps
        )  # For max frame rate make it 0"""
        self.env = gym.make(
            "SunburstMazeContinuous-v0",
            maze_file=map_path,
            max_steps_per_episode=self.max_steps,
            render_mode=self.render_mode,
            random_start_position=self.random_start_position,
            random_goal_position=self.random_goal_position,
            rewards=self.rewards,
            fov=self.fov,
            ray_length=self.ray_length,
            number_of_rays=self.number_of_rays,
        )
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
        "ray_length": 15,
        "number_of_rays": 40,
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
        # "save_path": f"/sunburst_maze_{map_version}",
        "loss_function": "mse",
        "learning_rate": 3e-4,
        "batch_size": 2000,
        #"mini_batch_size": 750,
        "n_mini_batches": 4,
        "optimizer": "adam",
        "PPO": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_updates_per_iteration": 8,  # hard update of the target model
            "clip": 0.2,
            "clip_grad_normalization": 0.5,
            "policy_kl_range": 0.0008,
            "policy_params": 5,
        },
        "map_path": map_path_train,
        "max_steps_per_episode": 500,
        "random_start_position": True,
        "random_goal_position": False,
        "rewards": {
            "is_goal": 10,
            "hit_wall": -0.001,
            "has_not_moved": -0.005,
            "new_square": 0.0,
            "max_steps_reached": -0.025,
            "penalty_per_step": -0.00002,
            "number_of_squares_visible": 0,
            "goal_in_sight": 0.001,
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
        "save_interval": 25,
        "render_fps": 5,
        "fov": fov_config["fov"],
        "ray_length": fov_config["ray_length"],
        "number_of_rays": fov_config["number_of_rays"],
        "transformer": {
            "sequence_length": 20,
            "n_embd": 196,
            "n_head": 4,
            "n_layer": 3,
            "dropout": 0.2,
            "decouple_positional_embedding": False,
        },
        "entropy": {"coefficient": 0.03, "min": 0.0001, "step": 100_000},
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
