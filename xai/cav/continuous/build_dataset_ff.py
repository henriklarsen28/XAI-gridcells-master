import ast
import copy
import json
import math
import os
import random as rd
import sys
from collections import deque

import pandas as pd
import torch

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

sys.path.append(project_root)

from concept_definition_ff import Concepts

from agent.ppo_ff.network_policy import FeedForwardNNPolicy
from env import SunburstMazeContinuous
from utils.calculate_fov import calculate_fov_matrix_size
from utils.sequence_preprocessing import add_to_sequence, padding_sequence
from utils.state_preprocess import state_preprocess
from xai.cav.continuous.eval_policy import eval_policy
from xai.cav.process_data import (
    build_random_dataset,
    find_model_files,
    grid_observation_dataset,
    save_config,
    save_to_csv,
    shuffle_and_trim_datasets,
    split_dataset_into_train_test,
)

device = torch.device("cpu")

rewards =  {
            "is_goal": 2,
            "hit_wall": -0.01,
            "has_not_moved": -0.005,
            "new_square": 0.0025,
            "max_steps_reached": -0.025,
            "penalty_per_step": -0.0002,
            "number_of_squares_visible": 0,
            "goal_in_sight": 0.1,
			"is_false_goal": -0.01,
	}
fov_config = {
            "fov": math.pi / 1.5,
            "ray_length": 10,
            "number_of_rays": 100,
            }

hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 250, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }


# Parameters
config = {
    "model_path": "../../../agent/ppo/models/feed-forward/colorful-sunset-826/actor",
    
    "model_name": "colorful-sunset-826", # TODO: change to the correct model name
    "model_episodes": [675, 1500, 2925], # TODO: change to the correct model episodes

    "env_name": "map_two_rooms_18_19", #TODO: change to the correct env name
    "env_path": "../../../env/random_generated_maps/goal/large/map_two_rooms_18_19.csv", # TODO: Change to the correct path for what the model was trained on
    # "env_path": "../../../env/map_v0/map_open_doors_horizontal.csv",
    "cav": {
        "dataset_max_length" : 1500
    },

    "grid_length": 4,
}

env = SunburstMazeContinuous(
    maze_file=config['env_path'],
    render_mode="human",
    max_steps_per_episode=hyperparameters["max_timesteps_per_episode"],
    random_start_position=True,
    rewards=rewards,
    fov=fov_config["fov"],
    ray_length=fov_config["ray_length"],
    number_of_rays=fov_config["number_of_rays"],
    grid_length=config["grid_length"]
)


def build_csv_dataset(actor_model_paths: list, dataset_path: str, dataset_subfolder = ''):
    # Load early agent data
    actor_model = actor_model_paths[0] # TODO: set up actor model paths


    # If the actor model is not specified, then exit
    if actor_model_paths == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    print(f"Testing {actor_model}", flush=True)

    con = Concepts(
        grid_pos_to_id=env.env_grid, # TODO: Build grid layout in continous environment
    )

    con.clear_datasets()

    # Extract out dimensions of observation and action spaces
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Build our policy the same way we build our actor model in PPO
    """model = torch.load(actor_model, map_location=device)

    for name, param in model.items():
        print(name, len(param))

    sys.exit()"""

    policy = FeedForwardNNPolicy(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model, map_location=device))

    # Evaluate policy
    collected_observations = eval_policy(policy=policy, env=env, device=device, render=True)

    #TODO: Update model

    for observation, position in collected_observations:
        # print(len(observation_sequence))
        if rd.random() > 0.4:
            con.in_grid_square(observation, position)


    path = os.path.join(dataset_path, dataset_subfolder)

    if os.path.exists(path) == False:
        os.makedirs(path)
    for key, val in con.datasets.items():
        filename = key
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_val:
                    filename = str(key) + '_' + str(sub_key)
                    data_preprocessed = shuffle_and_trim_datasets(sub_val)
                    save_to_csv(data_preprocessed, filename, path)
        else:
            if val:
                data_preprocessed = shuffle_and_trim_datasets(val)
                save_to_csv(data_preprocessed, filename, path)

    save_config(dataset_path)


def main():
    #model_path = os.path.join(config["model_path"], config["model_name"])
    model_files = find_model_files(config["model_path"], config["model_episodes"])
    dataset_path = os.path.join('./dataset/', config["model_name"], config["env_name"])

    dataset_directory_train = f"./dataset/{config['model_name']}/{config['env_name']}/train"
    dataset_directory_test = f"./dataset/{config['model_name']}/{config['env_name']}/test"

    if not os.path.exists(dataset_directory_train):
        os.makedirs(dataset_directory_train, exist_ok=True)
    if not os.path.exists(dataset_directory_test):
        os.makedirs(dataset_directory_test, exist_ok=True)

    print("model files:", model_files)

    build_csv_dataset(model_files, dataset_path, 'raw_data')
    build_random_dataset(dataset_path, "raw_data")
    split_dataset_into_train_test(dataset_path, ratio = 0.8)
    grid_observation_dataset(dataset_path, 'raw_data', model_name=config["model_name"], map_name=config["env_name"]) # specifically for grid layout concept

if __name__ == "__main__":
    main()
