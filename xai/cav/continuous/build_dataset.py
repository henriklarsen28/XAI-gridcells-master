import ast
import copy
import json
import math
import os
import random as rd
import re
import sys
from collections import deque

import pandas as pd
import torch

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

sys.path.append(project_root)

import gymnasium as gym

from agent.ppo.transformer_decoder_decoupled_policy import TransformerPolicyDecoupled
from env import SunburstMazeContinuous
from utils.calculate_fov import calculate_fov_matrix_size
from utils.sequence_preprocessing import add_to_sequence, padding_sequence
from utils.state_preprocess import state_preprocess
from xai.cav.concept_definition import Concepts
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


def get_grid_data(env_name):
    
    # TWO ROOMS

    if env_name == "two_rooms":
        goal_area = {
            "regular" : [35],
            "rotated" : [30]
        }
    elif env_name == "circular":
        goal_area = {
            "regular" : [4, 5, 10, 11],
            "rotated" : [28, 29, 34, 35]
        }
    else:
        print("Invalid environment name. Exiting.")
        sys.exit(0)


    grid_data = {
        grid_id: {
            model_num: 0
            for model_num in range(1000, 1701, 25)
        }
        for grid_id in range(36)
    }

    return goal_area, grid_data


def get_model_steps(actor_model_paths):
    model_steps = {
        int(re.search(r"policy_network_(\d+)\.pth", path).group(1))
        for path in actor_model_paths
    }

    model_steps = sorted(list(model_steps))
    print("Model steps: ", model_steps)

    return model_steps

def save_movements(env_name, goal_visitations_regular, goal_visitations_rotated, grid_data):
    with open(os.path.join(f"goal_visitations_{env_name}.json"), "w") as f: # change to two rooms
        json.dump(
            {
                "regular": goal_visitations_regular,
                "rotated": goal_visitations_rotated,
            },
            f,
        )
    
    # Save the grid data to a file
    with open(os.path.join("grid_data_two_rooms.json"), "w") as f: # change to two rooms
        json.dump(grid_data, f)


def find_closest_model_step(ep_num, model_steps):
    return min(model_steps, key=lambda x: abs(x - ep_num))


def build_csv_dataset(
    env: SunburstMazeContinuous,
    device,
    config: dict,
    actor_model_paths: list,
    dataset_path: str,
    dataset_subfolder: str = "",
    grid_size: int = None,
    model_files: list = None,
):

    # If the actor model is not specified, then exit
    if actor_model_paths == "":
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # If the actor model is not specified, then exit
    if actor_model_paths == "":
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    con = Concepts(
        grid_pos_to_id=env.env_grid,
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

    policy = TransformerPolicyDecoupled(
        input_dim=obs_dim,
        output_dim=act_dim,
        num_envs=6,
        block_size=config["transformer"]["sequence_length"],
        n_embd=config["transformer"]["n_embd"],
        n_head=config["transformer"]["n_head"],
        n_layer=config["transformer"]["n_layer"],
        dropout=config["transformer"]["dropout"],
        device=device,
    )

    model_steps = get_model_steps(actor_model_paths)
    
    goal_visitations_regular = {model: 0 for model in model_steps}
    goal_visitations_rotated = {model: 0 for model in model_steps}

    env_name = None
    if "two_rooms" in dataset_path:
        env_name = "two_rooms"
    elif "circular" in dataset_path:
        env_name = "circular"
        
    goal_area, grid_data = get_grid_data(env_name)

    # Evaluate policy
    for collected_observations in eval_policy(
        policy=policy,
        actor_model_paths=actor_model_paths,
        env=env,
        sequence_length=config["transformer"]["sequence_length"],
        device=device,
        render=True,
        max_steps=config["max_steps_per_episode"],
    ):
        # print("Collected observations", len(collected_observations), collected_observations[0])

        for _, position, model_num in collected_observations:
            # print("Model num: ", model_num)
            for position_step in position:
                if rd.random() > 0.4:
                    grid_id = con.in_grid_square(None, position_step)
                    if model_num in model_steps:
                        grid_data[grid_id][model_num] += 1
                        if grid_id in goal_area["regular"]:
                            goal_visitations_regular[model_num] += 1
                        elif grid_id in goal_area["rotated"]:
                            goal_visitations_rotated[model_num] += 1
    
    save_movements(env_name, goal_visitations_regular, goal_visitations_rotated, grid_data)

    path = os.path.join(dataset_path, dataset_subfolder)

    max_length = config["cav"]["dataset_max_length"]

    if os.path.exists(path) == False:
        os.makedirs(path)
    for key, val in con.datasets.items():
        filename = key
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_val:
                    filename = str(key) + "_" + str(sub_key)
                    data_preprocessed = shuffle_and_trim_datasets(sub_val, max_length)
                    if len(data_preprocessed) > 1:
                        save_to_csv(data_preprocessed, filename, path)
                    else:
                        print("Not enough data to save to CSV for ", key)
        else:
            if val:
                data_preprocessed = shuffle_and_trim_datasets(val, max_length)
                if len(data_preprocessed) > 1:
                    save_to_csv(data_preprocessed, filename, path)
                else:
                    print("Not enough data to save to CSV for ", key)
    del con.datasets
    del con
    # Save the config as a file for reference
    save_config(dataset_path, config)

    # Using the raw dataset, build a random dataset
    build_random_dataset(dataset_path, dataset_subfolder)

    # Split the dataset into a training and test set
    split_dataset_into_train_test(dataset_path, dataset_subfolder, ratio=0.8)

    # Build a dataset for grid observations
    grid_observation_dataset(
        dataset_path, grid_size
    )  # specifically for grid layout concept

    print("Finished building dataset!")

