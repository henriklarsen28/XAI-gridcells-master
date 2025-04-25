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

    goal_area_regular = [5, 11]
    goal_area_rotated = [34, 35]

    model_steps = {
        int(re.search(r"policy_network_(\d+)\.pth", path).group(1))
        for path in actor_model_paths
    }
    model_steps = sorted(list(model_steps))
    print("Model steps: ", model_steps)

    goal_visitations_regular = {model: 0 for model in model_steps}
    goal_visitations_rotated = {model: 0 for model in model_steps}

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

        for position, model_num in collected_observations:
            print("Model num: ", model_num)
            for position_step in position:
                if rd.random() > 0.4:
                    grid_id = con.in_grid_square(None, position_step)
                    if model_num in model_steps:
                        if grid_id in goal_area_regular:
                            goal_visitations_regular[model_num] += 1
                        elif grid_id in goal_area_rotated:
                            goal_visitations_rotated[model_num] += 1

    # Save the goal visitations to a file
    with open(os.path.join("goal_visitations_circular.json"), "w") as f:
        json.dump(
            {
                "regular": goal_visitations_regular,
                "rotated": goal_visitations_rotated,
            },
            f,
        )

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
