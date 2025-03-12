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

import gymnasium as gym
from agent.ppo.transformer_decoder_policy import TransformerPolicy
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
 
 
def build_csv_dataset(
        env:SunburstMazeContinuous, 
        device,
        config: dict,
        actor_model_paths: list, 
        dataset_path: str,
        dataset_subfolder: str = ''):
 
    # If the actor model is not specified, then exit
    if actor_model_paths == '':
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

    policy = TransformerPolicy(
        input_dim=obs_dim,
        output_dim=act_dim,
        num_envs=3,
        block_size=config["transformer"]["sequence_length"],
        n_embd=config["transformer"]["n_embd"],
        n_head=config["transformer"]["n_head"],
        n_layer=config["transformer"]["n_layer"],
        dropout=config["transformer"]["dropout"],
            device=device
    )
 
 
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

    count = 0
    for observation, position in collected_observations:
        for observation_step, position_step in zip(observation, position):
        # print(len(observation_sequence))
            if rd.random() > 0.4:
                con.in_grid_square(observation_step, position_step)
 
    path = os.path.join(dataset_path, dataset_subfolder)

    max_length = config["cav"]["dataset_max_length"]
 
    if os.path.exists(path) == False:
        os.makedirs(path)
    for key, val in con.datasets.items():
        filename = key
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_val:
                    filename = str(key) + '_' + str(sub_key)
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

    # Save the config as a file for reference
    save_config(dataset_path, config)
    
    # Using the raw dataset, build a random dataset
    build_random_dataset(dataset_path, dataset_subfolder)

    # Split the dataset into a training and test set
    split_dataset_into_train_test(dataset_path, dataset_subfolder, ratio = 0.8)

    # Build a dataset for grid observations
    grid_observation_dataset(dataset_path, dataset_subfolder, model_name=config["model_name"], map_name=config["env_name"]) # specifically for grid layout concept
    
    print("Finished building dataset!")
