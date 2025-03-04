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
 
from agent.ppo.gated_transformer_decoder_policy import TransformerPolicy
from agent.ppo.ppo import PPO_agent
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
 
device = torch.device("cpu")
 
fov_config = {
            "fov": math.pi / 1.5,
            "ray_length": 15,
            "number_of_rays": 40,
            }
 
 
config = {
 
        # MODEL PATHS
        "model_path": "../../../agent/ppo/models/transformers/expert-durian-1146/actor",
        "model_name": "expert-durian-1146", # TODO: change to the correct model name
        "model_episodes": [100, 150, 200], # TODO: change to the correct model episodes
 
        # PPO
        "policy_load_path": "../../../agent/ppo/models/transformers/expert-durian-1146/actor",
        "critic_load_path": None,
        
        # ENVIRONMENT
        "env_name": "map_two_rooms_18_19", #TODO: change to the correct env name
        "env_path": "../../../env/random_generated_maps/goal/large/map_two_rooms_18_19.csv", # TODO: Change to the correct path for what the model was trained on
        # "env_path": "../../../env/map_v0/map_open_doors_horizontal.csv",
        "grid_length": 4,
 
 
        "cav": {
            "dataset_max_length" : 1500
        },
        
        # RENDERING
        "train_mode": False,
        "map_path_train": None,
        "render": True,
        "render_mode": "human",
        
        # HYPERPARAMETERS
        "loss_function": "mse",
        "learning_rate": 3e-4,
        "batch_size": 3000,
        "mini_batch_size": 64,
        "n_mini_batches": 5,
        "optimizer": "adam",
        "PPO": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_updates_per_iteration": 10,  # hard update of the target model
            "clip": 0.2,
            "clip_grad_normalization": 0.5,
            "policy_kl_range": 0.0008,
            "policy_params": 5,
            "normalize_advantage": True,
        },
 
        "map_path": None,
        "max_steps_per_episode": 250,
        "random_start_position": True,
        "random_goal_position": False,
        "rewards": {
            "is_goal": 5,
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
            "sequence_length": 30,
            "n_embd": 128,
            "n_head": 6,
            "n_layer": 2,
            "dropout": 0.2,
            "decouple_positional_embedding": False,
        },
        "entropy": {"coefficient": 0.015, "min": 0.0001, "step": 1_000},
    }
 
print("config", config["PPO"])
 
env = SunburstMazeContinuous(
    maze_file=config['env_path'],
    max_steps_per_episode=config["max_steps_per_episode"],
    render_mode=config["render_mode"],
    random_start_position=config["random_start_position"],
    random_goal_position=config["random_goal_position"],
    rewards=config["rewards"],
    fov=fov_config["fov"],
    ray_length=fov_config["ray_length"],
    number_of_rays=fov_config["number_of_rays"],
    grid_length=config["grid_length"]
)
 
agent = PPO_agent(
    env=env,
    device=device,
    config=config
)
 
def build_csv_dataset(actor_model_paths: list, dataset_path: str, dataset_subfolder = '', max_length:int = 1500):
    # Load early agent data
    actor_model = actor_model_paths[0] # TODO: set up actor model paths
 
 
    # If the actor model is not specified, then exit
    if actor_model_paths == '':
        print(f"Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)
 
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
    collected_observations = eval_policy(policy=policy, actor_model_paths=actor_model_paths, env=env, sequence_length=config["transformer"]["sequence_length"], device=device, render=True, max_steps=config["max_steps_per_episode"])
    print("Collected observations", len(collected_observations), collected_observations[0])
 
    #TODO: Update model
 
    count = 0
    for observation, position in collected_observations:
        for observation_step, position_step in zip(observation, position):
        # print(len(observation_sequence))
            if rd.random() > 0.4:
                con.in_grid_square(observation_step, position_step)
 
    path = os.path.join(dataset_path, dataset_subfolder)
 
    if os.path.exists(path) == False:
        os.makedirs(path)
    for key, val in con.datasets.items():
        filename = key
        if isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if sub_val:
                    filename = str(key) + '_' + str(sub_key)
                    data_preprocessed = shuffle_and_trim_datasets(sub_val, max_length)
                    save_to_csv(data_preprocessed, filename, path)
        else:
            if val:
                data_preprocessed = shuffle_and_trim_datasets(val, max_length)
                save_to_csv(data_preprocessed, filename, path)
 
    save_config(dataset_path, config)
 
 
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
 
    build_csv_dataset(model_files, dataset_path, 'raw_data', max_length=config["cav"]["dataset_max_length"])
    #build_random_dataset(dataset_path, "raw_data")
    #split_dataset_into_train_test(dataset_path, ratio = 0.8)
    #grid_observation_dataset(dataset_path, 'raw_data', model_name=config["model_name"], map_name=config["env_name"]) # specifically for grid layout concept
 
 
if __name__ == "__main__":
    main()
 