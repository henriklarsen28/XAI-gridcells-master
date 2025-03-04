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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent import DTQN_Agent
from agent.dqn.replay_memory import ReplayMemory
from env import SunburstMazeDiscrete
from utils.calculate_fov import calculate_fov_matrix_size
from utils.sequence_preprocessing import add_to_sequence, padding_sequence
from utils.state_preprocess import state_preprocess
from xai.cav.concept_definition import Concepts
from xai.cav.process_data import save_to_csv, shuffle_and_trim_datasets, split_dataset_into_train_test, save_config, find_model_files, build_random_dataset, grid_observation_dataset

device = torch.device("cpu")
fov_config = {
    "fov": math.pi / 1.5,
    "ray_length": 20,
    "number_of_rays": 100,
}

half_fov = fov_config["fov"] / 2
matrix_size = calculate_fov_matrix_size(fov_config["ray_length"], half_fov)
num_states = matrix_size[0] * matrix_size[1]

 # Parameters
config = {
    "model_path": "../../agent/dqn/models/",
    "model_name": "model_rose-pyramid-152",
    "model_episodes": [400, 2500, 5000],

    "env_name": "map_two_rooms_9_8",
    "env_path": "../../env/random_generated_maps/goal/medium/map_two_rooms_9_8.csv",

    "max_steps_per_episode": 250,
    "random_start_position": True,
    "random_goal_position": False,
    "rewards": {
        "is_goal": 200 / 200,
        "hit_wall": -0.5 / 200,
        "has_not_moved": -0.2 / 200,
        "new_square": 2 / 200,
        "max_steps_reached": -0.5 / 200,
        "penalty_per_step": -0.01 / 200,
        "goal_in_sight": 0 / 200,
        "number_of_squares_visible": 0 / 200,
        "is_false_goal": 0 / 200,
    },
    # TODO
    "observation_space": {
        "position": True,
        "orientation": True,
        "steps_to_goal": False,
        "last_known_steps": 0,
        "salt_and_pepper_noise": 0,
    },
    "save_interval": 100,
    "memory_capacity": 200_000,
    "render_fps": 100,
    "num_states": num_states,
    "clip_grad_normalization": 3,
    "fov": math.pi / 1.5,
    "ray_length": 20,
    "number_of_rays": 100,
    "grid_length": 4, # 4x4 grid

    "transformer": {
        "sequence_length": 15,
        "n_embd": 128,
        "n_head": 8,
        "n_layer": 3,
        "dropout": 0.3,
        "state_dim": num_states,
        "decouple_positional_embedding": False,
    },

    "cav": {
        "dataset_max_length" : 1500
    }
    
    }


def build_csv_dataset(model_paths: list, dataset_path: str, dataset_subfolder: str = '', max_length: int = 1500):
    # Load early agent data

    epsilon = 0.2

    env = SunburstMazeDiscrete(
        maze_file=config["env_path"],
        render_mode="human",
        random_start_position=config["random_start_position"],
        # random_goal_position=config["random_goal_position"],
        rewards=config["rewards"],
        observation_space=config["observation_space"],
        fov=config["fov"],
        ray_length=config["ray_length"],
        number_of_rays=config["number_of_rays"],
        grid_length=config["grid_length"]
    )

    env.metadata["render_fps"] = 100
    agent = DTQN_Agent(
        env=env,
        epsilon=epsilon,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        clip_grad_normalization=1.0,
        learning_rate=0.0001,
        discount=0.99,
        memory_capacity=10000,
        device=device,
        seed=42,
        transformer_param=config["transformer"],
    )

    con = Concepts(
        grid_pos_to_id=env.env_grid,
    )

    # clear the datasets in the concept class
    con.clear_datasets()
    # con.grid_pos_to_id = env.env_grid

    # Load the model
    model_load_path = model_paths[0]
    agent.model.load_state_dict(torch.load(model_load_path, map_location=device))
    agent.model.eval()
    print("Using model: ", model_load_path)
    # Containing a tuple of observation sequence, legal_actions, position sequence, action sequence
    collected_sequences = run_agent(env, agent, model_paths)
    # print("Length of collected sequences: ", len(collected_sequences))
    print("Collected_seq",len(collected_sequences))

    for sequence in collected_sequences:

        observation_sequence, legal_actions, position_sequence, action_sequence = (
            sequence
        )
        # print(len(observation_sequence))
        if rd.random() > 0.4:
            # con.looking_at_wall(observation_sequence, legal_actions, action_sequence)
            # con.rotating_stuck(observation_sequence, action_sequence, position_sequence)
            # con.goal_in_sight(observation_sequence)
            # con.inside_box(observation_sequence, position_sequence[-1])
            con.in_grid_square(observation_sequence, position_sequence[-1])

    path = f"./dataset/{config["model_name"]}/{config["env_name"]}/raw_data"

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


def run_agent(env: SunburstMazeDiscrete, agent: DTQN_Agent, models: list):

    collected_sequences = deque()

    sequence_length = config["transformer"]["sequence_length"]
    max_episodes = 20
    observation_sequence = deque(maxlen=sequence_length)
    position_sequence = deque(maxlen=sequence_length)
    action_sequence = deque(maxlen=sequence_length)
    total_steps = 0
    # Testing loop over episodes
    for episode in range(0, max_episodes):

        # Load new model when the episode is larger than 60
        if episode == 60:
            model_load_path = models[1]
            print("Using model: ", model_load_path)
            agent.model.load_state_dict(
                torch.load(model_load_path, map_location=device)
            )
            agent.model.eval()

        if episode == 90:
            model_load_path = models[2]
            print("Using model: ", model_load_path)
            agent.model.load_state_dict(
                torch.load(model_load_path, map_location=device)
            )
            agent.model.eval()

        state, _ = env.reset(seed=42)
        done = False
        truncation = False
        steps_done = 0
        total_reward = 0
        while not done and not truncation:
            state = state_preprocess(state, device)

            observation_sequence = add_to_sequence(observation_sequence, state, device)
            position_sequence.append(env.position)
            tensor_sequence = torch.stack(list(observation_sequence))

            action, _ = agent.select_action(tensor_sequence)
            action_sequence.append(action)
            legal_actions = env.legal_actions()
            next_state, reward, done, truncation, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps_done += 1
            total_steps += 1
            # Make sure the sequence length is filled up
            if len(observation_sequence) >= sequence_length:
                collected_sequences.append(
                    (
                        copy.deepcopy(observation_sequence),
                        copy.deepcopy(legal_actions),
                        copy.deepcopy(position_sequence),
                        copy.deepcopy(action_sequence),
                    )
                )

        # Print log
        result = (
            f"Episode: {episode}, "
            f"Steps: {steps_done:}, "
            f"Reward: {total_reward:.2f}, "
        )
        print(result)

    return copy.deepcopy(collected_sequences)

def main():
    model_path = os.path.join(config["model_path"], config["model_name"])
    model_files = find_model_files(model_path, config["model_episodes"])
    dataset_path = os.path.join('./dataset/', config["model_name"], config["env_name"])

    dataset_directory_train = os.path.join(dataset_path, 'train')
    dataset_directory_test = os.path.join(dataset_path, 'test')

    if not os.path.exists(dataset_directory_train):
        os.makedirs(dataset_directory_train, exist_ok=True)
    if not os.path.exists(dataset_directory_test):
        os.makedirs(dataset_directory_test, exist_ok=True)

    build_csv_dataset(model_files, dataset_path, 'raw_data', max_length=config["cav"]["dataset_max_length"])
    build_random_dataset(dataset_path, "raw_data")
    split_dataset_into_train_test(dataset_path, ratio = 0.8)
    grid_observation_dataset(dataset_path, 'raw_data', model_name=config["model_name"], map_name=config["env_name"]) # specifically for grid layout concept


if __name__ == "__main__":
    main()

