import ast
import copy
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
from xai.cav.concept_definition import Concepts as con

device = torch.device("cpu")
fov_config = {
    "fov": math.pi / 1.5,
    "ray_length": 8,
    "number_of_rays": 100,
}

half_fov = fov_config["fov"] / 2
matrix_size = calculate_fov_matrix_size(fov_config["ray_length"], half_fov)
num_states = matrix_size[0] * matrix_size[1]

# Parameters
config = {
    "max_steps_per_episode": 250,
    "random_start_position": True,
    "random_goal_position": True,
    "rewards": {
        "is_goal": 200 / 200,
        "hit_wall": -0.05 / 200,
        "has_not_moved": -0.2 / 200,
        "new_square": 0.01 / 200,
        "max_steps_reached": -0.5 / 200,
        "penalty_per_step": -0.01 / 200,
        "goal_in_sight": 0 / 200,
        "number_of_squares_visible": 0 / 200,
    },
    # TODO
    "observation_space": {
        "position": True,
        "orientation": True,
        "steps_to_goal": False,
        "last_known_steps": 0,
        "salt_and_pepper_noise": 0.2,
    },
    "save_interval": 100,
    "memory_capacity": 200_000,
    "render_fps": 100,
    "num_states": num_states,
    "clip_grad_normalization": 3,
    "fov": math.pi / 1.5,
    "ray_length": 8,
    "number_of_rays": 100,
    "transformer": {
        "sequence_length": 15,
        "n_embd": 128,
        "n_head": 8,
        "n_layer": 3,
        "dropout": 0.3,
        "state_dim": num_states,
        "decouple_positional_embedding": False,
    },
    "grid_length": 4, # 4x4 grid
}

# CAV for stuck in wall -> Sees into wall and tries to move forward
def save_to_csv(dataset: deque, file_name: str):
    dataset = [[state.tolist() for state in sequence] for sequence in dataset]
    # Convert from list of tensors to list of numpy arrays
    df = pd.DataFrame(dataset)

    df.to_csv(f"./dataset/{file_name}", index=False)


def build_csv_dataset():
    # Load early agent data
    env_path = "../../env/map_v0/map_open_doors_90_degrees.csv"
    model_load_path = "../../agent/model/transformers/model_vivid-firebrand-872/sunburst_maze_map_v0_400.pth"

    epsilon = 0.2

    env = SunburstMazeDiscrete(
        maze_file=env_path,
        render_mode="human",
        random_start_position=config["random_start_position"],
        random_goal_position=config["random_goal_position"],
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

    # create a positive and negative dataset for each concept as type deque for the concepts

    # Load the model
    agent.model.load_state_dict(torch.load(model_load_path, map_location=device))
    agent.model.eval()

    # Containing a tuple of observation sequence, legal_actions, position sequence, action sequence
    collected_sequences = run_agent(env, agent)
    # print("Length of collected sequences: ", len(collected_sequences))
    
    # clear concept datasets

    grid_pos_to_id = {(0,1):0}

    grid_observations = {grid_id: deque() for grid_id in set(grid_pos_to_id.values())}

    for sequence in collected_sequences:

        observation_sequence, legal_actions, position_sequence, action_sequence = (
            sequence
        )
        # print(len(observation_sequence))
        if rd.random() > 0.4:
            # run each function in the concepts list
            con.positive_looking_at_wall(observation_sequence, legal_actions, action_sequence)
            con.positive_rotating_stuck(observation_sequence, action_sequence, position_sequence)
            con.positive_goal_in_sight(observation_sequence)
            con.positive_inside_box(observation_sequence, position_sequence[-1])
            con.in_grid_square(observation_sequence, grid_pos_to_id, position_sequence[-1])

    '''    
    # Shuffle the datasets
    rd.shuffle(negative_dataset_wall)
    rd.shuffle(negative_dataset_rotating)
    rd.shuffle(negative_dataset_goal)
    rd.shuffle(positive_dataset_inside_box)

    # Trim the negative datasets

    negative_dataset_wall = list(negative_dataset_wall)
    negative_dataset_rotating = list(negative_dataset_rotating)
    negative_dataset_goal = list(negative_dataset_goal)
    negative_dataset_inside_box = list(negative_dataset_inside_box)

    dataset_length = max(min(len(positive_dataset_goal), 1500), 1500)

    negative_dataset_wall = negative_dataset_wall[:dataset_length]
    negative_dataset_rotating = negative_dataset_rotating[:dataset_length]
    negative_dataset_goal = negative_dataset_goal[:dataset_length]
    negative_dataset_inside_box = negative_dataset_inside_box[:dataset_length]

    # Save the datasets to csv files
    save_to_csv(positive_dataset_wall, "positive_wall.csv")
    save_to_csv(negative_dataset_wall, "negative_wall.csv")

    save_to_csv(positive_dataset_rotating, "positive_rotating.csv")
    save_to_csv(negative_dataset_rotating, "negative_rotating.csv")

    save_to_csv(positive_dataset_goal, "positive_goal.csv")
    save_to_csv(negative_dataset_goal, "negative_goal.csv")

    save_to_csv(positive_dataset_inside_box, "positive_inside_box.csv")
    save_to_csv(negative_dataset_inside_box, "negative_inside_box.csv")

    print("Wall: ", len(positive_dataset_wall))
    print("Rotating: ", len(positive_dataset_rotating))
    print("Goal: ", len(positive_dataset_goal))
    print("Inside box: ", len(positive_dataset_inside_box))
    '''

def shuffle_and_trim_datasets():
    # load datasets from Concept class
    for dataset in con.datasets:
        # shuffle the dataset
        rd.shuffle(dataset)
        # trim the dataset
        dataset = dataset[:1500]
    


def split_dataset_into_train_test(
    dataset: deque, ratio: float = 0.8, concept: str = ""
):
    # Split the dataset into a training and test set
    train_size = int(len(dataset) * ratio)
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    train = [
        [torch.tensor(ast.literal_eval(state)) for state in states]
        for _, states in train_dataset.iterrows()
    ]
    test = [
        [torch.tensor(ast.literal_eval(state)) for state in states]
        for _, states in test_dataset.iterrows()
    ]

    save_to_csv(train, f"{concept}_train.csv")
    save_to_csv(test, f"{concept}_test.csv")

    return train_dataset, test_dataset


def run_agent(env: SunburstMazeDiscrete, agent: DTQN_Agent):

    collected_sequences = deque()

    sequence_length = config["transformer"]["sequence_length"]
    max_episodes = 130
    observation_sequence = deque(maxlen=sequence_length)
    position_sequence = deque(maxlen=sequence_length)
    action_sequence = deque(maxlen=sequence_length)
    total_steps = 0
    # Testing loop over episodes
    for episode in range(0, max_episodes):

        # Load new model when the episode is larger than 60
        if episode == 60:
            model_load_path = "../../agent/model/transformers/model_vivid-firebrand-872/sunburst_maze_map_v0_2500.pth"
            agent.model.load_state_dict(
                torch.load(model_load_path, map_location=device)
            )
            agent.model.eval()

        if episode == 90:
            model_load_path = "../../agent/model/transformers/model_vivid-firebrand-872/sunburst_maze_map_v0_5200.pth"
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
            # tensor_sequence = padding_sequence(tensor_sequence, sequence_length, device)
            # print(tensor_sequence.shape)
            # q_val_list = generate_q_values(env=self.env, model=self.agent.model)
            # self.env.q_values = q_val_list

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
    # build_csv_dataset()

    # Load the datasets
    positive_wall = pd.read_csv("./dataset/positive_wall.csv")
    negative_wall = pd.read_csv("./dataset/negative_wall.csv")

    positive_rotating = pd.read_csv("./dataset/positive_rotating.csv")
    negative_rotating = pd.read_csv("./dataset/negative_rotating.csv")

    positive_goal = pd.read_csv("./dataset/positive_goal.csv")
    negative_goal = pd.read_csv("./dataset/negative_goal.csv")

    random = pd.concat([negative_goal, positive_goal, negative_rotating, positive_rotating, negative_wall, positive_wall])
    random = random.sample(frac=1, random_state=42).reset_index(drop=True)

    random = random.sample(1500, random_state=42)
    # Split into positive and negative
    half = int(len(random) / 2)
    positive_random = random.iloc[:half]
    negative_random = random.iloc[half:]

    split_dataset_into_train_test(positive_random, ratio=0.8, concept="random_positive")
    split_dataset_into_train_test(negative_random, ratio=0.8, concept="random_negative")

    # Split the datasets into training and test sets

    # split_dataset_into_train_test(positive_wall, ratio=0.8, concept="wall_positive")
    # split_dataset_into_train_test(negative_wall, ratio=0.8, concept="wall_negative")

    # split_dataset_into_train_test(positive_rotating, ratio=0.8, concept="rotating_positive")
    # split_dataset_into_train_test(negative_rotating, ratio=0.8, concept="rotating_negative")

    # split_dataset_into_train_test(positive_goal, ratio=0.8, concept="goal_positive")
    # split_dataset_into_train_test(negative_goal, ratio=0.8, concept="goal_negative")


if __name__ == "__main__":
    main()
