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

from agent import DQN_Agent
from env import SunburstMazeContinuous
from utils.calculate_fov import calculate_fov_matrix_size
from utils.sequence_preprocessing import add_to_sequence, padding_sequence
from utils.state_preprocess import state_preprocess
from agent.ppo_test_code import FeedForwardNN
from agent.ppo_test_code.network_policy import FeedForwardNNPolicy
from xai.cav.cav_ff.eval_policy_ff import eval_policy

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


# CAV for stuck in wall -> Sees into wall and tries to move forward
def save_to_csv(dataset: deque, file_name: str):
    dataset = [state.tolist() for state in dataset]
    # Convert from list of tensors to list of numpy arrays
    df = pd.DataFrame(dataset)
    
    df.to_csv(f"./dataset/{file_name}", index=False)


'''def build_next_to_wall_dataset():
    # Load early agent data
    env_path = "../../../env/map_v0/map_open_doors_horizontal.csv"
    model_load_path = "../../../agent/model/feed_forward/serene-voice-977/sunburst_maze_map_v0_100.pth"

    epsilon = 0.2

    env = SunburstMazeDiscrete(
        maze_file=env_path,
        render_mode="human",
        random_start_position=config["random_start_position"],
        rewards=config["rewards"],
        observation_space=config["observation_space"],
        fov=config["fov"],
        ray_length=config["ray_length"],
        number_of_rays=config["number_of_rays"],
    )

    env.metadata["render_fps"] = 100
    agent = DQN_Agent(
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
    )

    # Load the model
    agent.model.load_state_dict(torch.load(model_load_path, map_location=device))
    agent.model.eval()

    # Containing a tuple of observation sequence, legal_actions, position sequence, action sequence
    collected_observations = run_agent(env, agent, model_paths)
    #print("Length of collected sequences: ", len(collected_sequences))
    positive_dataset_next_wall = deque()
    negative_dataset_next_wall = deque()


    

    for observation in collected_observations:
        obs, legal_actions, position, action = observation
        #print(len(observation_sequence))
        if rd.random() > 0.4:
            # Check if the agent is stuck in a wall
            positive_next_wall = positive_next_to_wall(obs, position)
            if positive_next_wall:
                positive_dataset_next_wall.append(obs)
            else:
                negative_dataset_next_wall.append(obs)

    # Shuffle the datasets
    rd.shuffle(negative_dataset_next_wall)

    # Trim the negative datasets
    negative_dataset_wall = list(negative_dataset_next_wall)


    negative_dataset_wall = negative_dataset_wall[:len(positive_dataset_next_wall)]

    # Save the datasets to csv files
    save_to_csv(positive_dataset_next_wall, "positive_next_wall.csv")
    save_to_csv(negative_dataset_wall, "negative_next_wall.csv")


def build_goal_dataset():
    # Load early agent data
    env_path = "../../../env/map_v0/map_open_doors_horizontal.csv"
    model_load_path = "../../../agent/model/feed_forward/serene-voice-977/sunburst_maze_map_v0_100.pth"

    epsilon = 0.2

    env = SunburstMazeDiscrete(
        maze_file=env_path,
        render_mode="human",
        random_start_position=config["random_start_position"],
        rewards=config["rewards"],
        observation_space=config["observation_space"],
        fov=config["fov"],
        ray_length=config["ray_length"],
        number_of_rays=config["number_of_rays"],
    )

    env.metadata["render_fps"] = 100
    agent = DQN_Agent(
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
    )

    # Load the model
    agent.model.load_state_dict(torch.load(model_load_path, map_location=device))
    agent.model.eval()

    # Containing a tuple of observation sequence, legal_actions, position sequence, action sequence
    collected_observations = run_agent(env, agent, model_paths)
    #print("Length of collected sequences: ", len(collected_sequences))
    positive_dataset_goal = deque()
    negative_dataset_goal = deque()
    

    for observation in collected_observations:
        obs, legal_actions, position, action = observation
        #print(len(observation_sequence))
        if rd.random() > 0.4:
            # Check if the agent is stuck in a wall
            positive_goal = positive_goal_in_sight(obs)
            if positive_goal:
                positive_dataset_goal.append(obs)
            else:
                negative_dataset_goal.append(obs)

    # Shuffle the datasets
    rd.shuffle(negative_dataset_goal)

    # Trim the negative datasets
    negative_dataset_goal = list(negative_dataset_goal)

    dataset_length = max(min(len(positive_dataset_goal), 1500),1500)
    

    negative_dataset_goal = negative_dataset_goal[:dataset_length]

    # Save the datasets to csv files
    save_to_csv(positive_dataset_goal, "positive_goal.csv")
    save_to_csv(negative_dataset_goal, "negative_goal.csv")'''

def shuffle_and_trim_datasets(dataset: deque):
    max_length = config["cav"]["dataset_max_length"]
    # shuffle the dataset
    data = list(dataset)
    rd.shuffle(data)
    # trim the dataset
    if len(data) >= max_length:
        data = data[:max_length]
    return data


def split_dataset_into_train_test(
    dataset_path: str, dataset_subfolder = '', ratio: float = 0.8
): 
    # check if the folder 'train' and 'test' exists in the dataset path, if not create them
    train_dir = os.path.join(dataset_path, "train" if not dataset_subfolder == '' else dataset_subfolder)
    test_dir = os.path.join(dataset_path, "test" if not dataset_subfolder == '' else dataset_subfolder)

    raw_data_dir = os.path.join(dataset_path, dataset_subfolder)
    
    print("Splitting dataset into training and test set")
    # walk through the dataset path directory
    for file in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file)
        # check if the file is a csv file
        if not file.endswith(".csv"):
            continue
        dataset = pd.read_csv(file_path)
        # Split the dataset into a training and test set
        train_size = int(len(dataset) * ratio)
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]

        print('train dataset', train_dataset)

        train = [
            [torch.tensor(ast.literal_eval(state)) for state in states]
            for _, states in train_dataset.iterrows()
        ]
        test = [
            [torch.tensor(ast.literal_eval(state)) for state in states]
            for _, states in test_dataset.iterrows()
        ]

        filename = os.path.splitext(file)[0]
        save_to_csv(train, f"{filename}_train", train_dir)
        save_to_csv(test, f"{filename}_test", test_dir)

def save_config(dataset_path: str):
    # path = f"./dataset/{config["model_name"]}/{config["env_name"]}"
    if os.path.exists(dataset_path) == False:
        os.makedirs(dataset_path)
    with open(os.path.join(dataset_path, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

def find_model_files(base_path: str, ep_ids: list):

    # find the model file that ends with a specific number followed by '.pth'
    try:
        files = os.listdir(base_path)
    except FileNotFoundError:
        print(f"Directory {base_path} does not exist.")
        return None

    ep_ids = sorted(ep_ids, reverse=False)

    model_files = []

    for num in ep_ids:
        for file in files:
            if file.endswith(f"_{num}.pth"):
                path = os.path.join(base_path, file)
                model_files.append(path)
    return model_files

def build_random_dataset(dataset_path: str, dataset_subfolder = ''):

    file_path = os.path.join(dataset_path, dataset_subfolder)
    # load and concatenate all CSV files
    print("Building random dataset")
    files = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.csv')]
    dataset = pd.concat([pd.read_csv(f) for f in files])
    
    # shuffle and sample the dataset
    random_sample = dataset.sample(n=1500, frac=None, random_state=42).reset_index(drop=True)
    
    # Split into positive and negative
    half = len(random_sample) // 2  # Use integer division directly
    random_positive = random_sample.iloc[:half]
    random_negative = random_sample.iloc[half:]

    random_positive.to_csv(f"{os.path.join(dataset_path, 'random_positive.csv')}", index=False)
    random_negative.to_csv(f"{os.path.join(dataset_path, 'random_negative.csv')}", index=False)

def get_positive_negative_data(concept: str, datapath: str):
    negative_files = []
    positive_file = None

    print('Datapath:', datapath)
    for file in os.listdir(datapath):
        file_path = os.path.join(datapath, file)
        if file.startswith(concept):
            positive_file = file_path
            print('Positive file:', positive_file)
        else:
            negative_files.append(file_path)

    if positive_file is None:
        raise FileNotFoundError("Positive file not found")
    
    positive_df = pd.read_csv(positive_file)
    
    # Determine sample size: at least 1500 lines or the length of the positive file content, whichever is greater
    sample_size = max(1500, len(positive_df))

    # Aggregate negative file content and then sample
    neg_dfs = []
    for neg_file in negative_files:
        neg_df = pd.read_csv(neg_file)
        neg_dfs.append(neg_df)

    negative_df = pd.concat(neg_dfs)
    negative_df = negative_df.sample(sample_size)
    
    return positive_df, negative_df


def grid_observation_dataset(dataset_path, dataset_subfolder, model_name: str, map_name: str):
    for i in range(15):
        concept = "grid_observations_" + str(i)
        negative_file_test = os.path.join(dataset_path, 'test', f"{concept}_negative_test.csv")
        negative_file_train = os.path.join(dataset_path, 'train',f"{concept}_negative_train.csv")

        if not os.path.exists(negative_file_test):
            positive_file_test, negative_file_test = get_positive_negative_data(concept, os.path.join(dataset_path, dataset_subfolder))
            positive_file_test.to_csv(os.path.join(dataset_path, 'test', f"{concept}_positive_test.csv"), index=False)
            negative_file_test.to_csv(os.path.join(dataset_path, 'test', f"{concept}_negative_test.csv"), index=False)
        
        if not os.path.exists(negative_file_train):
            positive_file_train, negative_file_train = get_positive_negative_data(concept, datapath = f"dataset/{model_name}/{map_name}/raw_data")
            positive_file_train.to_csv(os.path.join(dataset_path, 'train', f"{concept}_positive_train.csv"), index=False)
            negative_file_train.to_csv(os.path.join(dataset_path, 'train', f"{concept}_negative_train.csv"), index=False)

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
    obs_dim = env.observation_space.n
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
    #build_random_dataset(dataset_path, "raw_data")
    #split_dataset_into_train_test(dataset_path, ratio = 0.8)
    #grid_observation_dataset(dataset_path, 'raw_data', model_name=config["model_name"], map_name=config["env_name"]) # specifically for grid layout concept


    #build_next_to_wall_dataset()
    #build_goal_dataset()


if __name__ == "__main__":
    main()
