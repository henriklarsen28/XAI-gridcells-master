import math
import os
import sys
from collections import deque
import copy
import pandas as pd
import random as rd

import torch

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent import DTQN_Agent

from agent.replay_memory import ReplayMemory
from env import SunburstMazeDiscrete
from utils.calculate_fov import calculate_fov_matrix_size
from utils.sequence_preprocessing import add_to_sequence, padding_sequence
from utils.state_preprocess import state_preprocess

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
    "max_steps_per_episode": 250,
    "random_start_position": True,
    "rewards": {
        "is_goal": 200 / 200,
        "hit_wall": -1 / 200,
        "has_not_moved": -1 / 200,
        "new_square": 0.4 / 200,
        "max_steps_reached": -0.5 / 200,
        "penalty_per_step": -0.1 / 200,
        "goal_in_sight": 0.5 / 200,
    },
    # TODO
    "observation_space": {
        "position": True,
        "orientation": True,
        "steps_to_goal": False,
        "last_known_steps": 0,
    },
    "render_fps": 5,
    "fov": math.pi / 1.5,
    "ray_length": 10,
    "number_of_rays": 100,
    "transformer": {
        "sequence_length": 45,
        "n_embd": 128,
        "n_head": 8,
        "n_layer": 3,
        "dropout": 0.3,
        "state_dim": num_states,
    },
}


# CAV for stuck in wall -> Sees into wall and tries to move forward
def save_to_csv(dataset: deque, file_name: str):
    dataset = [[state.tolist() for state in sequence] for sequence in dataset]
    # Convert from list of tensors to list of numpy arrays
    df = pd.DataFrame(dataset)
    
    df.to_csv(f"./dataset/{file_name}", index=False)

def positive_looking_at_wall(sequence: deque, legal_actions: list, action_sequence: deque):
    # Look at the last 2 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
    # The sequence should be added to the CAV positive dataset

    last_action = action_sequence[-1]

    #print("Last action: ", last_action)
    if len(legal_actions) == 2 and last_action == 0:
        # Save the observation sequence to the positive dataset
        #print("Positive stuck in wall")
        return sequence

    return None


def positive_rotating_stuck(
    sequence: deque, action_sequence: deque, position_sequence: deque
):

    # The position is the same over the last 12 states, and the agent is rotating in place
    # The sequence should be added to the CAV positive dataset

    position_sequence = list(position_sequence)
    last_12_positions = position_sequence[-12:]
    if len(set(last_12_positions)) == 1:
        # Check if the agent is rotating in place
        action_sequence = list(action_sequence)
        last_12_actions = set(action_sequence[-12:])
        if (1 in last_12_actions or 2 in last_12_actions) and 0 not in last_12_actions:
            #print("Positive rotating stuck")
            return sequence
    return None


def build_stuck_in_wall_dataset():
    positive_dataset = deque()

    negative_dataset = deque()

    # Load the dataset


def build_csv_dataset():
    # Load early agent data

    env_path = "../../env/map_v0/map_colored_wall_closed_doors.csv"
    model_load_path = "../../agent/model/transformers/model_woven-glade-815/sunburst_maze_map_v0_100.pth"

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

    # Load the model
    agent.model.load_state_dict(torch.load(model_load_path, map_location=device))
    agent.model.eval()

    # Containing a tuple of observation sequence, legal_actions, position sequence, action sequence
    collected_sequences = run_agent(env, agent)
    #print("Length of collected sequences: ", len(collected_sequences))
    positive_dataset_wall = deque()
    negative_dataset_wall = deque()

    positive_dataset_rotating = deque()
    negative_dataset_rotating = deque()
    

    for sequence in collected_sequences:
        observation_sequence, legal_actions, position_sequence, action_sequence = sequence
        #print(len(observation_sequence))
        
        # Check if the agent is stuck in a wall
        positive_wall = positive_looking_at_wall(observation_sequence, legal_actions, action_sequence)
        if positive_wall is not None:
            positive_dataset_wall.append(positive_wall)
        else:
            negative_dataset_wall.append(observation_sequence)

        positive_stuck = positive_rotating_stuck(observation_sequence, action_sequence, position_sequence)
        if positive_stuck is not None:
            positive_dataset_rotating.append(positive_stuck)
        else:
            negative_dataset_rotating.append(observation_sequence)

    # Shuffle the datasets
    rd.shuffle(negative_dataset_wall)
    rd.shuffle(negative_dataset_rotating)

    # Trim the negative datasets
    
    negative_dataset_wall = list(negative_dataset_wall)
    negative_dataset_rotating = list(negative_dataset_rotating)

    negative_dataset_wall = negative_dataset_wall[:len(positive_dataset_wall)]
    negative_dataset_rotating = negative_dataset_rotating[:len(positive_dataset_rotating)]

    # Save the datasets to csv files
    save_to_csv(positive_dataset_wall, "positive_wall.csv")
    save_to_csv(negative_dataset_wall, "negative_wall.csv")
    save_to_csv(positive_dataset_rotating, "positive_rotating.csv")
    save_to_csv(negative_dataset_rotating, "negative_rotating.csv")

    print("Wall: ", len(positive_dataset_wall))
    print("Rotating: ", len(positive_dataset_rotating))


def run_agent(env: SunburstMazeDiscrete, agent: DTQN_Agent):

    collected_sequences = deque()

    sequence_length = config["transformer"]["sequence_length"]
    max_episodes = 50
    observation_sequence = deque(maxlen=sequence_length)
    position_sequence = deque(maxlen=sequence_length)
    action_sequence = deque(maxlen=sequence_length)
    total_steps = 0
    # Testing loop over episodes
    for episode in range(0, max_episodes):
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
            #tensor_sequence = padding_sequence(tensor_sequence, sequence_length, device)
            #print(tensor_sequence.shape)
            # q_val_list = generate_q_values(env=self.env, model=self.agent.model)
            # self.env.q_values = q_val_list

            action = agent.select_action(tensor_sequence)
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
                    (copy.deepcopy(observation_sequence), copy.deepcopy(legal_actions), copy.deepcopy(position_sequence), copy.deepcopy(action_sequence))
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
    build_csv_dataset()


if __name__ == "__main__":
    main()