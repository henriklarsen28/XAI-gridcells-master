import math
import os
import sys
from collections import deque

import torch

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(project_root)

import torch

from agent.dtqn_agent import DTQN_Agent
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
print(num_states)

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


def negative_stuck_in_wall(sequence: deque, legal_actions: list):
    # Look at the last 2 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
    # The sequence should be added to the CAV negative dataset
    last_state = sequence[-1]
    second_last_state = sequence[-2]

    if last_state != second_last_state and 0 in legal_actions:
        return sequence
    return None


def positive_stuck_in_wall(sequence: deque, legal_actions: list):
    # Look at the last 2 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
    # The sequence should be added to the CAV positive dataset

    last_state = sequence[-1]
    second_last_state = sequence[-2]

    if last_state == second_last_state and 0 not in legal_actions:
        # Save the observation sequence to the positive dataset
        return sequence

    return None


def negative_rotating_stuck(
    sequence: deque, action_sequence: deque, position_sequence: deque
):

    pass


def positive_rotating_stuck(
    sequence: deque, action_sequence: deque, position_sequence: deque
):

    # The position is the same over the last 12 states, and the agent is rotating in place
    # The sequence should be added to the CAV positive dataset

    pass


def build_stuck_in_wall_dataset():
    positive_dataset = deque()

    negative_dataset = deque()

    # Load the dataset


def build_csv_dataset():
    # Load early agent data

    env_path = "../env/map_v0/map_colored_wall_closed_doors.csv"
    model_load_path = "../agent/model/transformers/model_woven-glade-815/sunburst_maze_map_v0_5500.pth"

    epsilon = 0.3

    env = SunburstMazeDiscrete(
        maze_file=env_path,
        render_mode="human",
        rewards=config["rewards"],
        observation_space=config["observation_space"],
        fov=config["fov"],
        ray_length=config["ray_length"],
        number_of_rays=config["number_of_rays"],
    )
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

    for sequence in collected_sequences:
        observation_sequence, legal_actions, position_sequence, action_sequence = sequence

        # Check if the agent is stuck in a wall
       



def run_agent(env: SunburstMazeDiscrete, agent: DTQN_Agent):

    collected_sequences = deque()

    sequence_length = config["transformer"]["sequence_length"]
    max_episodes = 100
    observation_sequence = deque(maxlen=sequence_length)
    position_sequence = deque(maxlen=sequence_length)
    action_sequence = deque(maxlen=sequence_length)
    total_steps = 0
    # Testing loop over episodes
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=42)
        done = False
        truncation = False
        steps_done = 0
        total_reward = 0

        while not done and not truncation:
            state = state_preprocess(state, device)

            observation_sequence = add_to_sequence(observation_sequence, state, device)
            position_sequence = add_to_sequence(position_sequence, env.position, device)
            tensor_sequence = torch.stack(list(observation_sequence))
            tensor_sequence = padding_sequence(tensor_sequence, sequence_length, device)
            print(tensor_sequence.shape)
            # q_val_list = generate_q_values(env=self.env, model=self.agent.model)
            # self.env.q_values = q_val_list

            action = agent.select_action(tensor_sequence)
            action_sequence.append(action)
            legal_actions = env.get_legal_actions()
            next_state, reward, done, truncation, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps_done += 1
            total_steps += 1
            # Make sure the sequence length is filled up
            if total_steps > sequence_length:
                collected_sequences.append(
                    (observation_sequence, legal_actions, position_sequence, action_sequence)
                )

        # Print log
        result = (
            f"Episode: {episode}, "
            f"Steps: {steps_done:}, "
            f"Reward: {total_reward:.2f}, "
        )
        print(result)

    return collected_sequences
