import math
import os
import sys
from collections import deque

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(project_root)

import torch

from agent.dtqn_agent import DTQN_Agent
from env import SunburstMazeDiscrete
from utils.calculate_fov import calculate_fov_matrix_size
from utils.state_preprocess import state_preprocess
from utils.sequence_preprocessing import padding_sequence, add_to_sequence

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


def negative_stuck_in_wall(sequence: deque):
    # Look at the last 3 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
    # The sequence should be added to the CAV positive dataset

    pass


def positive_stuck_in_wall(sequence: deque):
    # Look at the last 3 states, if the agents last states are the same and the agent is not allowed to move forward the agent is stuck in a wall
    # The sequence should be added to the CAV positive dataset

    pass


def negative_rotating_stuck():
    pass


def positive_rotating_stuck():
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


def run_agent(env: SunburstMazeDiscrete, agent: DTQN_Agent):
    sequence_length = config["transformer"]["sequence_length"]
    max_episodes = 100
    sequence = deque(maxlen=sequence_length)

    # Testing loop over episodes
    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=42)
        done = False
        truncation = False
        steps_done = 0
        total_reward = 0

        while not done and not truncation:
            state = state_preprocess(state, device) # TODO: Import these, move them to utils

            sequence = add_to_sequence(sequence, state, device)
            tensor_sequence = torch.stack(list(sequence))
            tensor_sequence = padding_sequence(
                tensor_sequence, sequence_length, device
            )
            print(tensor_sequence.shape)
            # q_val_list = generate_q_values(env=self.env, model=self.agent.model)
            # self.env.q_values = q_val_list

            action = agent.select_action(tensor_sequence)
            next_state, reward, done, truncation, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps_done += 1

        # Print log
        result = (
            f"Episode: {episode}, "
            f"Steps: {steps_done:}, "
            f"Reward: {total_reward:.2f}, "
        )
        print(result)
