import os
import sys
from collections import deque

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(project_root)


import math

import gymnasium as gym
import keras as keras
import numpy as np
import pygame
import torch
import wandb
from dqn_agent import DQN_Agent
from explain_network import generate_q_values
from scipy.special import softmax

from env import SunburstMazeDiscrete
from utils.calculate_fov import calculate_fov_matrix_size
from utils.state_preprocess import state_preprocess

wandb.login()

# Define the CSV file path relative to the project root
map_path_train = os.path.join(project_root, "env/map_v0/map_closed_doors.csv")
map_path_test = os.path.join(project_root, "env/map_v0/map_closed_doors.csv")


device = torch.device("cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Was faster with cpu??? Loading between cpu and mps is slow maybe


# Seed everything for reproducible results
seed = 2024
np.random.seed(seed)
np.random.default_rng(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)

# For cuda seed
"""if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False"""


class Model_TrainTest:
    def __init__(self, config):

        # Define RL parameters
        self.train_mode = config["train_mode"]
        self.RL_load_path = config["RL_load_path"]
        self.save_path = config["save_path"]
        self.save_interval = config["save_interval"]

        self.clip_grad_normalization = config["clip_grad_normalization"]
        self.learning_rate = config["learning_rate"]
        self.discount_factor = config["discount_factor"]
        self.batch_size = config["batch_size"]
        self.update_frequency = config["target_model_update"]
        self.max_episodes = config["total_episodes"]
        self.max_steps = config["max_steps_per_episode"]
        self.render = config["render"]

        self.epsilon = config["epsilon"]
        self.epsilon_min = config["epsilon_min"]
        self.epsilon_decay = config["epsilon_decay"]

        self.memory_capacity = config["memory_capacity"]

        self.num_states = config["num_states"]
        self.render_fps = config["render_fps"]

        self.rewards = config["rewards"]
        self.random_start_position = config["random_start_position"]
        self.observation_space = config["observation_space"]

        self.fov = config["fov"]
        self.ray_length = config["ray_length"]
        self.number_of_rays = config["number_of_rays"]
        
        map_path = map_path_train
        if not self.train_mode:
             map_path = map_path_test


        # Define Env
        self.env = SunburstMazeDiscrete(
            maze_file=map_path,
            render_mode=render_mode,
            max_steps_per_episode=self.max_steps,
            random_start_position=self.random_start_position,
            rewards=self.rewards,
            observation_space=self.observation_space,
            fov=self.fov,
            ray_length=self.ray_length,
            number_of_rays=self.number_of_rays
        )
        self.env.metadata["render_fps"] = (
            self.render_fps
        )  # For max frame rate make it 0

        # Define the agent class
        self.agent = DQN_Agent(
            env=self.env,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            clip_grad_normalization=self.clip_grad_normalization,
            learning_rate=self.learning_rate,
            discount=self.discount_factor,
            memory_capacity=self.memory_capacity,
            device=device,
            seed=seed,
        )

    

    def train(self):
        """
        Reinforcement learning training loop.
        """

        total_steps = 0
        self.reward_history = []
        frames = []
        wandb.init(project="sunburst-maze", config=self)

        # Create the nessessary directories
        if not os.path.exists("./gifs"):
                    os.makedirs("./gifs")
        
        if not os.path.exists("./model"):
                    os.makedirs("./model")

        # Training loop over episodes
        for episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset()
            state = state_preprocess(state, device)
            done = False
            truncation = False
            steps_done = 0
            total_reward = 0
            stuck_behavior = 0


            while not done and not truncation:
                action, _= self.agent.select_action(state)
                
                next_state, reward, done, truncation, _ = self.env.step(action)
                if render_mode == "rgb_array":
                    if episode % 100 == 0:
                        frame = self.env._render_frame()
                        if type(frame) == np.ndarray:
                            frames.append(frame.astype(np.uint8))
                if render_mode == "human":
                    self.env.render()

                next_state = state_preprocess(next_state, device)
                self.agent.replay_memory.store(state, action, next_state, reward, done)

                if (
                    len(self.agent.replay_memory) > self.batch_size
                    and sum(self.reward_history) > 0
                ):  # Start learning after some episodes and the agent has achieved some reward
                    self.agent.learn(self.batch_size, (done or truncation))

                    # Update target-network weights
                    if total_steps % self.update_frequency == 0:
                        self.agent.hard_update()

                state = next_state
                total_reward += reward
                steps_done += 1

                if self.env.has_not_moved(self.env.position):
                    # record stuck behavior
                    stuck_behavior += 1

            # Appends for tracking history
            self.reward_history.append(total_reward)  # episode reward
            total_steps += steps_done

            # Decay epsilon at the end of each episode
            self.agent.update_epsilon()

            # Create gif
            gif = None
            if frames:
                if os.path.exists("./gifs") is False:
                    os.makedirs("./gifs")

                gif = self.env.create_gif(
                    gif_path=f"./gifs/{episode}.gif", frames=frames
                )
                frames.clear()


            # -- based on interval
            if episode % self.save_interval == 0:
                self.agent.save(self.save_path + "_" + f"{episode}" + ".pth")

                print("\n~~~~~~Interval Save: Model saved.\n")

            wandb.log(
                {
                    "Episode": episode,
                    "Reward per episode": total_reward,
                    "Epsilon": self.agent.epsilon,
                    "Steps done": steps_done,
                    "Gif:": (wandb.Video(gif, fps=4, format="gif") if gif else None),
                    "Stuck behavior": stuck_behavior,
                }
            )

    def test(self, max_episodes):
        """
        Reinforcement learning policy evaluation.
        """

        # Load the weights of the test_network
        self.agent.model.load_state_dict(torch.load(self.RL_load_path))
        self.agent.model.eval()

        q_val_list = generate_q_values(env=self.env, model=self.agent.model)
        self.env.q_values = q_val_list

        stuck_behavior = dict()
        
        # Testing loop over episodes
        for episode in range(1, max_episodes + 1):
            state, _ = self.env.reset(seed=seed)
            done = False
            truncation = False
            steps_done = 0
            total_reward = 0

            while not done and not truncation:
                state = state_preprocess(state, device)
                action, q_variance = self.agent.select_action(state)
                self.env.q_variance = q_variance
                next_state, reward, done, truncation, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                steps_done += 1
                if self.env.has_not_moved(self.env.position):
                    # record stuck behavior
                    stuck_behavior[episode] = self.env.position
                    print("Agent has not moved for 10 steps. Breaking the episode.")
                    break  # Skip the episode if the agent is stuck in a loop

            # Print log
            result = (
                f"Episode: {episode}, "
                f"Steps: {steps_done:}, "
                f"Reward: {total_reward:.2f}, "
            )
            # print(result)

            print("Stuck behavior: ", stuck_behavior)

        pygame.quit()  # close the rendering window


def get_num_states(map_path):

    num_rows = 0
    num_cols = 0
    with open(map_path, "r") as f:
        for line_num, line in enumerate(f):
            num_rows += 1
            num_cols = len(line.strip().split(","))
    num_states = num_rows * num_cols
    return num_states


if __name__ == "__main__":
    # Parameters:

    train_mode = False

    render = True
    render_mode = "human"

    if train_mode:
        render_mode = "rgb_array" if render else None

    map_version = map_path_train.split("/")[-2]

    # Read the map file to find the number of states
    # num_states = get_num_states(map_path_train)

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
        "train_mode": train_mode,
        "render": render,
        "render_mode": render_mode,
        "RL_load_path": f"./model/feed_forward/sunburst_maze_{map_version}_8900.pth",
        "save_path": f"./model/sunburst_maze_{map_version}",
        "loss_function": "mse",
        "learning_rate": 0.0005,
        "batch_size": 100,
        "optimizer": "adam",
        "total_episodes": 100_000,
        "epsilon": 1 if train_mode else -1,
        "epsilon_decay": 0.998,
        "epsilon_min": 0.1,
        "discount_factor": 0.90,
        "alpha": 0.1,
        "map_path": map_path_train,
        "target_model_update": 10,  # hard update of the target model
        "max_steps_per_episode": 250,
        "random_start_position": True,
        "rewards": {
            "is_goal": 1,
            "hit_wall": -0.001,
            "has_not_moved": -0.001,
            "new_square": 0.001,
            "max_steps_reached": -0.001,
            "penalty_per_step": -0.0001,
        },
        # TODO
        "observation_space": {
            "position": True,
            "orientation": True,
            "steps_to_goal": False,
            "last_known_steps": 0,
        },
        "save_interval": 100,
        "memory_capacity": 50_000,
        "render_fps": 5,
        "num_states": num_states,
        "clip_grad_normalization": 3,
        "fov": math.pi / 1.5,
        "ray_length": 20,
        "number_of_rays": 100,
    }

    # Run
    DRL = Model_TrainTest(config)
    # Train
    if train_mode:
        DRL.train()
    else:
        # Test
        DRL.test(max_episodes=config["total_episodes"])
