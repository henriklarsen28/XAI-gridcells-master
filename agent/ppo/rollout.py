import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

import copy
import multiprocessing as mp
from collections import deque

import numpy as np
import torch

from env import SunburstMazeContinuous
from utils import add_to_sequence, create_gif, padding_sequence


def run_episode(
    env: SunburstMazeContinuous,
    iteration_counter: int,
    sequence_length: int = 45,
    device=torch.device("cpu"),
    frame=False,
    policy_network=None,
    queue: mp.Queue = None,
    job_id: int = 0,
):

    assert env is not None, "Environment is not defined"


    #state, _ = new_env.reset()
    done = False

    max_steps = env.max_steps_per_episode

    # Initialize the lists to store the observations, actions, log_probs, rewards, and frames
    state_sequence = deque(maxlen=sequence_length)

    observations = []
    actions = []
    log_probs = []
    rewards = []
    frames = []
    episode_rewards = []
    timesteps = 0

    for ep_timestep in range(max_steps):
        timesteps += 1
        state_sequence = add_to_sequence(state_sequence, state, device)
        tensor_sequence = torch.stack(list(state_sequence))
        tensor_sequence = padding_sequence(tensor_sequence, sequence_length, device)
        action, log_prob = get_action(tensor_sequence, policy_network)
        action = action[0]
        state, reward, terminated, turnicated, _ = env.step(action)

        if (
            env.render_mode == "rgb_array"
            and iteration_counter % 30 == 0
            and len(rewards) == 0
        ):  # Create gif on the first episode in the rollout
            frame = env.render()
            if type(frame) == np.ndarray:
                frames.append(frame)

        if env.render_mode == "human":
            env.render()

        observations.append(tensor_sequence)
        actions.append(action)
        log_probs.append(log_prob)
        episode_rewards.append(reward)

        done = terminated or turnicated
        if done:
            break

    queue.put(
        (job_id, observations, actions, log_probs, episode_rewards, ep_timestep, frames)
    )
