"""
This file is used only to evaluate our trained policy/actor after
training in main.py with ppo.py. I wrote this file to demonstrate
that our trained policy exists independently of our learning algorithm,
which resides in ppo.py. Thus, we can test our trained policy without
relying on ppo.py.
"""

import copy
from collections import deque

import gymnasium as gym
import torch

from agent.ppo.transformer_decoder_decoupled_policy import TransformerPolicyDecoupled
from env import SunburstMazeContinuous
from utils.sequence_preprocessing import add_to_sequence

"""def random_maps(
    env: SunburstMazeContinuous,
    random_map: bool = False,
    map_path_random_files: list = None,
):

    if random_map:
        # Select and load a new random map
        map_path = map_path_random_files[0]
        map_path_random_files.pop(0)
        map_path_random_files.append(map_path)
        env = gym.make(
            "SunburstMazeContinuous-v0",
            maze_file=map_path,
            max_steps_per_episode=env.get_wrapper_attr("max_steps_per_episode"),
            render_mode=env.get_wrapper_attr("render_mode"),
            random_start_position=env.get_wrapper_attr("random_start_position"),
            random_goal_position=env.get_wrapper_attr("random_goal_position"),
            rewards=env.get_wrapper_attr("rewards"),
            fov=env.get_wrapper_attr("fov"),
            ray_length=env.get_wrapper_attr("ray_length"),
            number_of_rays=env.get_wrapper_attr("number_of_rays"),
        )

    return env"""


def _log_summary(ep_len, ep_ret, ep_num):
    """
    Print to stdout what we've logged so far in the most recent episode.

    Parameters:
            None

    Return:
            None
    """
    # Round decimal places for more aesthetic logging messages
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_ret}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)


def rollout(
    policy,
    actor_model,
    env: SunburstMazeContinuous,
    render,
    sequence_length,
    device,
    max_steps=None,
):
    """
    Returns a generator to roll out each episode given a trained policy and
    environment to test on.

    Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on
            render - Specifies whether to render or not

    Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.

    Note:
            If you're unfamiliar with Python generators, check this out:
                    https://wiki.python.org/moin/Generators
            If you're unfamiliar with Python "yield", check this out:
                    https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    """

    # Rollout until user kills process
    collected_observation_sequences = deque()
    collected_positions = []

    print("Using model", actor_model)

    while True:

        obs, _ = env.reset()
        done = False

        # number of timesteps so far
        t = 0

        # Logging data
        ep_len = 0  # episodic length
        ep_ret = 0  # episodic return

        ep_obs = deque(maxlen=sequence_length)

        while t < max_steps and not done:
            t += 1

            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            # Render environment if specified, off by default
            if render:
                env.render()

            ep_obs.append(obs)

            tensor_obs = torch.stack(list(ep_obs)).to(device)
            tensor_obs = preprocess_ep_obs(tensor_obs, sequence_length, device)
            tensor_obs = tensor_obs.unsqueeze(0)
            if t > sequence_length:
                observation_sequence = list(tensor_obs.squeeze(0))
                collected_observation_sequences.append(
                    copy.deepcopy(observation_sequence)
                )
                # print("Length of collected observation sequences", len(collected_observation_sequences))
            position = (int(env.position[0]), int(env.position[1]))
            collected_positions.append(copy.deepcopy(position))

            # Query deterministic action from policy and run it
            action, _, _, _ = policy(tensor_obs)
            action = action[0].detach().cpu().numpy()
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            # Sum all episodic rewards as we go along
            ep_ret += rew

        # Track episodic length
        ep_len = t

        # returns episodic length and return in this iteration
        yield ep_len, ep_ret, collected_observation_sequences, collected_positions


def preprocess_ep_obs(ep_obs, sequence_length, device):
    # Convert sequence to tensor and pad if necessary
    seq_len = len(ep_obs)
    padded_obs = torch.zeros(sequence_length, *ep_obs[-1].shape).to(device)

    padded_obs[-seq_len:] = ep_obs  # Right-align sequence

    return padded_obs


def update_model(actor_model_paths, curr_index):
    if ep_num == len(actor_model_paths) - 1:
        print("Reached the end of the model.")
        return None
    # return the next model in the list not depending on the current episode number

def eval_policy(
    policy: TransformerPolicyDecoupled,
    actor_model_paths,
    env,
    sequence_length,
    device,
    render=False,
    max_steps=None,
):
    """
    The main function to evaluate our policy with. It will iterate a generator object
    "rollout", which will simulate each episode and return the most recent episode's
    length and return. We can then log it right after. And yes, eval_policy will run
    forever until you kill the process.

    Parameters:
            policy - The trained policy to test, basically another name for our actor model
            env - The environment to test the policy on
            render - Whether we should render our episodes. False by default.

    Return:
            None

    NOTE: To learn more about generators, look at rollout's function description
    """
    # Rollout with the policy and environment, and log each episode's data
    # collected_observations = deque ()

    # Load in the actor model saved by the PPO algorithm
    curr_model_index = 0
    actor_model = actor_model_paths[curr_model_index]

    policy.load_state_dict(torch.load(actor_model, map_location=device))

    collected_observations = deque()

    max_episodes = 120

    model_num = int(actor_model.split("_")[-1].split(".")[0])

    for ep_num, (
        ep_len,
        ep_ret,
        collected_observation_sequences,
        collected_positions,
    ) in enumerate(
        rollout(
            policy,
            actor_model,
            env,
            render,
            sequence_length,
            device,
            max_steps=max_steps,
        )
    ):
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
        collected_observations.append(
            (
                # copy.deepcopy(collected_observation_sequences),
                None,
                copy.deepcopy(collected_positions),
                model_num,
            )
        )
        if ep_num % 5 == 0:
            if ep_num == 0:
                continue
            else:
                actor_model = actor_model_paths[curr_model_index + 1]
                curr_model_index += 1
                policy.load_state_dict(torch.load(actor_model, map_location=device))
                # extract the model number from the path
                model_num = int(actor_model.split("_")[-1].split(".")[0])
                print("Model number: ", model_num)

        """if ep_num == 25:
            actor_model = actor_model_paths[1]
            policy.load_state_dict(torch.load(actor_model, map_location=device))
            print("Using model: ", actor_model)
        if ep_num == 75:
            actor_model = actor_model_paths[2]
            policy.load_state_dict(torch.load(actor_model, map_location=device))
            print("Using model: ", actor_model)"""

        if ep_num == max_episodes:
            break

        ep_num += 1

        if ep_num % 3 == 0:
            print("Collected observations", len(collected_observations))

            yield copy.deepcopy(collected_observations)
            collected_observations.clear()
            print("Cleared collected observations")

    yield copy.deepcopy(collected_observations)
