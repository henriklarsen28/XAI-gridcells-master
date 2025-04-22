import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import multiprocessing
import multiprocessing as mp
import queue
import random as rd
import time
from collections import deque
from multiprocessing import Process, Queue

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import gc
import wandb
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformer_decoder import Transformer
from transformer_decoder_policy import TransformerPolicy
from transformer_decoder_decoupled import TransformerDecoupled
from transformer_decoder_decoupled_policy import TransformerPolicyDecoupled
#from transformer_decoder_combined import Transformer

# from gated_transformer_decoder_combined import Transformer

torch.autograd.set_detect_anomaly(True)

# from action import  kl_divergence

from env import SunburstMazeContinuous
from utils import add_to_sequence, create_gif, padding_sequence

map_path_random = os.path.join(project_root, "env/random_generated_maps/goal/stretched")
map_path_random_files = [
    os.path.join(map_path_random, f)
    for f in os.listdir(map_path_random)
    if os.path.isfile(os.path.join(map_path_random, f))
]


def env_2_id_dict() -> dict:
    env_to_id = {}
    for i, env in enumerate(map_path_random_files):
        env_to_id[env] = i
    return env_to_id


def id_2_env_dict() -> dict:
    id_to_env = {}
    for i, env in enumerate(map_path_random_files):
        id_to_env[i] = env
    return id_to_env


class PPO_agent:

    def __init__(self, env: SunburstMazeContinuous, device, config):
        self.config = config
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Hyperparameters
        self.__init_hyperparameters(config)

        transformer_param = config["transformer"]
        # Transformer params
        n_embd = transformer_param["n_embd"]  # Embedding dimension
        n_head = transformer_param[
            "n_head"
        ]  # Number of attention heads (in multi-head attention)
        n_layer = transformer_param["n_layer"]  # Number of decoder layers
        dropout = transformer_param["dropout"]  # Dropout probability
        self.sequence_length = transformer_param["sequence_length"]  # Replace value
        self.device = device

        self.env_2_id = env_2_id_dict()

        self.policy_params = {
            "input_dim": self.obs_dim,
            "output_dim": self.act_dim,
            "block_size": self.sequence_length,
            "num_envs": len(self.env_2_id),
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
            "device": self.device,
        }
        self.critic_params = {
            "input_dim": self.obs_dim,
            "output_dim": 1,
            "block_size": self.sequence_length,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "dropout": dropout,
            "device": self.device,
        }


        if self.decouple_pos:
            
            self.policy_network = TransformerPolicyDecoupled(
                input_dim=self.obs_dim,
                output_dim=self.act_dim,
                block_size=self.sequence_length,
                num_envs=len(self.env_2_id),
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                device=self.device,
            )
            self.critic_network = TransformerDecoupled(
                input_dim=self.obs_dim,
                output_dim=1,
                block_size=self.sequence_length,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                device=self.device,
            )
        else:
            self.policy_network = TransformerPolicy(
                input_dim=self.obs_dim,
                output_dim=self.act_dim,
                block_size=self.sequence_length,
                num_envs=len(self.env_2_id),
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                device=self.device,
            )

            self.critic_network = Transformer(
                input_dim=self.obs_dim,
                output_dim=1,
                block_size=self.sequence_length,
                n_embd=n_embd,
                n_head=n_head,
                n_layer=n_layer,
                dropout=dropout,
                device=self.device,
            )

        """self.policy_network = Transformer(
            input_dim=self.obs_dim,
            output_dim=self.act_dim,
            block_size=self.sequence_length,
            num_envs=len(self.env_2_id),
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=self.device,
        )"""
        


        self.policy_network.to(self.device)
        self.critic_network.to(self.device)


        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_network.parameters(),
            lr=self.learning_rate,
        )

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)


    def learn(self, total_timesteps, iteration_counter = 0):
        self.__init_learn()
        timestep_counter = 0

        while timestep_counter < total_timesteps:

            print("Iteration: ", iteration_counter)

            (
                obs_batch,
                actions_batch,
                log_probs_batch,
                batch_rews,
                rtgs,
                batch_env_classes_target,
                lens,
                frames,
            ) = self.rollout(iteration_counter)

            # rollouts = self.rollout(iteration_counter)  # Collect rollouts
            # minibatches = self.batch_rollouts(rollouts)  # Create mini-batches
            """
            # Minibatches
            minibatches = self.generate_minibatches(
                obs_batch,
                actions_batch,
                log_probs_batch,
                rtgs,
                batch_env_classes_target,
            )"""

            timestep_counter += sum(lens)
            iteration_counter += 1
            """
            for (
                obs_batch,
                actions_batch,
                log_probs_batch,
                rtgs,
                batch_env_classes_target,
            ) in minibatches:"""

            # print("Obs: ", obs, obs.shape)
            # Calculate the advantages
            value, _, _, _ = self.evaluate(
                obs_batch,
                actions_batch,
            )
            # print(value, value.shape)
            kl_div = self.kl_divergence(obs_batch, actions_batch)

            # advantages, returns = self.compute_gae(rewards_batch, value, dones_batch)

            advantages = rtgs - value.detach()

            # Normalize the advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
                # print(value, value.shape)

            for i in range(self.n_updates_per_iteration):
                print(f"Iteration: {iteration_counter}. Update: {i}")
                value_new, current_log_prob, entropy, env_classes = self.evaluate(
                    obs_batch, actions_batch
                )
                # print("Current log prob: ", current_log_prob)
                # print("Log probs batch: ", log_probs_batch)
                ratio = torch.exp(current_log_prob - log_probs_batch)

                surrogate_loss1 = ratio * advantages
                surrogate_loss2 = (
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                )

                policy_loss_ppo = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()
                env_class_loss = self.env_loss_factor * F.cross_entropy(
                    env_classes, batch_env_classes_target.float()
                )

                # print(env_class_loss)

                policy_loss = (
                    policy_loss_ppo + env_class_loss
                )  # - self.entorpy_coefficient * entropy
                # print("Kl",kl_div, "Entropy", entropy)

                critic_loss = nn.MSELoss()(value_new, rtgs)

                # Normalize the gradients
                self.policy_optimizer.zero_grad(set_to_none=True)
                policy_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    self.policy_network.parameters(), self.clip_grad_normalization
                )
                self.policy_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic_network.parameters(), self.clip_grad_normalization
                )
                self.critic_optimizer.step()
                self.entorpy_coefficient_decay()

            gif = None
            if frames:

                gif = create_gif(
                    gif_path=f"./{self.gif_path}/{iteration_counter}.gif", frames=frames
                )
                frames.clear()

            lens = np.array(lens)

            torch.cuda.empty_cache()
            wandb.log(
                {
                    "Timesteps": timestep_counter,
                    "Rewards per episode": rtgs.mean().item(),
                    # "Rewards mean": rewads_mean,
                    "Policy_ppo loss": policy_loss_ppo.mean().item(),
                    "Env_class loss": env_class_loss.item(),
                    "Policy loss": policy_loss.item(),
                    "Critic loss": critic_loss.item(),
                    # "Environment": self.env_2_id[self.env.maze_file],
                    "Steps done": lens.mean(),
                    "Entropy": entropy.item(),
                    "Entropy coefficient": self.entorpy_coefficient,
                    "KL divergence": kl_div.item(),
                    "Gif:": (wandb.Video(gif, fps=4, format="gif") if gif else None),
                },
                commit=True,
            )

            # Clean the variables
            del obs_batch
            del actions_batch
            del log_probs_batch
            del batch_rews
            del rtgs
            del batch_env_classes_target
            del lens
            del frames
            del policy_loss
            del critic_loss
            del policy_loss_ppo
            del env_class_loss
            del value
            del value_new
            del current_log_prob
            del entropy
            del env_classes
            del kl_div
            del ratio
            del surrogate_loss1
            del surrogate_loss2
            del advantages



            gc.collect()
            torch.cuda.empty_cache()

            if iteration_counter % self.save_interval == 0 or iteration_counter == 0:
                torch.save(
                    self.policy_network.state_dict(),
                    f"./models/transformers/ppo/{self.run.name}/policy_network_{iteration_counter}.pth",
                )
                torch.save(
                    self.critic_network.state_dict(),
                    f"./models/transformers/ppo/{self.run.name}/critic_network_{iteration_counter}.pth",
                )
                """torch.save(
                    self.network.state_dict(),
                    f"./model/transformers/ppo/model_{self.run.name}/network_{iteration_counter}.pth",
                )"""


    """ def run_episode(
        self, env: SunburstMazeContinuous, policy_network, render, i_so_far
    ):
        self.policy_optimizer.zero_grad()
        env_class_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Unfreeze the rest of the policy network
        for param in self.policy_network.blocks.parameters():
            param.requires_grad = True
        for param in self.policy_network.ln_f.parameters():
            param.requires_grad = True
        for param in self.policy_network.output.parameters():
            param.requires_grad = True

        # Freeze the env class network
        for param in self.policy_network.env_class.parameters():
            param.requires_grad = False

    def worker(
        self,
        env: SunburstMazeContinuous,
        policy_network,
        render,
        i_so_far,
        output_queue,
    ):
        try:
            result = self.run_episode(env, policy_network, render, i_so_far)
            (
                worker_obs,
                worker_acts,
                worker_log_probs,
                worker_rews,
                worker_dones,
                worker_env_classes_target,
                lens,
                frames,
            ) = result

            output_queue.put(
                (
                    worker_obs,
                    worker_acts,
                    worker_log_probs,
                    worker_rews,
                    worker_dones,
                    worker_env_classes_target,
                    lens,
                    frames,
                )
            )
        except Exception as e:
            print(f"Worker failed: {e}")
            output_queue.put(None)
    """

    def rollout(self, i_so_far):
        """
        Collects sequential data for a transformer-based PPO agent.

        Returns:
            batch_obs - Batched observations with padding.
            batch_acts - Batched actions with padding.
            batch_log_probs - Log probabilities of actions.
            batch_rtgs - Rewards-To-Go.
            batch_lens - Lengths of each episode.
            attention_masks - Masking for padded sequences.
            frames - Rendered frames for visualization.
        """

        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_dones = []
        batch_rtgs = []
        batch_lens = []
        batch_env_classes_target = []
        frames = []
        t = 0  # Total timesteps in batch

        while t < self.batch_size:
            ep_obs = deque(maxlen=self.sequence_length)
            next_ep_obs = deque(maxlen=self.sequence_length)
            ep_rews = []
            ep_dones = []

            # Reset environment
            self.env = self.random_maps(env=self.env, random_map=True)
            obs, _ = self.env.reset()

            done = False
            for ep_t in range(self.max_steps):
                t += 1

                obs = torch.tensor(obs, dtype=torch.float, device=self.device)
                ep_obs.append(obs)

                tensor_obs = torch.stack(list(ep_obs)).to(self.device)
                tensor_obs = self.preprocess_ep_obs(tensor_obs)

                batch_obs.append(tensor_obs)

                # Get action and log probability (transformer expects a full sequence, so we pass collected states)
                action, log_prob = self.get_action(tensor_obs)  # Pass full sequence

                obs, reward, terminated, truncated, _ = self.env.step(action)
                #obs = obs.flatten()
                if self.render_mode == "rgb_array" and i_so_far % 30 == 0:
                    frame = self.env.render()
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)

                if self.render_mode == "human":
                    self.env.render()

                done = terminated or truncated



                ep_rews.append(reward)
                ep_dones.append(done)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                env_id = torch.tensor(self.env_2_id[self.env.unwrapped.maze_file])
                batch_env_classes_target.append(
                    torch.nn.functional.one_hot(
                        env_id,
                        num_classes=len(self.env_2_id),
                    )
                )

                # print("Env", self.env.unwrapped.maze_file, self.env_2_id[self.env.unwrapped.maze_file])

                if done:
                    break

            # Store full episode sequence

            # batch_obs.append(torch.stack(ep_tensor_seq))
            batch_lens.append(ep_t + 1)
            batch_rews.append(torch.tensor(ep_rews, dtype=torch.float))
            # batch_values.append(torch.tensor(ep_values, dtype=torch.float))
            # batch_next_values.append(torch.tensor(ep_next_values, dtype=torch.float))
            #batch_dones.append(torch.tensor(ep_dones, dtype=torch.float))

        batch_obs = torch.stack(batch_obs)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(
            batch_log_probs, dtype=torch.float, device=self.device
        )
        batch_lens = torch.tensor(batch_lens)

        # Compute RTGs
        batch_rtgs = self.compute_rtgs(batch_rews)

        batch_env_classes_target = torch.stack(batch_env_classes_target).to(self.device)
        # advantages, returns = self.compute_gae(batch_rews, batch_values, batch_next_values, batch_dones)

        # print(batch_lens)
        return (
            batch_obs,
            batch_acts,
            batch_log_probs,
            batch_rews,
            batch_rtgs,
            batch_env_classes_target,
            batch_lens,
            frames,
        )

    def preprocess_ep_obs(self, ep_obs):
        # Convert sequence to tensor and pad if necessary
        seq_len = len(ep_obs)
        padded_obs = torch.zeros(self.sequence_length, *ep_obs[-1].shape).to(
            self.device
        )

        padded_obs[-seq_len:] = ep_obs  # Right-align sequence

        return padded_obs

    def get_action(self, obs):

        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)

        mean, _, _, _ = self.policy_network(obs)

        # mean, _, std, _, _ = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        # action = torch.tensor(action, dtype=torch.float32, device=self.device)
        action = action.squeeze(0)
        return action.cpu().detach().numpy(), log_prob.detach()[0]

    def evaluate(self, obs, actions):
        # obs = self.preprocess_ep_obs(obs)
        V, _ = self.critic_network(obs)
        #mean, V, _, env_class, _ = self.policy_network(obs)
        V = V.squeeze()
        mean, _, env_class, _ = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return V, log_prob, entropy.mean(), env_class

    def kl_divergence(self, obs, actions):
        mean, _, _, _ = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
        old_dist = torch.distributions.MultivariateNormal(actions, self.cov_mat)

        return torch.distributions.kl_divergence(old_dist, dist).mean()

    def compute_rtgs(self, rewards):
        rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(rewards):
            discounted_reward = 0  # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for reward in reversed(ep_rews):
                discounted_reward = reward + discounted_reward * self.gamma
                rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        rtgs = torch.tensor(rtgs, dtype=torch.float, device=self.device)
        return rtgs

    def compute_gae(
        self,
        rewards,
        values,
        next_values,
        dones,
    ):
        episodes, steps, _ = rewards.shape

        # Initialize advantage and returns tensors
        advantages = torch.zeros_like(rewards, dtype=torch.float32)
        returns = torch.zeros_like(rewards, dtype=torch.float32)

        # Last advantage is initialized as zero
        last_advantage = torch.zeros(
            (episodes, 1), dtype=torch.float32, device=rewards.device
        )

        # Iterate backward through time steps
        for t in reversed(range(steps)):
            # Bootstrapped value for next step

            # Temporal difference error
            td_error = (
                rewards[:, t]
                + self.gamma * next_values[:, t] * (1 - dones[:, t])
                - values[:, t]
            )

            # GAE formula
            last_advantage = (
                td_error
                + self.gamma * self.gae_lambda * last_advantage * (1 - dones[:, t])
            )

            advantages[:, t] = last_advantage

        # Compute returns
        returns = advantages + values

        return advantages, returns

    """

    def compute_gae(self, rewards, states, gamma=0.99, lam=0.95):

        advantages = torch.zeros_like(
            states[:, 0, 0], dtype=torch.float32, device=self.rollout_device
        )
        last_advantage = 0
        values, _ = self.critic_network(states)
        values = values.detach()
        for ep_rewards in rewards:

            for t in reversed(range(len(ep_rewards))):
                delta = ep_rewards[t] + gamma * values[t + 1] - values[t]
                last_advantage = delta + gamma * lam * last_advantage
                advantages[t] = last_advantage

        advantages = advantages.unsqueeze(1)
        return advantages"""

    def compute_entropy(self, log_probs):
        return torch.mean(-log_probs)

    def __init_hyperparameters(self, config):
        self.clip_grad_normalization = config["PPO"]["clip_grad_normalization"]
        self.clip = config["PPO"]["clip"]
        self.learning_rate = config["learning_rate"]
        self.n_updates_per_iteration = config["PPO"]["n_updates_per_iteration"]

        self.gamma = config["PPO"]["gamma"]
        self.gae_lambda = config["PPO"]["gae_lambda"]
        self.policy_params = config["PPO"]["policy_params"]
        self.kl_range = config["PPO"]["policy_kl_range"]
        self.batch_size = config["batch_size"]
        self.n_mini_batches = config["n_mini_batches"]
        self.save_interval = config["save_interval"]
        # self.max_episodes = config["total_episodes"]
        self.max_steps = config["max_steps_per_episode"]
        self.render = config["render"]
        self.render_mode = config["render_mode"]
        self.entorpy_coefficient = config["entropy"]["coefficient"]
        self.entropy_min = config["entropy"]["min"]
        self.entropy_step = (self.entorpy_coefficient - self.entropy_min) / config[
            "entropy"
        ]["step"]
        self.normalize_advantage = config["PPO"]["normalize_advantage"]
        self.env_loss_factor = config["PPO"]["env_loss_factor"]

        self.decouple_pos = config["transformer"]["decouple_positional_embedding"]

    def __init_learn(self):
        wandb.login()
        self.run = wandb.init(project="sunburst-maze-continuous", config=self)

        self.gif_path = f"./gifs/{self.run.name}"

        # Create the nessessary directories
        if not os.path.exists(self.gif_path):
            os.makedirs(self.gif_path)

        model_path = f"./models/transformers/ppo/{self.run.name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        config_path = f"{model_path}/config"
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        # Save the config file as json
        with open(f"{config_path}/policy_params.json", "w") as f:
            f.write(str(self.policy_params))

        with open(f"{config_path}/critic_params.json", "w") as f:
            f.write(str(self.critic_params))

        torch.save(
            self.policy_network.state_dict(),
            f"./models/transformers/ppo/{self.run.name}/policy_network_0.pth",
        )
        torch.save(
            self.critic_network.state_dict(),
            f"./models/transformers/ppo/{self.run.name}/critic_network_0.pth",
        )

    def entorpy_coefficient_decay(self):
        self.entorpy_coefficient -= self.entropy_step
        self.entorpy_coefficient = max(self.entropy_min, self.entorpy_coefficient)

    def generate_minibatches(
        self, obs, actions, log_probs, rtgs, env_class_target
    ):
        minibatches = []

        indices = torch.randperm(self.batch_size)
        mini_batch_size = self.batch_size // self.n_mini_batches
        for start in range(0, self.batch_size, mini_batch_size):
            end = start + mini_batch_size
            if end < self.batch_size:
                mini_batch_indices = indices[start:end]
                minibatches.append(
                    (
                        obs[mini_batch_indices],
                        actions[mini_batch_indices],
                        log_probs[mini_batch_indices],
                        rtgs[mini_batch_indices],
                        env_class_target[mini_batch_indices],
                    )
                )

        return minibatches

    def random_maps(
        self,
        env: SunburstMazeContinuous,
        random_map: bool = False,
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

        return env

    """def random_maps(
        self,
        env: SunburstMazeContinuous,
        random_map: bool = False,
    ):
        if random_map:
            # Select and load a new random map
            map_path = map_path_random_files[0]
            map_path_random_files.pop(0)
            map_path_random_files.append(map_path)
            env = gym.make(
                "SunburstMazeContinuous-v0",
                maze_file=map_path,
                max_steps_per_episode=env.max_steps_per_episode,
                render_mode=env.render_mode,
                random_start_position=env.random_start_position,
                random_goal_position=env.random_goal_position,
                rewards=env.rewards,
                fov=env.fov,
                ray_length=env.ray_length,
                number_of_rays=env.number_of_rays,
            )

        return env"""

    """def random_maps(
        env: SunburstMazeContinuous,
        random_map: bool = False,
    ):
        if random_map:
            # Select and load a new random map
            map_path = map_path_random_files[0]
            map_path_random_files.pop(0)
            map_path_random_files.append(map_path)

            env = SunburstMazeContinuous(
                maze_file=map_path,
                render_mode=env.render_mode,
                rewards=env.rewards,
                max_steps_per_episode=env.max_steps_per_episode,
                random_start_position=env.random_start_position,
                fov=env.fov,
                ray_length=env.ray_length,
                number_of_rays=env.number_of_rays,
            )

        return env"""
    def load_models(self):
        policy_network_path = self.config["policy_load_path"]
        critic_network_path = self.config["critic_load_path"]
        self.policy_network.load_state_dict(
            torch.load(
                policy_network_path
            )
        )
        self.critic_network.load_state_dict(
            torch.load(
                critic_network_path
            )
        )

    def load_model(self, policy_path, critic_path):
        self.policy_network.load_state_dict(torch.load(policy_path))
