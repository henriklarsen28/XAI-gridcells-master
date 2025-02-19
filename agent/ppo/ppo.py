import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import multiprocessing as mp
import random as rd
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from torch import nn
from transformer_decoder import Transformer
from transformer_decoder_policy import TransformerPolicy

torch.autograd.set_detect_anomaly(True)

from action import evaluate, get_action
from rollout import run_episode

from env import SunburstMazeContinuous
from utils import add_to_sequence, create_gif, padding_sequence

map_path_random = os.path.join(project_root, "env/random_generated_maps/goal")
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


def make_envs(env: dict):
    """def _init():
    new_env = SunburstMazeContinuous(
        maze_file=env.maze_file,
        render_mode=env.render_mode,
        rewards=env.rewards,
        fov=env.fov,
        ray_length=env.ray_length,
        number_of_rays=env.number_of_rays,
    )
    return new_env"""
    """new_env = gym.make(
            "SunburstMazeContinuous-v0",
            maze_file=env_params["maze_file"],
            max_episode_steps=env_params["max_steps_per_episode"],
            render_mode=None,
            random_start_position=env_params["random_start_position"],
            rewards=env_params["rewards"],
            fov=env_params["fov"],
            ray_length=env_params["ray_length"],
            number_of_rays=env_params["number_of_rays"],
        )"""

    def _init():
        new_env = gym.make(
            "SunburstMazeContinuous-v0",
            maze_file=env.get_wrapper_attr("maze_file"),
            max_steps_per_episode=env.get_wrapper_attr("max_steps_per_episode"),
            render_mode=env.get_wrapper_attr("render_mode"),
            random_start_position=env.get_wrapper_attr("random_start_position"),
            rewards=env.get_wrapper_attr("rewards"),
            fov=env.get_wrapper_attr("fov"),
            ray_length=env.get_wrapper_attr("ray_length"),
            number_of_rays=env.get_wrapper_attr("number_of_rays"),
        )
        new_env = TimeLimit(
            new_env,
            env.get_wrapper_attr("max_steps_per_episode"),
        )
        return new_env

    return _init


class PPO_agent:

    def __init__(self, env: SunburstMazeContinuous, device, config):
        wandb.login()
        self.config = config
        self.run = wandb.init(project="sunburst-maze-continuous", config=self)

        self.gif_path = f"./gifs/{self.run.name}"

        # Create the nessessary directories
        if not os.path.exists(self.gif_path):
            os.makedirs(self.gif_path)

        model_path = f"./model/transformers/ppo/model_{self.run.name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

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
        self.train_device = device
        self.rollout_device = torch.device("cpu")

        self.env_2_id = env_2_id_dict()

        self.policy_network = TransformerPolicy(
            input_dim=self.obs_dim,
            output_dim=self.act_dim,
            block_size=self.sequence_length,
            num_envs=len(self.env_2_id),
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=self.rollout_device,
        )

        self.critic_network = Transformer(
            input_dim=self.obs_dim,
            output_dim=1,
            block_size=self.sequence_length,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=self.rollout_device,
        )

        self.policy_network.to(self.rollout_device)
        self.critic_network.to(self.rollout_device)

        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=self.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_network.parameters(),
            lr=self.learning_rate,
        )

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5).to(
            self.train_device
        )
        self.cov_mat = torch.diag(self.cov_var).to(self.train_device)

    def learn(self, total_timesteps):

        timestep_counter = 0
        iteration_counter = 0

        while timestep_counter < total_timesteps:

            """self.env = self.random_maps(
                self.env, random_map=True, iteration_counter=iteration_counter
            )"""

            print("Iteration: ", iteration_counter)

            # Before rollout move the model to the cpu
            self.policy_network.device = self.rollout_device
            self.policy_network.to(self.rollout_device)
            self.critic_network.device = self.rollout_device
            self.critic_network.to(self.rollout_device)
            self.cov_mat = self.cov_mat.to(self.rollout_device)

            (
                obs_batch,
                actions_batch,
                log_probs_batch,
                env_classes_target_batch,
                rtgs_batch,
                lens,
                frames,
            ) = self.rollout(iteration_counter)

            # After rollout move the model to the device
            obs_batch = obs_batch.to(self.train_device)
            actions_batch = actions_batch.to(self.train_device)
            log_probs_batch = log_probs_batch.to(self.train_device)
            env_classes_target_batch = env_classes_target_batch.to(self.train_device)
            rtgs_batch = rtgs_batch.to(self.train_device)

            self.policy_network.device = self.train_device
            self.policy_network.to(self.train_device)
            self.critic_network.device = self.train_device
            self.critic_network.to(self.train_device)
            self.cov_mat = self.cov_mat.to(self.train_device)

            # Minibatches
            """minibatches = self.generate_minibatches(
                obs_batch,
                actions_batch,
                log_probs_batch,
                env_classes_batch,
                env_classes_target_batch,
                rtgs_batch,
            )"""

            timestep_counter += sum(lens)
            iteration_counter += 1

            """for (
                obs_batch,
                actions_batch,
                log_probs_batch,
                env_classes_batch,
                env_classes_target_batch,
                rtgs_batch,
            ) in minibatches:"""

            # print("Obs: ", obs, obs.shape)
            # Calculate the advantages
            value, _, _ = evaluate(
                obs_batch,
                actions_batch,
                self.policy_network,
                self.critic_network,
                self.cov_mat,
            )
            rtgs_batch = rtgs_batch.unsqueeze(1)

            advantages = rtgs_batch - value.detach()

            # Normalize the advantages
            advantages = (rtgs_batch - rtgs_batch.mean()) / (rtgs_batch.std() + 1e-10)
            advantages = rtgs_batch
            for _ in range(self.n_updates_per_iteration):
                value, current_log_prob, env_classes_batch = evaluate(
                    obs_batch,
                    actions_batch,
                    self.policy_network,
                    self.critic_network,
                    self.cov_mat,
                )
                entropy = self.compute_entropy(current_log_prob)

                ratio = torch.exp(current_log_prob - log_probs_batch)

                surrogate_loss1 = ratio * advantages

                surrogate_loss2 = (
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                )
                policy_loss_ppo = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()
                env_class_loss = F.cross_entropy(
                    env_classes_batch, env_classes_target_batch.float()
                )


                policy_loss = policy_loss_ppo - 0.01 * entropy

                critic_loss = nn.MSELoss()(value, rtgs_batch)

                self.env_network_backprop(env_class_loss)

                print("Policy loss step")
                self.policy_network.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.critic_network.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optimizer.step()
                print("After")

            gif = None
            if frames:

                gif = create_gif(
                    gif_path=f"./{self.gif_path}/{iteration_counter}.gif", frames=frames
                )
                frames.clear()

            wandb.log(
                {
                    "Episode": lens,
                    "Reward per episode": rtgs_batch.mean().item(),
                    "Policy_ppo loss": policy_loss_ppo.item(),
                    "Env_class loss": env_class_loss.item(),
                    "Policy loss": policy_loss.item(),
                    "Critic loss": critic_loss.item(),
                    "Environment": self.env_2_id[self.env.maze_file],
                    "Steps done": lens[0],
                    "Gif:": (wandb.Video(gif, fps=4, format="gif") if gif else None),
                },
                commit=True,
            )

            if iteration_counter % self.save_interval == 0:
                torch.save(
                    self.policy_network.state_dict(),
                    f"./model/transformers/ppo/model_{self.run.name}/policy_network_{iteration_counter}.pth",
                )
                torch.save(
                    self.critic_network.state_dict(),
                    f"./model/transformers/ppo/model_{self.run.name}/critic_network_{iteration_counter}.pth",
                )

    def env_network_backprop(self, env_class_loss):
        # Freeze the rest of the policy network
        for param in self.policy_network.blocks.parameters():
            param.requires_grad = False
        for param in self.policy_network.ln_f.parameters():
            param.requires_grad = False
        for param in self.policy_network.output.parameters():
            param.requires_grad = False
        for param in self.policy_network.env_class.parameters():
                param.requires_grad = True

        self.policy_optimizer.zero_grad()
        env_class_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Unfreeze the rest of the policy network
        for param in self.policy_network.parameters():
            param.requires_grad = True

        # Freeze the env class network
        for param in self.policy_network.env_class.parameters():
            param.requires_grad = False

    def rollout(self, iteration_counter):
        observations = []
        actions = []
        log_probs = []
        rewards = []
        env_classes_target = []
        rtgs = []
        lens = []
        frames = []

        episode_rewards = []

        timesteps = 0

        state_sequence = deque(maxlen=self.sequence_length)

        while timesteps < self.batch_size:
            episode_rewards = []
            # env_params = self.env.get_params()
            # envs = AsyncVectorEnv([make_envs(self.env) for _ in range(4)])

            # Reset the environment. sNote that obs is short for observation.
            self.env = self.random_maps(self.env, random_map=True)

            state, _ = self.env.reset()
            done = False

            for ep_timestep in range(self.max_steps):
                timesteps += 1
                state_sequence = add_to_sequence(
                    state_sequence, state, self.rollout_device
                )
                tensor_sequence = torch.stack(list(state_sequence), dim=0)
                tensor_sequence = padding_sequence(
                    tensor_sequence, self.sequence_length, self.rollout_device
                )
                action, log_prob = get_action(
                    tensor_sequence, self.policy_network, self.cov_mat
                )
                action = action[0]
                state, reward, terminated, turnicated, _ = self.env.step(action)

                if (
                    self.env.render_mode == "rgb_array"
                    and iteration_counter % 30 == 0
                    and len(rewards) == 0
                ):  # Create gif on the first episode in the rollout
                    frame = self.env.render()
                    if type(frame) == np.ndarray:
                        frames.append(frame)

                if self.env.render_mode == "human":
                    self.env.render()

                observations.append(tensor_sequence)
                actions.append(action)
                log_probs.append(log_prob)
                env_classes_target.append(torch.nn.functional.one_hot(torch.tensor(self.env_2_id[self.env.maze_file]), num_classes=len(self.env_2_id)))
                episode_rewards.append(reward)

                done = terminated or turnicated
                if done:
                    break

            lens.append(ep_timestep + 1)
            rewards.append(torch.tensor(episode_rewards))

        print("Timesteps: ", timesteps)

        # Reshape the data
        obs = torch.stack(observations).to(self.rollout_device)
        actions = torch.tensor(actions).to(self.rollout_device)
        log_probs = torch.tensor(log_probs).to(self.rollout_device)
        env_classes_target = torch.stack(env_classes_target).to(self.rollout_device)
        rtgs = self.compute_rtgs(rewards)
        # rtgs = self.compute_gae(rewards, obs, self.gamma, self.gae_lambda)
        # print(rtgs.shape, actions.shape)
        # Create a sequence of rtgs

        return (
            obs,
            actions,
            log_probs,
            env_classes_target,
            rtgs,
            lens,
            frames,
        )

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
        rtgs = torch.tensor(rtgs, dtype=torch.float, device=self.rollout_device)
        return rtgs

    def compute_gae(self, rewards, states, gamma=0.99, lam=0.95):
        """Computes Generalized Advantage Estimation (GAE)."""

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
        return advantages

    def compute_entropy(self, log_probs):
        return torch.mean(-log_probs)

    def pad_rtgs(self, rtgs):

        processed_rtgs = []

        for i in range(len(rtgs)):
            rtg_tensor = torch.tensor(
                rtgs[:i], dtype=torch.float32, device=self.rollout_device
            )

            # If RTG sequence is too short, pad it
            if len(rtg_tensor) < self.sequence_length:
                padded_rtg = F.pad(
                    rtg_tensor, (self.sequence_length - len(rtg_tensor), 0)
                )  # Left-padding
            else:
                padded_rtg = rtg_tensor[
                    -self.sequence_length :
                ]  # Truncate from the beginning

            processed_rtgs.append(padded_rtg)

        return torch.stack(processed_rtgs)

    def __init_hyperparameters(self, config):
        self.clip_grad_normalization = config["clip_grad_normalization"]
        self.clip = config["clip"]
        self.learning_rate = config["learning_rate"]
        self.n_updates_per_iteration = config["n_updates_per_iteration"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.batch_size = config["batch_size"]
        self.mini_batch_size = config["mini_batch_size"]
        self.save_interval = config["save_interval"]
        # self.max_episodes = config["total_episodes"]
        self.max_steps = config["max_steps_per_episode"]
        self.render = config["render"]
        self.render_mode = config["render_mode"]

    def generate_minibatches(
        self, obs, actions, log_probs, env_classes, env_classes_target, rtgs
    ):
        minibatches = []
        for _ in range(self.n_updates_per_iteration):
            idxs = np.random.randint(0, len(obs), self.mini_batch_size)
            minibatches.append(
                (
                    obs[idxs],
                    actions[idxs],
                    log_probs[idxs],
                    env_classes[idxs],
                    env_classes_target[idxs],
                    rtgs[idxs],
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

    def load_model(self, policy_path, critic_path):
        self.policy_network.load_state_dict(torch.load(policy_path))
        self.critic_network.load_state_dict(torch.load(critic_path))
