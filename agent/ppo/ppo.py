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
from attention_pooling import AttentionPooling
from gated_transformer_decoder import Transformer
from gated_transformer_decoder_policy import TransformerPolicy
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import TimeLimit
from torch import nn

# from gated_transformer_decoder_combined import Transformer

torch.autograd.set_detect_anomaly(True)

from action import evaluate, get_action, kl_divergence
from rollout import run_episode

from env import SunburstMazeContinuous
from utils import add_to_sequence, create_gif, padding_sequence

map_path_random = os.path.join(project_root, "env/random_generated_maps/goal/large")
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
            max_episode_steps=env.get_wrapper_attr("max_steps_per_episode"),
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
        self.device = device

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

        """self.network = Transformer(
            input_dim=self.obs_dim,
            output_dim=self.act_dim,
            num_envs=len(self.env_2_id),
            block_size=self.sequence_length,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=self.rollout_device,
        )"""

        """self.policy_attention_pooling = AttentionPooling(
            hidden_dim=self.act_dim,"""

        self.policy_network.to(self.device)
        self.critic_network.to(self.device)

        """self.policy_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )"""

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

    def learn(self, total_timesteps):

        timestep_counter = 0
        iteration_counter = 0

        while timestep_counter < total_timesteps:

            print("Iteration: ", iteration_counter)

            (
                obs_batch,
                actions_batch,
                log_probs_batch,
                env_classes_target_batch,
                _, #value
                _, #advantages,
                rtgs_batch, #returns_batch,
                _, #rewads_mean,
                lens,
                frames,
            ) = self.rollout(iteration_counter)

            # Minibatches
            minibatches = self.generate_minibatches(
                obs_batch,
                actions_batch,
                log_probs_batch,
                env_classes_target_batch,
                #value,
                # advantages,
                rtgs_batch,
            )

            timestep_counter += sum(lens)
            iteration_counter += 1

            for (
                obs_batch,
                actions_batch,
                log_probs_batch,
                env_classes_target_batch,
                rtgs_batch,
            ) in minibatches:

                # print("Obs: ", obs, obs.shape)
                # Calculate the advantages
                value, _, _, _ = evaluate(
                    obs_batch,
                    actions_batch,
                    self.policy_network,
                    self.critic_network,
                    self.cov_mat,
                )
                # print(value, value.shape)

                """# Normalize rewards
                all_rewards = np.concatenate(rewards_batch)
                mean = np.mean(all_rewards)
                std = np.std(all_rewards) + 1e-8
                rewards_batch = [(np.array(rewards) - mean) / std for rewards in rewards_batch]

                rewards_batch = list(rewards_batch)"""

                # advantages, returns = self.compute_gae(rewards_batch, value, dones_batch)

                # rtgs_batch = rtgs_batch.unsqueeze(1)

                # print("RTGS: ", rtgs_batch, rtgs_batch.shape)
                # print("Value: ", value, value.shape)

                advantages = rtgs_batch - value.clone().detach()

                # Normalize the advantages
                # advantages = rtgs_batch

                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
                for _ in range(self.n_updates_per_iteration):
                    value_new, current_log_prob, entropy, env_classes_batch = evaluate(
                        obs_batch,
                        actions_batch,
                        self.policy_network,
                        self.critic_network,
                        self.cov_mat,
                    )

                    kl_div = kl_divergence(
                        obs_batch, actions_batch, self.policy_network, self.cov_mat
                    )

                    ratio = torch.exp(
                        torch.clamp(
                            current_log_prob - log_probs_batch, min=-20.0, max=5.0
                        )
                    )

                    surrogate_loss1 = ratio * advantages

                    """policy_loss_ppo = (
                        -torch.min(surrogate_loss1, surrogate_loss2)
                    ).mean()"""
                    env_class_loss = F.cross_entropy(
                        env_classes_batch, env_classes_target_batch.float()
                    )

                    """policy_loss = (
                        policy_loss_ppo
                        + self.policy_params * kl_div
                        - self.entorpy_coefficient * entropy
                    )"""
                    policy_loss_ppo = -torch.where(
                        (kl_div >= self.kl_range)
                        & (surrogate_loss1 > advantages),
                        surrogate_loss1 - self.policy_params * kl_div,
                        surrogate_loss1 - self.kl_range,
                    )

                    policy_loss = policy_loss_ppo.mean() - self.entorpy_coefficient * entropy
                    # print("Kl",kl_div, "Entropy", entropy)

                    value_clipped = value.detach() + torch.clamp(
                        value_new - value.detach(),
                        -self.config["PPO"]["clip"],
                        self.config["PPO"]["clip"],
                    )

                    critic_loss = torch.max(
                        nn.MSELoss()(value_new, rtgs_batch),
                        nn.MSELoss()(value_clipped, rtgs_batch),
                    )

                    # loss = policy_loss + 0.5 * critic_loss
                    # Normalize the gradients

                    print("Policy loss step")
                    self.policy_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy_network.parameters(), self.clip_grad_normalization
                    )
                    self.policy_optimizer.step()

                    # self.env_network_backprop(env_class_loss)
                    # self.network.zero_grad()
                    # loss.backward(retain_graph=True)
                    # self.policy_optimizer.step()

                    self.critic_optimizer.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_network.parameters(), self.clip_grad_normalization
                    )
                    self.critic_optimizer.step()
                    print("After")
                    self.entorpy_coefficient_decay()

            gif = None
            if frames:

                gif = create_gif(
                    gif_path=f"./{self.gif_path}/{iteration_counter}.gif", frames=frames
                )
                frames.clear()

            lens = np.array(lens)

            wandb.log(
                {
                    "Episode": lens,
                    "Rewards per episode": rtgs_batch.mean().item(),
                    #"Rewards mean": rewads_mean,
                    "Policy_ppo loss": policy_loss_ppo.mean().item(),
                    "Env_class loss": env_class_loss.item(),
                    "Policy loss": policy_loss.item(),
                    "Critic loss": critic_loss.item(),
                    "Environment": self.env_2_id[self.env.maze_file],
                    "Steps done": lens.mean(),
                    "Entropy": entropy.item(),
                    "Entropy coefficient": self.entorpy_coefficient,
                    "KL divergence": kl_div.item(),
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
                """torch.save(
                    self.network.state_dict(),
                    f"./model/transformers/ppo/model_{self.run.name}/network_{iteration_counter}.pth",
                )"""

    def env_network_backprop(self, env_class_loss):
        """# Freeze the rest of the policy network
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
        for param in self.policy_network.blocks.parameters():
            param.requires_grad = True
        for param in self.policy_network.ln_f.parameters():
            param.requires_grad = True
        for param in self.policy_network.output.parameters():
            param.requires_grad = True

        # Freeze the env class network
        for param in self.policy_network.env_class.parameters():
            param.requires_grad = False"""

    def rollout(self, iteration_counter):
        observations = []
        episode_observations = []
        actions = []
        log_probs = []
        rewards = []
        env_classes_target = []
        rtgs = []
        dones = []
        lens = []
        frames = []

        episode_rewards = []

        timesteps = 0

        state_sequence = deque(maxlen=self.sequence_length)

        while timesteps < self.batch_size:
            episode_rewards = []
            done_list = []
            ep_obs = []
            # env_params = self.env.get_params()
            # envs = AsyncVectorEnv([make_envs(self.env) for _ in range(4)])

            # Reset the environment. sNote that obs is short for observation.
            #self.env = self.random_maps(self.env, random_map=True)

            state, _ = self.env.reset()
            done = False

            for ep_timestep in range(self.max_steps):
                timesteps += 1
                state_sequence = add_to_sequence(state_sequence, state, self.device)
                tensor_sequence = torch.stack(list(state_sequence), dim=0)
                tensor_sequence = padding_sequence(
                    tensor_sequence, self.sequence_length, self.device
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

                ep_obs.append(tensor_sequence)
                observations.append(tensor_sequence)
                actions.append(action)
                log_probs.append(log_prob)
                env_classes_target.append(
                    torch.nn.functional.one_hot(
                        torch.tensor(self.env_2_id[self.env.maze_file]),
                        num_classes=len(self.env_2_id),
                    )
                )
                episode_rewards.append(reward)

                done = terminated or turnicated
                done_list.append(done)
                if done:
                    break

            """queue = mp.Queue()
            processes = []
            for i in range(4):
                p = mp.Process(target=run_episode, args=(i,self.env, iteration_counter, self.sequence_length, self.device, False, self.policy_network, queue))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            print("Processes joined")
            results = [queue.get() for p in processes]"""

            lens.append(ep_timestep + 1)
            episode_observations.append(ep_obs)
            rewards.append(torch.tensor(episode_rewards))
            dones.append(done_list)

        print("Timesteps: ", timesteps)

        # Reshape the data

        obs = torch.stack(observations).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        log_probs = torch.tensor(log_probs).to(self.device)
        env_classes_target = torch.stack(env_classes_target).to(self.device)
        all_rewards = np.concatenate(rewards)
        rewards_mean = np.mean(all_rewards)
        rtgs = self.compute_rtgs(rewards)
        # rtgs = None
        #advantages, returns, value = self.compute_gae(
        #    rewards, episode_observations, dones
        #)
        # print(rtgs, rtgs.shape)
        # print(rtgs.shape, actions.shape)
        # Create a sequence of rtgs

        return (
            obs,
            actions,
            log_probs,
            env_classes_target,
            None, #value,
            None, # advantages,
            rtgs,
            rewards_mean,
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
        rtgs = torch.tensor(rtgs, dtype=torch.float, device=self.device)
        return rtgs

    def compute_gae(self, rewards, obs, dones):

        advantages = []
        returns = []
        values = []

        for ep_rewards, ep_obs, ep_dones in reversed(list(zip(rewards, obs, dones))):
            ep_obs = torch.stack(ep_obs).to(self.device)
            ep_values, _ = self.critic_network(ep_obs)
            ep_len = len(ep_rewards)
            adv = np.zeros(ep_len)
            ret = np.zeros(ep_len)
            last_adv = 0  # Initialize last advantage
            last_return = 0  # Initialize last return

            # Compute GAE and returns in reverse order
            for t in reversed(range(ep_len)):
                next_value = (
                    ep_values[t + 1] if (t + 1 < ep_len and not ep_dones[t]) else 0
                )
                delta = ep_rewards[t] + self.gamma * next_value - ep_values[t]
                adv[t] = delta + self.gamma * self.gae_lambda * last_adv * (
                    1 - ep_dones[t]
                )  # Stop bootstrapping if done
                last_adv = adv[t]  # Update last advantage

                ret[t] = ep_rewards[t] + self.gamma * last_return * (
                    1 - ep_dones[t]
                )  # Stop discounting if done
                last_return = ret[t]  # Update return

                advantages.append(adv[t])
                returns.append(ret[t])
                values.append(ep_values[t])

        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float, device=self.device)
        values = torch.tensor(values, dtype=torch.float, device=self.device)
        advantages = advantages.flatten()
        returns = returns.flatten()
        values = values.flatten()

        return advantages, returns, values
        """ # Iterate through episodes in reverse order
        for ep_rews, ep_dones in zip(reversed(rewards), reversed(dones)):
            for t in reversed(range(len(ep_rews))):
                delta = (
                    ep_rews[t]
                    + self.gamma * values_extended[t + 1] * (1 - ep_dones[t])
                    - values[t]
                )
                gae = delta + self.gamma * self.gae_lambda * (1 - ep_dones[t]) * gae
                advantages.insert(0, gae)
        """
        advantages = torch.tensor(advantages, dtype=torch.float, device=self.device)

        # Compute value targets as advantages + value estimates
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
        # self.mini_batch_size = config["mini_batch_size"]
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

    def entorpy_coefficient_decay(self):
        self.entorpy_coefficient -= self.entropy_step
        self.entorpy_coefficient = max(self.entropy_min, self.entorpy_coefficient)

    def generate_minibatches(
        self, obs, actions, log_probs, env_classes_target, returns
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
                        env_classes_target[mini_batch_indices],
                        #values[mini_batch_indices],
                        #advantages[mini_batch_indices],
                        returns[mini_batch_indices],
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
