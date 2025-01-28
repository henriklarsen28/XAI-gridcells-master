import os
from collections import deque

import numpy as np
import torch
import wandb
from torch import nn
torch.autograd.set_detect_anomaly(True)
from transformer_decoder import Transformer
from transformer_decoder_policy import TransformerPolicy

from env import SunburstMazeContinuous
from utils.sequence_preprocessing import (
    add_to_sequence,
    padding_sequence,
    padding_sequence_int,
)
from utils.state_preprocess import state_preprocess_continuous


class PPO_agent:

    def __init__(self, env: SunburstMazeContinuous, device, config):
        wandb.login()
        self.config = config
        self.run = wandb.init(project="sunburst-maze-continuous", config=self)

        gif_path = f"./gifs/{self.run.name}"

        # Create the nessessary directories
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)

        model_path = f"./model/transformers/ppo/model_{self.run.name}"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        self.env = env
        self.obs_dim = config["observation_size"]
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

        self.policy_network = TransformerPolicy(
            input_dim=self.obs_dim,
            output_dim=self.act_dim,
            block_size=self.sequence_length,
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

        self.action_low = torch.tensor(env.action_space.low).to(self.device)
        self.action_high = torch.tensor(env.action_space.high).to(self.device)

    def learn(self, total_timesteps):
        print("Learning")

        timestep_counter = 0
        iteration_counter = 0

        while timestep_counter < total_timesteps:

            obs, actions, log_probs, rtgs, lens, frames = self.rollout(iteration_counter)
            # Minibatches
            minibatches = self.generate_minibatches(obs, actions, log_probs, rtgs)

            timestep_counter += sum(lens)
            iteration_counter += 1

            for obs_batch, actions_batch, log_probs_batch, rtgs_batch in minibatches:
                
            
                #print("Obs: ", obs, obs.shape)
                # Calculate the advantages
                value, _ = self.evaluate(obs_batch, actions_batch)
                rtgs_batch = rtgs_batch.unsqueeze(2)

                advantages = rtgs_batch - value.detach()


                # Normalize the advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
                for _ in range(self.n_updates_per_iteration):
                    value, current_log_prob = self.evaluate(obs_batch, actions_batch)

                    #print(current_log_prob)
                    ratio = torch.exp(current_log_prob - log_probs_batch)
                    ratio = ratio.unsqueeze(2)
    
                    surrogate_loss1 = ratio * advantages

                    #print("Surrogate loss1: ", surrogate_loss1, surrogate_loss1.shape)
                    surrogate_loss2 = (
                        torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                    )
                    # Increase the size of rtgs to be 300 x 15
    
                    policy_loss = (-torch.min(surrogate_loss1, surrogate_loss2)).mean()

                    critic_loss = nn.MSELoss()(value, rtgs_batch)

                    print("Policy loss step")
                    self.policy_network.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    self.policy_optimizer.step()
                    

                    self.critic_network.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    print("After")

            gif = None
            if frames:
                if os.path.exists("./gifs") is False:
                    os.makedirs("./gifs")

                gif = self.env.create_gif(
                    gif_path=f"./gifs/{iteration_counter}.gif", frames=frames
                )
                frames.clear()

            wandb.log(
                {
                    "Episode": lens,
                    "Reward per episode": rtgs.mean().item(),
                    "Policy loss": policy_loss.item(),
                    "Critic loss": critic_loss.item(),
                    "Steps done": lens[0],
                    "Gif:": (wandb.Video(gif, fps=4, format="gif") if gif else None),
                },
                commit=True,
            )

            if iteration_counter % self.update_frequency == 0:
                torch.save(
                    self.policy_network.state_dict(),
                    f"./model/transformers/ppo/model_{self.run.name}/policy_network_{iteration_counter}.pth",
                )
                torch.save(
                    self.critic_network.state_dict(),
                    f"./model/transformers/ppo/model_{self.run.name}/critic_network_{iteration_counter}.pth",
                )

    def rollout(self, iteration_counter, render=False):
        observations = []
        actions = []
        log_probs = []
        rewards = []
        rtgs = []
        lens = []
        frames = []

        episode_rewards = []

        timesteps = 0

        state_sequence = deque(maxlen=self.sequence_length)

        while timesteps < self.batch_size:
            episode_rewards = []
            reward_sequence = deque(maxlen=self.sequence_length)
            state, _ = self.env.reset()
            done = False


            

            for ep_timestep in range(self.max_steps):
                timesteps += 1
                state = state_preprocess_continuous(state, device=self.device)
                state_sequence = add_to_sequence(state_sequence, state, self.device)
                tensor_sequence = torch.stack(list(state_sequence))
                tensor_sequence = padding_sequence(
                    tensor_sequence, self.sequence_length, self.device
                )
                action, log_prob = self.get_action(tensor_sequence)
                last_action = action[:,-1,:].cpu().detach().numpy()

                #last_log_prob = log_prob[-1,-1]
                
                state, reward, done, turnicated, _ = self.env.step(last_action[0])

                if self.render_mode == "rgb_array" and len(rewards) == 0: # Create gif on the first episode in the rollout
                    frame = self.env._render_frame()
                    if type(frame) == np.ndarray:
                        frames.append(frame)

                if self.render_mode == "human":
                    self.env.render()

                # Reward sequence # TODO: Build a reward sequence for the PPO
                reward_sequence = add_to_sequence(reward_sequence, reward, self.device)
                tensor_reward_sequence = torch.stack(list(reward_sequence))
                tensor_reward_sequence = padding_sequence(
                    tensor_reward_sequence, self.sequence_length, self.device
                )

                observations.append(tensor_sequence)
                actions.append(action)
                log_probs.append(log_prob)
                episode_rewards.append(tensor_reward_sequence)


                if done:
                    break

            lens.append(ep_timestep + 1)
            rewards.append(torch.stack(episode_rewards))

        print("Timesteps: ", timesteps)
        # Reshape the data
        
        obs = torch.stack(observations).to(self.device)
        actions = torch.stack(actions).to(self.device)
        log_probs = torch.stack(log_probs).to(self.device)
        rtgs = self.compute_rtgs(rewards)

        return obs, actions, log_probs, rtgs, lens, frames
        

    def compute_rtgs(self, rewards):
        rtgs = []  # To store RTG tensors for each episode

        for ep_rewards in rewards:
            discounted_reward = 0
            for reward in reversed(ep_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                rtgs.insert(0, discounted_reward)
        #print("RTGS: ", rtgs, rtgs[0].shape)
        rtgs = torch.stack(rtgs).to(self.device)
        #rtgs = torch.unsqueeze(rtgs, 2)

        return rtgs

    def get_action(self, obs):
        
        obs = obs.unsqueeze(0)

        mean, std = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

        action = dist.sample()
        log_prob = dist.log_prob(action)
        scaled_action = torch.clamp(
            self.action_low + (self.action_high - self.action_low) * ((action + 1) / 2),  # Transform from [0, 1] to [low, high]
            self.action_low, self.action_high
        )

        return scaled_action, log_prob

    def evaluate(self, obs, actions):
        V = self.critic_network(obs)

        mean, std = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

        log_prob = dist.log_prob(actions)

        return V, log_prob

    def __init_hyperparameters(self, config):
        self.clip_grad_normalization = config["clip_grad_normalization"]
        self.clip = config["clip"]
        self.learning_rate = config["learning_rate"]
        self.n_updates_per_iteration = config["n_updates_per_iteration"]
        self.gamma = config["gamma"]
        self.batch_size = config["batch_size"]
        self.mini_batch_size = config["mini_batch_size"]
        self.update_frequency = config["target_model_update"]
        # self.max_episodes = config["total_episodes"]
        self.max_steps = config["max_steps_per_episode"]
        self.render = config["render"]
        self.render_mode = config["render_mode"]

    def generate_minibatches(self, obs, actions, log_probs, rtgs):
        minibatches = []
        for _ in range(self.n_updates_per_iteration):
            idxs = np.random.randint(0, len(obs), self.mini_batch_size)
            minibatches.append((obs[idxs], actions[idxs], log_probs[idxs], rtgs[idxs]))

        return minibatches


    def load_model(self, policy_path, critic_path):
        self.policy_network.load_state_dict(torch.load(policy_path))
        self.critic_network.load_state_dict(torch.load(critic_path))
