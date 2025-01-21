import os
from collections import deque

import numpy as np
import torch
import wandb
from torch import nn
torch.autograd.set_detect_anomaly(True)
from transformer_decoder import Transformer

from env import SunburstMazeContinuous
from utils.sequence_preprocessing import (
    add_to_sequence,
    padding_sequence,
    padding_sequence_int,
)
from utils.state_preprocess import state_preprocess


class PPO_agent:

    def __init__(self, env: SunburstMazeContinuous, device, config):
        wandb.login()

        self.run = wandb.init(project="sunburst-maze-continuous", config=self)

        gif_path = f"./gifs/{self.run.name}"

        # Create the nessessary directories
        if not os.path.exists(gif_path):
            os.makedirs(gif_path)

        self.frames = []

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
        self.policy_network = Transformer(
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
        print("Learning")

        timestep_counter = 0
        iteration_counter = 0

        while timestep_counter < total_timesteps:

            self.frames = []

            obs, actions, log_probs, rtgs, lens = self.rollout(iteration_counter)
            timestep_counter += sum(lens)
            iteration_counter += 1
            #print("Obs: ", obs, obs.shape)
            # Calculate the advantages
            value, _ = self.evaluate(obs, actions)
            rtgs = rtgs.unsqueeze(2)

            advantages = rtgs - value.detach() # TODO: Feil i values, log_prob eller rtgs????

            # Normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
            steps_done = 0
            for _ in range(self.n_updates_per_iteration):
                value, current_log_prob = self.evaluate(obs, actions)

                #print(current_log_prob)
                ratio = torch.exp(current_log_prob - log_probs)
                ratio = ratio.unsqueeze(2)
 
                surrogate_loss1 = ratio * advantages

                #print("Surrogate loss1: ", surrogate_loss1, surrogate_loss1.shape)
                surrogate_loss2 = (
                    torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
                )
                # Increase the size of rtgs to be 300 x 15
  
                policy_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

                critic_loss = nn.MSELoss()(value, rtgs)

                print("Policy loss step")
                self.policy_network.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_optimizer.step()
                

                self.critic_network.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                print("After")
                steps_done += 1

            gif = None
            if self.frames:
                if os.path.exists("./gifs") is False:
                    os.makedirs("./gifs")

                gif = self.env.create_gif(
                    gif_path=f"./gifs/{iteration_counter}.gif", frames=self.frames
                )
                self.frames.clear()

            wandb.log(
                {
                    "Episode": lens,
                    "Reward per episode": rtgs.mean().item(),
                    "Policy loss": policy_loss.item(),
                    "Critic loss": critic_loss.item(),
                    "Steps done": steps_done,
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

    def rollout(self, iteration_counter):
        observations = []
        actions = []
        log_probs = []
        rewards = []
        rtgs = []
        lens = []

        episode_rewards = []

        timesteps = 0

        state_sequence = deque(maxlen=self.sequence_length)
        action_sequence = deque(maxlen=self.sequence_length)
        log_prob_sequence = deque(maxlen=self.sequence_length)

        while timesteps < self.batch_size:
            episode_rewards = []
            reward_sequence = deque(maxlen=self.sequence_length)
            state, _ = self.env.reset()
            done = False
            

            for ep_timestep in range(self.max_steps):
                timesteps += 1

                state_sequence = add_to_sequence(state_sequence, state, self.device)
                tensor_sequence = torch.stack(list(state_sequence))
                tensor_sequence = padding_sequence(
                    tensor_sequence, self.sequence_length, self.device
                )
                action, log_prob = self.get_action(tensor_sequence)
                last_action = action[-1].detach().numpy()
                #last_log_prob = log_prob[-1,-1]
                
                state, reward, done, turnicated, _ = self.env.step(last_action)

                if self.render_mode == "rgb_array":
                    if iteration_counter % 100 == 0:
                        frame = self.env._render_frame()
                        if type(frame) == np.ndarray:
                            self.frames.append(frame)
                if self.render_mode == "human":
                    self.env.render()

                action_sequence = add_to_sequence(action_sequence, last_action, self.device)
                tensor_action_sequence = torch.stack(list(action_sequence))
                tensor_action_sequence = padding_sequence_int(
                    tensor_action_sequence, self.sequence_length, self.device
                )

                #log_prob_sequence = add_to_sequence(
                #    log_prob_sequence, last_log_prob, self.device
                #)
                #tensor_log_prob_sequence = torch.stack(log_prob)
                #tensor_log_prob_sequence = padding_sequence(
                 #   tensor_log_prob_sequence, self.sequence_length, self.device
                #)

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

        return obs, actions, log_probs, rtgs, lens
        

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
        """for ep_rewards in rewards:
            seq_len = len(ep_rewards)
            rtg = torch.zeros((len(ep_rewards),seq_len, 1), device=self.device)
            discounted_reward = 0

            # Compute RTG in reverse
            for t in reversed(range(seq_len)):
                print(t)
                print("Ep_rewards: ", ep_rewards[t])
                discounted_reward = ep_rewards[t] + self.gamma * discounted_reward
                rtg[t] = discounted_reward

            rtgs.append(rtg)
        print("RTGS: ", rtgs)
        return torch.stack(rtgs)"""

    def get_action(self, obs):
        obs = obs.unsqueeze(0)

        mean = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.squeeze(0), log_prob.squeeze(0).detach()

    def evaluate(self, obs, actions):
        V = self.critic_network(obs)

        mean = self.policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)

        log_prob = dist.log_prob(actions)

        return V, log_prob

    def __init_hyperparameters(self, config):
        self.clip_grad_normalization = config["clip_grad_normalization"]
        self.clip = config["clip"]
        self.learning_rate = config["learning_rate"]
        self.n_updates_per_iteration = config["n_updates_per_iteration"]
        self.gamma = config["discount_factor"]
        self.batch_size = config["batch_size"]
        self.update_frequency = config["target_model_update"]
        # self.max_episodes = config["total_episodes"]
        self.max_steps = config["max_steps_per_episode"]
        self.render_mode = config["render"]
