import os
import sys

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
import numpy as np
import torch
from replay_memory import ReplayMemory
from torch import nn, optim

from neural_network_ff_torch import DQN_Network


class DQN_Agent:
    """
    DQN Agent Class. This class defines some key elements of the DQN algorithm,
    such as the learning method, hard update, and action selection based on the
    Q-value of actions or the epsilon-greedy policy.
    """

    def __init__(
        self,
        env,
        epsilon,
        epsilon_min,
        epsilon_decay,
        clip_grad_normalization,
        learning_rate,
        discount,
        memory_capacity,
        device,
        seed,
    ):

        # To save the history of network loss
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount
        self.device = device

        self.action_space = env.action_space
        self.action_space.seed(seed)

        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(memory_capacity, self.device)

        # Initiate the network models
        self.model = DQN_Network(
            num_actions=self.action_space.n, input_dim=self.observation_space.n + 4, device=self.device
        ).to(self.device)

        self.target_model = (
            DQN_Network(
                num_actions=self.action_space.n, input_dim=self.observation_space.n + 4, device=self.device
            )
            .to(self.device)
            .eval()
        )

        # Copy the weights of the model
        self.target_model.load_state_dict(self.model.state_dict())

        self.clip_grad_normalization = clip_grad_normalization  # For clipping exploding gradients caused by high reward value
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)


    def select_action(self, state):
        """
        Selects an action using epsilon-greedy strategy OR based on the Q-values.

        Parameters:
            state (torch.Tensor): Input tensor representing the state.

        Returns:
            action (int): The selected action.
        """

        # Exploration: epsilon-greedy
        if np.random.random() < self.epsilon:
            return self.action_space.sample(), 0

        # Exploitation: the action is selected based on the Q-values.
        with torch.no_grad():
            Q_values = self.model(state)
            action = torch.argmax(Q_values).item()
            q_variance = torch.var(Q_values).item()
            return action, q_variance

    def learn(self, batch_size, done):

        states, actions, next_states, rewards, dones = self.replay_memory.sample(
            batch_size
        )

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        current_q_values = self.model(states)

        current_q_values = current_q_values.gather(dim=1, index=actions)

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            future_q_values = self.target_model(next_states).max(dim=1, keepdim=True)[
                0
            ]  # not argmax (cause we want the maxmimum q-value, not the action that maximize it)

        future_q_values[dones] = 0

        target = rewards + (self.discount * future_q_values)

        loss = self.critertion(current_q_values, target)

        # Update the running loss and learned counts for logging and plotting
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = (
                self.running_loss / self.learned_counts
            )  # The average loss for the episode
            self.loss_history.append(episode_loss)

            self.running_loss = 0
            self.learned_counts = 0

        self.optimizer.zero_grad()  # Zero the gradients
        loss.backward()

        # Clip the gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.clip_grad_normalization
        )

        self.optimizer.step()  # Update the parameters of the main network using the optimizer

    def hard_update(self):
        """
        Perform a hard update of the target model's parameters.
        This method copies the parameters from the main model to the target model.
        It ensures that the target model has the same weights as the main model.
        """

        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        """
        Update the value of epsilon for epsilon-greedy exploration.

        This method decreases epsilon over time according to a decay factor, ensuring
        that the agent becomes less exploratory and more exploitative as training progresses.
        """

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """
        Save the parameters of the main network to a file with .pth extention.

        """
        torch.save(self.model.state_dict(), path)

    


