import numpy as np
import torch
from replay_memory import ReplayMemory
from torch import nn, optim

from agent.neural_network_ff_torch import DQN_Network
from agent.transformer_decoder import TransformerDQN


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
        transformer_param
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


        n_embd = transformer_param["n_embd"]  # Embedding dimension
        n_head = transformer_param["n_head"]  # Number of attention heads (in multi-head attention)
        n_layer = transformer_param["n_layer"]  # Number of decoder layers
        dropout = transformer_param["dropout"]  # Dropout probability

        state_dim = self.observation_space.n + 4  # Replace value

        sequence_length = transformer_param["sequence_length"] # Replace value

        # Initiate the network models
        self.model = TransformerDQN(
            state_dim, self.action_space.n, sequence_length, n_embd, n_head, n_layer, dropout, self.device
            )
        self.model = self.model.to(self.device)


        self.target_model = TransformerDQN(
            state_dim, self.action_space.n, sequence_length, n_embd, n_head, n_layer, dropout, self.device
            )
        self.target_model = self.target_model.to(self.device).eval()

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
            return self.action_space.sample()

        # Exploitation: the action is selected based on the Q-values.
        with torch.no_grad():
            #state = torch.tensor(state).to(self.device)
            #state = torch.stack(list(state))
            state = state.unsqueeze(0)
            #print(state.shape)
            Q_values = self.model(state)
            action = torch.argmax(Q_values[0,0,:]).item()
            return action

    def learn(self, batch_size, done):

        states, actions, next_states, rewards, dones = self.replay_memory.sample(
            batch_size
        )

        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        #print(states.shape, actions.shape)
        current_q_values = self.model(states)
        #print(current_q_values[:,0,:])
        current_q_values = current_q_values[:,:,0].gather(dim=1, index=actions)

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            future_q_values = self.target_model(next_states).max(dim=1, keepdim=True)[
                0
            ]  # not argmax (cause we want the maxmimum q-value, not the action that maximize it)

        future_q_values[dones] = 0

        targets = rewards + (self.discount * future_q_values)

        loss = self.critertion(current_q_values, targets)

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

    


