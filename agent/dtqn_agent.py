import numpy as np
import torch
from replay_memory import ReplayMemory
from torch import nn, optim

from agent.neural_network_ff_torch import DQN_Network
from agent.transformer_decoder import TransformerDQN


def get_attention_gradients(module, grad_input, grad_output):
    global attention_gradients
    attention_gradients = grad_output[0]



class DTQN_Agent:
    """
    DTQN Agent Class. This class defines some key elements of the DQN algorithm,
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
        transformer_param,
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
        n_head = transformer_param[
            "n_head"
        ]  # Number of attention heads (in multi-head attention)
        n_layer = transformer_param["n_layer"]  # Number of decoder layers
        dropout = transformer_param["dropout"]  # Dropout probability

        state_dim = self.observation_space.n + 4  # Replace value

        sequence_length = transformer_param["sequence_length"]  # Replace value

        # Initiate the network models
        self.model = TransformerDQN(
            state_dim,
            self.action_space.n,
            sequence_length,
            n_embd,
            n_head,
            n_layer,
            dropout,
            self.device
        )
        self.model = self.model.to(self.device)

        self.target_model = TransformerDQN(
            state_dim,
            self.action_space.n,
            sequence_length,
            n_embd,
            n_head,
            n_layer,
            dropout,
            self.device,
        )
        self.target_model = self.target_model.to(self.device).eval()

        # Copy the weights of the model
        self.target_model.load_state_dict(self.model.state_dict())

        self.clip_grad_normalization = clip_grad_normalization  # For clipping exploding gradients caused by high reward value
        self.critertion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def calculate_gradients(self, state, next_state, reward, block=0):
        state = state.unsqueeze(0)
        next_state = next_state.unsqueeze(0)
        gradient_list = []
        for i in range(8):
            attention_layer = self.model.blocks[block].sa.heads[i]
            hook = attention_layer.register_backward_hook(get_attention_gradients)
            Q_values, att_weights_list = self.model(state)
            action = torch.argmax(Q_values[:, -1, :]).item()
            
            future_q, _ = self.model(next_state)

            target = reward + self.discount * torch.max(future_q[:, -1, :])
            loss = self.critertion(Q_values, target)
            loss.backward(retain_graph=True)
            gradient_list.append(attention_gradients)
        #print("Gradient_list: ", gradient_list)
        return gradient_list

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
            # state = torch.tensor(state).to(self.device)
            # state = torch.stack(list(state))
            state = state.unsqueeze(0)
            # print(state.shape)
            Q_values, att_weights_list = self.model(state)
            action = torch.argmax(Q_values[:, -1, :]).item()
            # print("Q_vals: ", Q_values)
            # print("Selected q_val: ", Q_values[:,-1,:], "Action: ", action)
            # print("att_weights_list: ", att_weights_list)
            return action, att_weights_list

    def learn(self, batch_size, done):

        states, actions, next_states, rewards, dones = self.replay_memory.sample(
            batch_size
        )

        # Sends in the sequence of states, actions, next_states, rewards, and dones
        # For training we will use every q_value in the sequence, but only the last q_value will be used for selecting an action
        actions = actions.unsqueeze(2)
        rewards = rewards.unsqueeze(2)
        dones = dones.unsqueeze(2)
        # print(states.shape, actions.shape)
        current_q_values = self.model(states)
        # print(current_q_values[:,0,:])
        # print(current_q_values.shape, actions.shape, rewards.shape, dones.shape)
        current_q_values = current_q_values.gather(2, actions).squeeze()

        # Compute the maximum Q-value for the next states using the target network
        with torch.no_grad():
            future_q_values = self.target_model(next_states).max(dim=2, keepdim=True)[
                0
            ]  # not argmax (cause we want the maxmimum q-value, not the action that maximize it)

        future_q_values[dones] = 0

        targets = rewards.squeeze() + (self.discount * future_q_values.squeeze())
        # print(current_q_values.shape, targets.shape)
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
