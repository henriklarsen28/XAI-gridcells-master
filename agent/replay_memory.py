from collections import deque

import numpy as np
import torch


class ReplayMemory:
    def __init__(self, capacity, device):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """
        self.capacity = capacity

        self.states = deque(maxlen=capacity)
        self.q_value_sequence = deque(maxlen=capacity)
        self.prev_q_value_sequence = deque(maxlen=capacity)
        self.prev_action_sequence = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)

        self.device = device

    def store(
        self,
        state,
        prev_q_value,
        prev_action_sequence,
        action,
        q_value,
        next_state,
        reward,
        done,
    ):
        """
        Append (store) the transitions to their respective deques
        """

        self.states.append(state)
        self.q_value_sequence.append(q_value)
        self.prev_q_value_sequence.append(prev_q_value)
        self.prev_action_sequence.append(prev_action_sequence)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """

        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack(
            [
                torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device)
                for i in indices
            ]
        ).to(self.device)

        prev_q_values = torch.stack(
            [
                torch.as_tensor(
                    self.prev_q_value_sequence[i], dtype=torch.float32, device=self.device
                )
                for i in indices
            ]
        ).to(self.device)

        prev_actions = torch.stack(
            [
                torch.as_tensor(
                    self.prev_action_sequence[i], dtype=torch.int64, device=self.device
                )
                for i in indices
            ]
        ).to(self.device)

        actions = torch.stack(
            [
                torch.as_tensor(self.actions[i], dtype=torch.int64, device=self.device)
                for i in indices
            ]
        ).to(self.device)

        q_values = torch.stack(
            [
                torch.as_tensor(
                    self.q_value_sequence[i], dtype=torch.float32, device=self.device
                )
                for i in indices
            ]
        ).to(self.device)

        next_states = torch.stack(
            [
                torch.as_tensor(
                    self.next_states[i], dtype=torch.float32, device=self.device
                )
                for i in indices
            ]
        ).to(self.device)

        rewards = torch.stack(
            [
                torch.as_tensor(
                    self.rewards[i], dtype=torch.float32, device=self.device
                )
                for i in indices
            ]
        ).to(self.device)
        dones = torch.stack(
            [
                torch.as_tensor(self.dones[i], dtype=torch.bool, device=self.device)
                for i in indices
            ]
        ).to(self.device)

        return states, prev_q_values, prev_actions, actions, q_values, next_states, rewards, dones

    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque
        represents the length of the entire memory.
        """

        return len(self.dones)
