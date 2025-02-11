import torch
from collections import deque
import numpy as np


def padding_sequence_int(sequence: torch.tensor, max_length, device):
    """
    Pad the sequence with zeros to the max_length
    """
    last_state = sequence[-1]
    if len(sequence) < max_length:
        for _ in range(max_length - len(sequence)):
            sequence = torch.cat(
                [
                    sequence,
                    torch.as_tensor(
                        last_state, dtype=torch.int64, device=device
                    ).unsqueeze(0),
                ]
            )
    return sequence


def padding_sequence(sequence: torch.Tensor, max_length, device):
    """
    Pad the sequence with zeros to the max_length
    """
    """last_state = sequence[-1] #TODO: Pad with 0 in the beginning
    if len(sequence) < max_length:
        for _ in range(max_length - len(sequence)):
            sequence = torch.cat(
                [
                    sequence,
                    torch.as_tensor(
                        last_state, dtype=torch.float32, device=device
                    ).unsqueeze(0),
                ]
            )
    return sequence"""
    seq_len, feature_dim = sequence.shape

    if seq_len < max_length:
        # Create zero padding tensor of shape [padding_len, feature_dim]
        padding = torch.zeros((max_length - seq_len, feature_dim), dtype=torch.float32, device=device)
        # Concatenate padding at the beginning
        sequence = torch.cat([padding, sequence], dim=0)

    return sequence




def add_to_sequence(sequence: deque, state: torch.Tensor, device):
    """
    Add the new state to the sequence
    """
    if isinstance(state, np.ndarray):
        state = torch.tensor(state, dtype=torch.float32, device=device)
    #torch.as_tensor(state, dtype=torch.float32, device=devic)
    sequence.append(state)
    return sequence