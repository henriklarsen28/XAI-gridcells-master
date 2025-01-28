import torch
from collections import deque


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


def padding_sequence(sequence: torch.tensor, max_length, device):
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
                        last_state, dtype=torch.float32, device=device
                    ).unsqueeze(0),
                ]
            )
    return sequence


def add_to_sequence(sequence: deque, state, device):
    """
    Add the new state to the sequence
    """
    state = torch.tensor(state, dtype=torch.float32, device=device)
    #torch.as_tensor(state, dtype=torch.float32, device=devic)
    sequence.append(state)
    return sequence