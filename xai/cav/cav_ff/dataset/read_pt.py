import torch


activations = torch.load("positive_next_wall_activations_0_episode_4000.pt")

print(activations[-1])

activations = torch.load("positive_next_wall_activations_0_episode_10.pt")

print(activations[-1])