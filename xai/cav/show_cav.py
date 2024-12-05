
import torch

cavs = torch.load("./dataset/activations/positive_rotating_activations_1_episode_2500.pt")


print(cavs[0].shape)
