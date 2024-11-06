import torch
import torch.nn as nn
from transformer_decoder import TransformerDQN


class ObservationCNN(nn.Module):
    def __init__(self, input_dim) -> None:
        super(ObservationCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64 * 5 * 3, input_dim)

    def forward(self, x):
        x = self.conv_layers()
        x = x.view(x.size(0), -1)
        x = self.fc


class CNNTransformer(nn.Module):

    def __init__(
        self,
        state_dim = 684,
        input_dim = 128,
        output_dim = 3,
        block_size = 45,
        n_embd = 128,
        n_head = 8,
        n_layer = 3,
        dropout = 0.3,
        device,
    ):
        self.embedding = ObservationCNN(input_dim)
        decoder_layers = TransformerDQN(
            input_dim=input_dim,
            output_dim=output_dim,
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=device
        )

    def forward(self, x): # x should be (observation_sequence, q_val_sequence, last_action_sequence)
        observation_sequence, q_val_sequence, last_action_sequence = x


