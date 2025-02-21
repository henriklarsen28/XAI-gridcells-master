import torch
from torch import nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        attn_scores = self.attn_weights(x).squeeze(-1)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        return (x * attn_weights).sum(dim=1)  # Weighted sum over time steps
