import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out, wei


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(head_size * n_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_output = []
        att_weights = []

        for h in self.heads:
            out, wei = h(x)
            head_output.append(out)
            att_weights.append(wei)

        out = torch.cat(head_output, dim=-1)
        out = self.dropout(self.proj(out))

        return out, att_weights


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, 2 * n_embd),
            nn.GELU(),
            nn.Linear(2 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        sa_out, att_weights = self.sa(self.ln1(x))
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, att_weights


class TransformerDQN(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        block_size,
        n_embd,
        n_head,
        n_layer,
        dropout,
        device,
    ):
        super(TransformerDQN, self).__init__()
        self.device = device
        self.vision_embedding = nn.Linear(input_dim, n_embd)
        self.token_embedding = nn.Linear(n_embd+1+3, n_embd)  # nn.Embedding (long, int)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.output = nn.Linear(
            n_embd, output_dim
        )  # Optional: add hidden layers after the final decoder layer
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        observation_sequence, q_val_sequence, last_action_sequence = x
        batch_size, sequence_length, state_dim = observation_sequence.shape

        observation_embedding = self.vision_embedding(observation_sequence.to(torch.float32))

        x = torch.cat((observation_embedding, q_val_sequence, last_action_sequence), dim=-1)
        tok_emb = self.token_embedding(x.to(torch.float32))
        pos_emb = self.position_embedding(
            torch.arange(sequence_length, device=self.device)
        )

        x = tok_emb + pos_emb
        x = self.dropout(x)

        att_weights_list = []

        for block in self.blocks:
            x, att_weights = block(x)
            att_weights_list.append(att_weights)

        # x = self.blocks(x)
        x = self.ln_f(x)

        x = self.output(x.to(torch.float32))

        return x, att_weights_list

"""  def forward(self, x): # x should be (observation_sequence, q_val_sequence, last_action_sequence)
        observation_sequence, q_val_sequence, last_action_sequence = x

        #Create embedding with CNN
        
        last_action_sequence = last_action_sequence.unsqueeze(2)
        print("Observation embedding: ", observation_embedding.shape)
        print("Q_val_sequence: ", q_val_sequence.shape)
        print("Last action sequence: ", last_action_sequence)
        input_sequence = torch.cat((observation_embedding, q_val_sequence, last_action_sequence), dim=-1)
        print("Input sequence: ", input_sequence.shape)

        output, att_weight = self.decoder_layers(input_sequence)

        return output, att_weight"""