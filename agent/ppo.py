from transformer_decoder import Transformer
from env import SunburstMazeContinuous


class PPO_agent:

    def __init__(self, env: SunburstMazeContinuous, device, **transformer_param):

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        n_embd = transformer_param["n_embd"]  # Embedding dimension
        n_head = transformer_param[
            "n_head"
        ]  # Number of attention heads (in multi-head attention)
        n_layer = transformer_param["n_layer"]  # Number of decoder layers
        dropout = transformer_param["dropout"]  # Dropout probability
        sequence_length = transformer_param["sequence_length"]  # Replace value


        self.policy_network = Transformer(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.shape[0],
            sequence_length=sequence_length,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=device,
        )

        self.value_network = Transformer(
            input_dim=env.observation_space.shape[0],
            output_dim=1,
            sequence_length=sequence_length,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=dropout,
            device=device,
        )
