import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from .dqn.dqn_agent import DQN_Agent
from scipy.special import softmax

from env.sunburstmaze_discrete import SunburstMazeDiscrete
from utils.state_preprocess import state_preprocess

device = torch.device("cpu")
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Was faster with cpu??? Loading between cpu and mps is slow maybe

RL_load_path = "./model/transformers/seq_len_45/visionary-hill-816"


def grad_sam(
    attention_weights, gradients, block=0, episode=0, step=0, rgb_array=None, plot=False
):
    count = 0
    grad_sams = []
    for i in range(len(attention_weights)):
        # Apply ReLU to the gradients
        grad = torch.relu(gradients[i])
        att_w = attention_weights[i].squeeze(0)
        grad = grad.squeeze(0)
        # print("Att_w:", att_w.shape, type(att_w))
        # print(grad.shape, type(grad))

        # Multiply the gradients with the attention weights
        grad_sam = att_w @ grad
        grad_sams.append(grad_sam)
        # print(f'grad_sam_{count}:', grad_sam.shape, grad_sam)

    if plot:
        plot(
            grad_sams,
            block,
            att_heads=len(attention_weights),
            episode=episode,
            step=step,
            rgb_array=rgb_array,
        )

    return grad_sams


def plot(grad_sams, block, att_heads, episode=0, step=0, rgb_array=None):
    fig, axes = plt.subplots(3, math.ceil(att_heads / 3), figsize=(30, 10))
    fig.suptitle(f"Grad-SAM for block {block}")
    for idx, grad_sam in enumerate(grad_sams):
        # Show the grad-sam as a heatmap
        sns.heatmap(grad_sam, ax=axes[math.floor(idx / 3), idx % 3])
        # sns.heatmap(att_w, ax=axes[math.floor(i / 3), i % 3])
        axes[math.floor(idx / 3), idx % 3].set_title(f"Attention head {idx}")
    axes[2, 2].imshow(rgb_array)
    # Nplt.show()
    plt.savefig(
        f"./attention_plot/grad_sam_block_{block}_episode_{episode}_step_{step}.png"
    )
    plt.close()


class ExplainNetwork:
    def __init__(self, RL_load_path=""):
        self.RL_load_path = RL_load_path

    def generate_q_values(self, env: SunburstMazeDiscrete, model):

        orientation_range = [0, 1, 2, 3]  # north, east, south, west

        x_range = env.initial_map.shape[1]
        y_range = env.initial_map.shape[0]

        q_val_list_to_position = []

        for x in range(x_range):
            for y in range(y_range):
                if env.env_map[y, x] == 1:
                    continue
                env.position = (y, x)
                q_list = []
                q_value_list = np.zeros(4)
                for orientation in orientation_range:
                    env.orientation = orientation
                    state = env._get_observation()
                    state = state_preprocess(state, device=device)
                    tensor_state = torch.stack(list(state)).to(device)
                    tensor_state = tensor_state.unsqueeze(0)
                    
                    tensor_state = tensor_state.unsqueeze(0)
                    q_values = model(tensor_state)[0].detach().numpy()
                    q_list.append(q_values[0,0])
                for orientation in orientation_range:
                    forward = q_list[orientation][0]
                    left = q_list[(orientation + 1) % 4][1] / 2
                    right = q_list[orientation - 1][2] / 2
                    q_value_sum = forward + left + right
                    q_value_list[orientation] = q_value_sum

                q_value_list = softmax(q_value_list)
                dicti = {(y, x): q_value_list}
                q_val_list_to_position.append(dicti)

        return q_val_list_to_position

class ExplainNetworkFF:
    def __init__(self, RL_load_path=""):
        self.RL_load_path = RL_load_path

    def generate_q_values(self, env: SunburstMazeDiscrete, model):

        orientation_range = [0, 1, 2, 3]  # north, east, south, west

        x_range = env.initial_map.shape[1]
        y_range = env.initial_map.shape[0]

        q_val_list_to_position = []

        for x in range(x_range):
            for y in range(y_range):
                if env.env_map[y, x] == 1:
                    continue
                env.position = (y, x)
                q_list = []
                q_value_list = np.zeros(4)
                for orientation in orientation_range:
                    env.orientation = orientation
                    state = env._get_observation()
                    tensor_state = state_preprocess(state, device=device)

                    q_values = model(tensor_state)
                    q_list.append(q_values)
                for orientation in orientation_range:
                    forward = q_list[orientation][0]
                    left = q_list[(orientation + 1) % 4][1] / 2
                    right = q_list[orientation - 1][2] / 2
                    q_value_sum = forward + left + right
                    q_value_list[orientation] = q_value_sum

                q_value_list = softmax(q_value_list)
                dicti = {(y, x): q_value_list}
                q_val_list_to_position.append(dicti)

        return q_val_list_to_position


    # Grad-SAM

    # concepts

    """
    - Toril sees goal
    - Toril sees wall
    - Toril sees open space
    - Toril is spinning
    - Toril moves away from goal
    - Toril moves towards goal

    """
