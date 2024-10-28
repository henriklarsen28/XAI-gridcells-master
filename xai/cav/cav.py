import ast
import math
import os
import sys
from collections import defaultdict

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent import TransformerDQN
from utils.calculate_fov import calculate_fov_matrix_size
from utils.custom_dataset import CAV_dataset

activations = {}


def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()

    return hook


def get_activations(model: TransformerDQN, input, layer: int):
    """
    Get the activations of the model for the training data.
    """
    block = model.blocks[layer]
    block.register_forward_hook(get_activation(f"block_{layer}"))

    activation = model(input)
    


def create_activation_dataset(dataset_path: str):

    model_load_path = "../../agent/model/transformers/model_woven-glade-815/sunburst_maze_map_v0_100.pth"

    fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 10,
        "number_of_rays": 100,
    }
    half_fov = fov_config["fov"] / 2
    matrix_size = calculate_fov_matrix_size(fov_config["ray_length"], half_fov)
    num_states = matrix_size[0] * matrix_size[1]
    num_states += 4

    sequence_length = 45
    n_embd = 128
    n_head = 8
    n_layer = 3
    dropout = 0.3
    state_dim = num_states
    device = torch.device("cpu")

    action_space = 3

    # Initiate the network models
    model = TransformerDQN(
        state_dim,
        action_space,
        sequence_length,
        n_embd,
        n_head,
        n_layer,
        dropout,
        device,
    )
    model = model.to(device)

    # Load the model
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()
    print(model.blocks)

    # Activation file name
    activation_file = "dataset/" + dataset_path.split("/")[-1].split(".")[0] + "_activations.pt"

    # Read the dataset
    dataset = pd.read_csv(dataset_path)
    # print(dataset.head(10))
    activation_list = []
    for _, states in dataset.iterrows():
        # Get the state
        sequence = [torch.tensor(ast.literal_eval(state)) for state in states]
        # print(sequence)

        state_tensor = torch.stack(sequence).float().to(device)
        # state = torch.tensor(state_as_list).float().to(device)
        # print(state_tensor)

        state_tensor = state_tensor.unsqueeze(0)
        # Get the activations of the model
        activation = get_activations(model, state_tensor, 1)
        #print(activations)
        activation_list.append(activations)

    
    torch.save(activation_list, activation_file)

    return activation_file
        
    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

def cav_model(positive_file: str, negative_file: str):

    positive_dataset = torch.load(positive_file)
    negative_dataset = torch.load(negative_file)
    print(len(positive_dataset))
    # Label the datasets
    positive_labels = [1] * len(positive_dataset)
    negative_labels = [0] * len(negative_dataset)
    positive = CAV_dataset(positive_dataset, positive_labels)
    negative = CAV_dataset(negative_dataset, negative_labels)






def main():
    negative_file = create_activation_dataset("./dataset/negative_wall.csv")

    cav_model(negative_file, negative_file)

if __name__ == "__main__":
    main()
