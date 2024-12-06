import ast
import copy
import math
import os
import random as rd
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, random_split

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent.transformer_decoder_decoupled import TransformerDQN
from utils.calculate_fov import calculate_fov_matrix_size
from utils.custom_dataset import CAV_dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("mps") if torch.cuda.is_available() else torch.device("cpu")

activations = {}

"""episode_numbers = [
    "10",
    "30",
    "50",
    "80",
    "100",
    "500",
    "1000",
    "1500",
    "2000",
    "2500",
    "3000",
    "3500",
    "4000",
    "4500",
    "5000",
    "5200",
]"""

episode_numbers = [
    "10",
    "30",
    "1500",
    "2000",
    "5200",
]


def get_activation(name):
    def hook(module, input, output):
        activations[name] = {
            "input": input[0],
            "output": output,
        }

    return hook


def get_activations(
    model: TransformerDQN, input, block_name: str = None, embedding: bool = False
):
    """
    Get the activations of the model for the training data.
    """
    activations.clear()

    if embedding:
        q_value = get_embedding_activations(model, input)
        return q_value

    block_int = int(block_name.split("_")[-1])

    block = model.blocks[block_int]
    block.register_forward_hook(get_activation(block_name))
    q_value = model(input)

    return q_value


def get_embedding_activations(model: TransformerDQN, input):

    embedding = model.token_embedding
    embedding.register_forward_hook(get_activation("embedding"))
    q_value = model(input)
    return q_value


def create_activation_dataset(
    dataset_path: str,
    model_path: str,
    block: int = 0,
    embedding: bool = False,
    requires_grad: bool = False,
):

    fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 8,
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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(model)
    episode_number = model_path.split("_")[-1].split(".")[0]
    # Activation file name
    if embedding:
        activation_file = f"dataset/activations/{dataset_path.split('/')[-1].split('.')[0]}_embedding_episode_{episode_number}.pt"
    else:
        activation_file = f"dataset/activations/{dataset_path.split('/')[-1].split('.')[0]}_activations_{block}_episode_{episode_number}.pt"

    # Read the dataset
    dataset = pd.read_csv(dataset_path)
    # print(dataset.head(10))
    sequences = [
        [torch.tensor(ast.literal_eval(state)) for state in states]
        for _, states in dataset.iterrows()
    ]

    state_tensors = torch.stack(
        [
            torch.stack(sequence).float().to(device).requires_grad_(requires_grad)
            for sequence in sequences
        ]
    ).requires_grad_(requires_grad)
    print(state_tensors.shape)
    if embedding:
        q_val = get_activations(model, state_tensors, embedding=True)
        activation = activations["embedding"]["output"]
        # print("Hello", activation.shape)
        # activation_list.append(activation.clone().requires_grad_(requires_grad))
        # q_val_list.append(q_val.clone().detach())

    else:
        block_name = f"block_{block}"
        q_val = get_activations(model, state_tensors, block_name)
        activation = activations[block_name]["output"][0]
        print(activation.shape)

        # print("Hello", activation.shape)
        # activation_list.append(activation.requires_grad_(requires_grad))
        # q_val_list.append(q_val.clone())

    # torch.save(activation_list, activation_file)
    # q_val_list_file = f"dataset/q_val_{dataset_path.split('/')[-1].split('.')[0]}_episode_{episode_number}.pt"
    # if not os.path.exists(q_val_list_file):
    #    pass
    #    # torch.save(q_val_list, q_val_list_file)

    assert isinstance(activation, torch.Tensor), "Activation must be a tensor"

    return activation, q_val


class CAV:

    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

    def __init__(self):
        self.model = None
        self.cav_coef = None
        self.cav_list = []
        self.tcav_list = []

    def read_dataset(self, concept, model_path, block, embedding, sensitivity):

        positive, q_values_positive = create_activation_dataset(
            f"./dataset/{concept}_positive_train.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
        )
        assert isinstance(positive, torch.Tensor), "Positive must be a tensor"

        negative, q_values_negative = create_activation_dataset(
            f"./dataset/{concept}_negative_train.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
        )
        assert isinstance(negative, torch.Tensor), "Negative must be a tensor"

        positive_test, q_values_positive_test = create_activation_dataset(
            f"./dataset/{concept}_positive_test.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
        )
        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

        negative_test, q_values_negative_test = create_activation_dataset(
            f"./dataset/{concept}_negative_test.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
        )
        assert isinstance(negative, torch.Tensor), "Negative_test must be a tensor"

        return (
            positive,
            negative,
            positive_test,
            negative_test,
            q_values_positive,
            q_values_negative,
            q_values_positive_test,
            q_values_negative_test,
        )

    def cav_model(self, positive_train, negative_train, positive_test, negative_test):

        positive_train_labels = np.ones(len(positive_train))
        negative_train_labels = np.zeros(len(negative_train))

        positive_test_labels = np.ones(len(positive_test))
        negative_test_labels = np.zeros(len(negative_test))
        # Split the dataset

        positive_train_clone = positive_train.clone()
        negative_train_clone = negative_train.clone()
        positive_test_clone = positive_test.clone()
        negative_test_clone = negative_test.clone()

        # TODO: Refactor this shit
        positive_train_np = [
            positive_train_clone[i].detach().flatten().numpy()
            for i in range(len(positive_train))
        ]
        negative_train_np = [
            negative_train_clone[i].detach().flatten().numpy()
            for i in range(len(negative_train))
        ]
        train_data = np.concatenate((positive_train_np, negative_train_np), axis=0)
        train_labels = np.concatenate(
            (positive_train_labels, negative_train_labels), axis=0
        )

        positive_test_np = [
            positive_test_clone[i].detach().numpy().flatten()
            for i in range(len(positive_test))
        ]
        negative_test_np = [
            negative_test_clone[i].detach().numpy().flatten()
            for i in range(len(negative_test))
        ]
        test_data = np.concatenate((positive_test_np, negative_test_np), axis=0)
        test_labels = np.concatenate(
            (positive_test_labels, negative_test_labels), axis=0
        )
        # Train the model
        self.model = LogisticRegression()

        self.model.fit(train_data, train_labels)

        # Test the model
        score = self.model.score(test_data, test_labels)

        cav_coef = self.model.coef_
        return score, cav_coef

    def calculate_single_cav(
        self,
        block: int,
        episode_number: str,
        positive_train,
        negative_train,
        positive_test,
        negative_test,
    ):

        # positive_file = f"dataset/activations/positive_{concept}_activations_{block}.pt"
        # negative_file = f"dataset/activations/negative_{concept}_activations_{block}.pt"
        accuracy, cav = self.cav_model(
            positive_train, negative_train, positive_test, negative_test
        )
        self.cav_list.append((block, episode_number, accuracy))
        return cav

    def calculate_cav(self, concept: str, model_dir: str, sensitivity: bool = False):

        model_list = os.listdir(model_dir)

        for model in model_list:
            model_path = os.path.join(model_dir, model)
            episode_number = model.split("_")[-1].split(".")[0]

            if episode_number not in episode_numbers:
                continue

            (
                positive,
                negative,
                positive_test,
                negative_test,
                _,
                _,
                q_values_positive_test,
                _,
            ) = self.read_dataset(concept, model_path, 0, False, sensitivity)

            cav = self.calculate_single_cav(
                0,
                episode_number,
                positive,
                negative,
                positive_test,
                negative_test,
            )
            # Calculate the tcav
            if sensitivity:
                self.calculate_tcav(
                    cav, positive, q_values_positive_test, 0, episode_number
                )

            for block in range(1, 4):
                # load dataset
                (
                    positive,
                    negative,
                    positive_test,
                    negative_test,
                    _,
                    _,
                    q_values_positive_test,
                    _,
                ) = self.read_dataset(concept, model_path, 0, False, sensitivity)

                print("Block: ", block, model_path, episode_number)
                cav = self.calculate_single_cav(
                    block, episode_number, positive, negative, positive, negative
                )

                if sensitivity:
                    self.calculate_tcav(
                        cav,
                        positive_test,
                        q_values_positive_test,
                        block,
                        episode_number,
                    )
        # Save the CAV list
        torch.save(self.cav_list, f"./cav_list_{concept}.pt")
        torch.save(self.tcav_list, f"./tcav_list_{concept}.pt")

    def calculate_sensitivity(
        self, activations: torch.Tensor, network_output: torch.Tensor, cav: np.ndarray
    ):

        assert isinstance(
            activations, torch.Tensor
        ), f"Activations must be a tensor{type(activations)}"
        # assert all(type(a) for a in activations) == torch.tensor, f"Activations must be a tensor {type(activations)}"
        assert isinstance(
            network_output, torch.Tensor
        ), "Network output must be a tensor"
        # assert all(type(n) for n in network_output) == torch.Tensor, "Network output must be a tensor"

        outputs = torch.autograd.grad(
            outputs=network_output,
            inputs=activations,
            grad_outputs=torch.ones_like(network_output),
            retain_graph=True,
        )[0]

        grad_flattened = outputs.view(outputs.size(0), -1).detach().cpu().numpy()

        return np.dot(grad_flattened, cav.T)

    def calculate_tcav(
        self,
        cav: np.ndarray,
        positive_test_data,
        q_values_positive: torch.Tensor,
        block,
        episode_number,
    ):

        # Get the activations
        activations = positive_test_data
        network_output = q_values_positive
        sensitivity = self.calculate_sensitivity(activations, network_output, cav)
        tcav = (sensitivity > 0).mean()

        self.tcav_list.append((block, episode_number, tcav))

    def load_cav(self, concept: str):
        cav_list = torch.load(f"./cav_list_{concept}.pt")
        return cav_list

    def plot_cav(self, concept: str, tcav: bool = False):

        if len(self.cav_list) == 0:
            self.cav_list = self.load_cav(concept)
        print(self.cav_list[0])

        self.plot(concept, self.cav_list)

    def plot_tcav(self, concept: str):
        if len(self.tcav_list) == 0:
            self.tcav_list = self.load_cav(concept)

        self.plot(concept, self.tcav_list)

    def plot(self, concept: str, cav_list: list):

        # Extract data
        blocks = np.array([t[0] for t in cav_list])
        episode_numbers = np.array([int(t[1]) for t in cav_list])
        accuracies = np.array([t[2] for t in cav_list])

        # Create a grid for interpolation
        block_lin = np.linspace(blocks.min(), blocks.max(), 4)
        episode_lin = np.linspace(episode_numbers.min(), episode_numbers.max(), 55)
        block_grid, episode_grid = np.meshgrid(block_lin, episode_lin)

        # Interpolate accuracy values onto the grid
        accuracy_grid = griddata(
            (blocks, episode_numbers),
            accuracies,
            (block_grid, episode_grid),
            method="cubic",
        )

        norm = Normalize(vmin=0.5, vmax=1)
        colors = cm.viridis

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        surf = ax.plot_surface(
            block_grid,
            episode_grid,
            accuracy_grid,
            cmap=colors,
            norm=norm,
            edgecolor="k",
        )

        # Labels
        ax.set_xlabel("Block")
        ax.set_ylabel("Episode Number")
        ax.set_zlabel("Accuracy")

        # range
        ax.set_xlim(blocks.min(), blocks.max())
        ax.set_ylim(episode_numbers.min(), episode_numbers.max())
        ax.set_zlim(0, 1)

        ax.xaxis.set_major_locator(MultipleLocator(1))

        # Colorbar for accuracy
        fig.colorbar(surf, ax=ax, label="Accuracy")

        plt.savefig(f"./cav_{concept}.png")
        plt.show()


class Analysis:
    def __init__(self, average: int = 5):
        self.average = average
        self.total_tcav = {}

    def add_total_tcav_scores(self, tcav_list: list):

        for block, episode, tcav in tcav_list:
            if block not in self.total_tcav:
                self.total_tcav[block] = {}
            if episode not in self.total_tcav[block]:
                self.total_tcav[block][episode] = 0

            self.total_tcav[block][episode] += tcav

    def calculate_average_tcav(self):
        for block in self.total_tcav:
            for episode in self.total_tcav[block]:
                self.total_tcav[block][episode] /= self.average

    def get_tcav(self):
        return self.total_tcav


def main():
    cav = CAV()
    model_load_path = "../../agent/model/transformers/model_vivid-firebrand-872"
    concept = "random"
    # positive_file = "dataset/positive_wall_activations.pt"
    # negative_file = "dataset/negative_wall_activations.pt"

    average = 5
    total_tcav = {}
    analysis = Analysis(average)
    for _ in range(average):
        cav.calculate_cav(concept, model_load_path)
        tcav = cav.tcav_list
        analysis.add_total_tcav_scores(tcav)

    analysis.calculate_average_tcav()
    total_tcav = analysis.get_tcav()
    cav.tcav_list = total_tcav
    # cav.plot_cav(concept)

    # cav.calculate_cav(concept, model_load_path, sensitivity=True)
    # cav.cav_list = torch.load(f"./cav_list_{concept}.pt")
    # cav.plot_cav(concept)
    cav.plot_tcav(concept)

    # cav.calculate_random_cav("goal", model_load_path)
    # cav.plot_cav("random")


if __name__ == "__main__":
    main()
