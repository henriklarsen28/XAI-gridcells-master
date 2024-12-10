import ast
import math
import os
import random as rd
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

sys.path.append(project_root)

from agent.neural_network_ff_torch import DQN_Network
from utils.calculate_fov import calculate_fov_matrix_size
from utils.custom_dataset import CAV_dataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

activations = {}
hook_handles = {}

episode_numbers = [
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
    "6000",
    "7000",
    "8000",
    "9000",

]


def hook(module, input, output):
    activations[module] = output[0].detach()


def get_activations(model: DQN_Network, input, layer: int):
    """
    Get the activations of the model for the training data.
    """
    activations.clear()

    layer_activation = model.FC[layer]

    if isinstance(layer_activation, torch.nn.Linear):

        hook_handles[layer] = layer_activation.register_forward_hook(
            hook
        )  # TODO: Change this for feed forward

        activation = model(input)


def create_activation_dataset(dataset_path: str, model_path: str, layer: int = 0):

    fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 8,
        "number_of_rays": 100,
    }
    half_fov = fov_config["fov"] / 2
    matrix_size = calculate_fov_matrix_size(fov_config["ray_length"], half_fov)
    num_states = matrix_size[0] * matrix_size[1]
    num_states += 4

    state_dim = num_states

    action_space = 3

    # Initiate the network models
    model = DQN_Network(
        num_actions=action_space,
        input_dim=state_dim,
        device=device,
    )
    model = model.to(device)

    # Load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(model)
    episode_number = model_path.split("_")[-1].split(".")[0]
    # Activation file name
    activation_file = f"dataset/{dataset_path.split('/')[-1].split('.')[0]}_activations_{layer}_episode_{episode_number}.pt"

    # Read the dataset
    dataset = pd.read_csv(dataset_path)
    # print(dataset.head(10))
    activation_list = []
    activations.clear()
    for _, state in dataset.iterrows():
        # Get the state
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        # print(sequence)

        # state = torch.tensor(state_as_list).float().to(device)
        # print(state_tensor)

        state_tensor = state_tensor.unsqueeze(0)
        layer_model = model.FC[layer]
        if isinstance(layer_model, torch.nn.Linear):

            # Get the activations of the model
            activation = get_activations(model, state_tensor, layer)

            # print(activations)
            # print("Test", activations[layer_model])
            activation_list.append(activations[layer_model])
    if isinstance(layer_model, torch.nn.Linear):
        torch.save(activation_list, activation_file)

        return activation_file


class CAV:

    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

    def __init__(self):
        self.model = None
        self.cav_coef = None
        self.cav_list = []

    def cav_model(self, positive_file: str, negative_file: str):
        positive_dataset = torch.load(positive_file, weights_only=False)
        negative_dataset = torch.load(negative_file, weights_only=False)
        # Label the datasets
        positive_labels = [1] * len(positive_dataset)
        negative_labels = [0] * len(negative_dataset)
        dataset = CAV_dataset(positive_dataset, positive_labels)
        negative = CAV_dataset(negative_dataset, negative_labels)

        dataset.concat(negative)
        # Split the dataset

        length_train = int(0.8 * len(dataset))
        length_test = len(dataset) - length_train

        train_dataset, test_dataset = random_split(dataset, [length_train, length_test])
        # print("Hello", dataset[0][0])
        train_data = [dataset[i][0].numpy() for i in train_dataset.indices]
        train_labels = [dataset[i][1] for i in train_dataset.indices]
        test_data = [dataset[i][0].numpy() for i in test_dataset.indices]
        test_labels = [dataset[i][1] for i in test_dataset.indices]

        # Scale the data

        scaler = StandardScaler()

        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        # Train the model
        self.model = LogisticRegression()

        # print(self.model)

        # print("Train data: ", train_data)
        self.model.fit(train_data, train_labels)

        # Test the model
        score = self.model.score(test_data, test_labels)

        self.cav_coef = self.model.coef_
        return score

    def calculate_cav(self, concept: str, model_dir: str):

        model_list = os.listdir(model_dir)

        for model in model_list:
            print(model)
            if model.split("_")[-1].split(".")[0] not in episode_numbers:
                continue

            model_path = os.path.join(model_dir, model)
            episode_number = model.split("_")[-1].split(".")[0]
            for layer in range(5):

                print("Block: ", layer, model_path, episode_number)
                negative_file = create_activation_dataset(
                    f"./dataset/negative_{concept}.csv", model_path, layer
                )
                positive_file = create_activation_dataset(
                    f"./dataset/positive_{concept}.csv", model_path, layer
                )

                if negative_file is None or positive_file is None:
                    continue
                # positive_file = f"dataset/activations/positive_{concept}_activations_{block}.pt"
                # negative_file = f"dataset/activations/negative_{concept}_activations_{block}.pt"
                accuracy = self.cav_model(positive_file, negative_file)
                # Add the CAV to the list of CAVs
                self.cav_list.append((layer, int(episode_number), accuracy))

        # Save the CAV list
        torch.save(self.cav_list, f"./cav_list_{concept}.pt")

    def random_cav_model(self, random_file: str):

        dataset = torch.load(random_file, weights_only=False)

        length_train = int(0.8 * len(dataset))
        length_test = len(dataset) - length_train

        train_dataset, test_dataset = random_split(dataset, [length_train, length_test])
        train_data = [dataset[i][0].numpy() for i in train_dataset.indices]
        train_labels = [dataset[i][1] for i in train_dataset.indices]
        test_data = [dataset[i][0].numpy() for i in test_dataset.indices]
        test_labels = [dataset[i][1] for i in test_dataset.indices]

        # Scale the data
        # scaler = StandardScaler()

        # Train the model
        self.model = LogisticRegression()
        self.model.fit(train_data, train_labels)

        # Test the model
        score = self.model.score(test_data, test_labels)
        print("Score: ", score)
        self.cav_coef = self.model.coef_
        return score

    def calculate_random_cav(self, concept: str, model_dir: str):

        model_list = os.listdir(model_dir)
        for model in model_list:
            episode_number = model.split("_")[-1].split(".")[0]
            for block in range(3):
                print("Block: ", block)
                negative_file = f"dataset/negative_{concept}_activations_{block}_episode_{episode_number}.pt"
                positive_file = f"dataset/positive_{concept}_activations_{block}_episode_{episode_number}.pt"

                positive_dataset = torch.load(positive_file)
                negative_dataset = torch.load(negative_file)
                print(len(positive_dataset))
                # Label the datasets
                positive_labels = [
                    rd.randint(0, 1) for _ in range(len(positive_dataset))
                ]
                # print(positive_labels)
                negative_labels = [
                    rd.randint(0, 1) for _ in range(len(negative_dataset))
                ]
                dataset = CAV_dataset(positive_dataset, positive_labels)
                negative = CAV_dataset(negative_dataset, negative_labels)

                dataset.concat(negative)
                # Save to pt file
                random_file = f"dataset/random_{concept}_dataset_{block}.pt"
                torch.save(dataset, random_file)

                accuracy = self.random_cav_model(random_file)
                self.cav_list.append((block, episode_number, accuracy))

        # Save the CAV list
        torch.save(self.cav_list, f"./cav_list_{concept}.pt")

    def load_cav(self, concept: str):
        self.cav_list = torch.load(f"./cav_list_{concept}.pt")

    def plot_cav(self, concept: str):
        import numpy as np
        from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.mplot3d import Axes3D
        from scipy.interpolate import griddata

        if len(self.cav_list) == 0:
            self.load_cav(concept)
        print(self.cav_list[0])

        # Extract data
        blocks = np.array([t[0] for t in self.cav_list])
        episode_numbers = np.array([int(t[1]) for t in self.cav_list])
        accuracies = np.array([t[2] for t in self.cav_list])

        # Create a grid for interpolation
        block_lin = np.linspace(blocks.min(), blocks.max(), 3)
        episode_lin = np.linspace(episode_numbers.min(), episode_numbers.max(), 55)
        block_grid, episode_grid = np.meshgrid(block_lin, episode_lin)

        # Interpolate accuracy values onto the grid
        accuracy_grid = griddata(
            (blocks, episode_numbers),
            accuracies,
            (block_grid, episode_grid),
            method="cubic",
        )

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Surface plot
        surf = ax.plot_surface(
            block_grid, episode_grid, accuracy_grid, cmap="viridis", edgecolor="k"
        )

        # Labels
        ax.set_xlabel("Layer")
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


def main():
    cav = CAV()
    model_load_path = "../../../agent/model/feed_forward/serene-voice-977"
    # positive_file = "dataset/positive_wall_activations.pt"
    # negative_file = "dataset/negative_wall_activations.pt"
    cav.calculate_cav("goal", model_load_path)
    cav.plot_cav("goal")

    # cav.calculate_random_cav("rotating", model_load_path)
    # cav.plot_cav("random")


if __name__ == "__main__":
    main()
