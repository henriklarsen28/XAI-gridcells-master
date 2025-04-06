# TODO: CAR


import ast
import copy
import math
import os
import pickle
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
import wandb

# from logistic_regression import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, random_split

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent.ppo.transformer_decoder_decoupled_policy import TransformerPolicyDecoupled # TODO: This should be decoupled transformer
from utils import CAV_dataset, build_numpy_list_cav
from utils.calculate_fov import calculate_fov_matrix_size

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

activations = {}


def get_activation(name):
    def hook(module, input, output):
        activations[name] = {
            "input": input[0],
            "output": output,
        }

    return hook


def get_activations(
    model: TransformerPolicyDecoupled, input, block_name: str = None, embedding: bool = True
):
    """
    Get the activations of the model for the training data.
    """
    activations.clear()

    if embedding:
        output = get_embedding_activations(model, input)
        # print('q val type', q_value[0])
        return output

    block_int = int(block_name.split("_")[-1])

    block = model.blocks[block_int]
    block.register_forward_hook(get_activation(block_name))
    output, _, _, _ = model(input)
    # print("Block activations")
    return output


def get_embedding_activations(model: TransformerPolicyDecoupled, input):

    embedding = model.token_embedding
    embedding.register_forward_hook(get_activation("embedding"))
    output, _, _, _ = model(input)
    # print("Embedding activations")
    # print('q val', q_value.size())
    return output


def create_activation_dataset(
    dataset_path: str,
    model_path: str,
    block: int = 0,
    embedding: bool = False,
    requires_grad: bool = False,
    action_index: int = 0,
):
    if not os.path.exists(dataset_path):
        return torch.tensor([]), torch.tensor([])
    print("Creating activation dataset for ", model_path)
    fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 15,
        "number_of_rays": 40,
    }
    half_fov = fov_config["fov"] / 2
    matrix_size = calculate_fov_matrix_size(fov_config["ray_length"], half_fov)
    num_states = matrix_size[0] * matrix_size[1]
    num_states += 1

    num_envs = 6
    sequence_length = 30
    n_embd = 196
    n_head = 8
    n_layer = 2
    dropout = 0.2
    state_dim = num_states

    action_space = 2

    # Initiate the network models
    model = TransformerPolicyDecoupled(
        input_dim=state_dim,
        output_dim=action_space,
        num_envs=num_envs,
        block_size=sequence_length,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        device=device,
    )
    model = model.to(device)

    # Load the model
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    episode_number = model_path.split("_")[-1].split(".")[0]

    # Read the dataset
    dataset = pd.read_csv(dataset_path)
    print("Reading dataset:", dataset_path)
    # print(dataset.head(10))
    sequences = [
        [torch.tensor(ast.literal_eval(state)) for state in states]
        for _, states in dataset.head(1500).iterrows()
    ]

    state_tensors = torch.stack(
        [torch.stack(sequence).float().to(device) for sequence in sequences]
    ).requires_grad_(requires_grad)
    # print(state_tensors.shape)
    if embedding:
        output = get_activations(model, state_tensors, embedding=embedding)
        activation = activations["embedding"]["output"]
        # print("Activation embedding:", activation.shape, activation)

    else:
        block_name = f"block_{block}"
        output = get_activations(model, state_tensors, block_name, embedding=embedding)
        activation = activations[block_name]["output"][0]
        # TODO: Look inside the block
        # TODO: After attention, after feed forward
        #print("Activations:", activation)

    assert isinstance(activation, torch.Tensor), "Activation must be a tensor"

    return activation, output


class CAR:

    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

    def __init__(self, kernel = "rbf", kernel_width=None):
        self.model = None
        self.cav_coef = None
        self.kernel = kernel
        self.kernel_width = kernel_width
        self.cav_list = []

    def read_dataset(
        self,
        concept,
        dataset_directory_train,
        dataset_directory_test,
        model_path,
        block,
        embedding: bool = False,
        sensitivity: bool = True,
        action_index: int = 0,
        save_path: str = None,
        episode_number: str = None,
    ):

        positive, output_positive = create_activation_dataset(
            f"{dataset_directory_train}/{concept}_positive_train.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
            action_index=action_index,
        )
        assert isinstance(positive, torch.Tensor), "Positive must be a tensor"

        negative, output_negative = create_activation_dataset(
            f"{dataset_directory_train}/{concept}_negative_train.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
            action_index=action_index,
        )
        assert isinstance(negative, torch.Tensor), "Negative must be a tensor"

        positive_test, output_positive_test = create_activation_dataset(
            f"{dataset_directory_test}/{concept}_positive_test.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
            action_index=action_index,
        )
        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

        negative_test, output_negative_test = create_activation_dataset(
            f"{dataset_directory_test}/{concept}_negative_test.csv",
            model_path,
            block,
            embedding=embedding,
            requires_grad=sensitivity,
            action_index=action_index,
        )
        assert isinstance(negative, torch.Tensor), "Negative_test must be a tensor"

        # Make sure the activations are saved the same way as the concept plots
        if embedding:
            block = 0
        else:
            block += 1

        self.save_activations(
            positive, positive_test, concept, save_path, block, episode_number
        )

        return (
            positive,
            negative,
            positive_test,
            negative_test,
            output_positive,
            output_negative,
            output_positive_test,
            output_negative_test,
        )

    def save_activations(
        self, activations_train, activations_test, concept, save_path, block, episode
    ):
        save_path_activations_train = os.path.join(
            save_path, f"activations/train/{concept}/{episode}"
        )
        save_path_activations_test = os.path.join(
            save_path, f"activations/test/{concept}/episode_{episode}"
        )
        file_name = f"activation_{concept}_block_{block}.pt"

        os.makedirs(save_path_activations_train, exist_ok=True)
        os.makedirs(save_path_activations_test, exist_ok=True)

        save_path_activations_train = os.path.join(
            save_path_activations_train, file_name
        )
        save_path_activations_test = os.path.join(save_path_activations_test, file_name)

        torch.save(activations_train, save_path_activations_train)
        torch.save(activations_test, save_path_activations_test)

        print(f"Activations train saved to {save_path_activations_train}")
        print(f"Activations test saved to {save_path_activations_test}")
    
    def get_kernel_function(self) -> callable:
        """
        Get the kernel funtion underlying the CAR
        Returns: kernel function as a callable with arguments (h1, h2)
        """
        if self.kernel == "rbf":
            if self.kernel_width is not None:
                kernel_width = self.kernel_width
            else:
                kernel_width = 1.0
            latent_dim = self.concept_reps.shape[-1]
            # We unstack the tensors to return a kernel matrix of shape len(h1) x len(h2)!
            return lambda h1, h2: torch.exp(
                -torch.sum(
                    ((h1.unsqueeze(1) - h2.unsqueeze(0)) / (latent_dim * kernel_width))
                    ** 2,
                    dim=-1,
                )
            )
        elif self.kernel == "linear":
            return lambda h1, h2: torch.einsum(
                "abi, abi -> ab", h1.unsqueeze(1), h2.unsqueeze(0)
            )

    def cav_model(
        self,
        positive_train: torch.Tensor,
        negative_train: torch.Tensor,
        positive_test: torch.Tensor,
        negative_test: torch.Tensor,
        save_path: str,
        concept: str,
        episode: str,
        block: str,
    ):

        positive_train_labels = np.ones(len(positive_train))
        negative_train_labels = np.zeros(len(negative_train))

        positive_test_labels = np.ones(len(positive_test))
        negative_test_labels = np.zeros(len(negative_test))
        # Split the dataset


        positive_train_np = build_numpy_list_cav(positive_train)
        negative_train_np = build_numpy_list_cav(negative_train)


        train_data = np.concatenate((positive_train_np, negative_train_np), axis=0)
        train_labels = np.concatenate(
            (positive_train_labels, negative_train_labels), axis=0
        )

        positive_test_np = build_numpy_list_cav(positive_test)
        negative_test_np = build_numpy_list_cav(negative_test)

        test_data = np.concatenate((positive_test_np, negative_test_np), axis=0)
        test_labels = np.concatenate(
            (positive_test_labels, negative_test_labels), axis=0
        )

        # Shuffle the dataset
        train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
        test_data, test_labels = shuffle(test_data, test_labels, random_state=42)
        # Train the model
        self.model = SVC(kernel=self.kernel, C=0.01, gamma=0.01)

        self.model.fit(train_data, train_labels)

        # save the model as pickle file
        save_path_models = os.path.join(save_path, "models_car")
        os.makedirs(save_path_models, exist_ok=True)

        concept_model_path = os.path.join(save_path_models, concept)
        os.makedirs(concept_model_path, exist_ok=True)

        save_path_models = os.path.join(concept_model_path, f"{concept}_block_{block}_episode_{episode}.pkl")
        pickle.dump(self.model, open(save_path_models, "wb"))



        #score = np.mean(test_data @ cav_coef.T > 0)
        y_pred = self.model.predict(test_data)
        score = accuracy_score(test_labels, y_pred)
        print("Accuracy: ", score)
        score = (score - 0.5) * 2
        # Perform relu on the score
        score = max(0, score)
        
        return score

    def calculate_single_cav(
        self,
        block: int,
        episode_number: str,
        positive_train,
        negative_train,
        positive_test,
        negative_test,
        save_path,
        concept,
    ):

        # positive_file = f"dataset/activations/positive_{concept}_activations_{block}.pt"
        # negative_file = f"dataset/activations/negative_{concept}_activations_{block}.pt"
        accuracy = self.cav_model(
            positive_train,
            negative_train,
            positive_test,
            negative_test,
            save_path,
            concept,
            str(episode_number),
            str(block),
        )
        self.cav_list.append((block, episode_number, accuracy))

    def calculate_cav(
        self,
        concept: str,
        dataset_directory_train: str,
        dataset_directory_test: str,
        model_dir: str,
        sensitivity: bool = False,
        action_index: int = 0,
        episode_numbers: list = None,
        save_path: str = None,
    ):
        model_list = os.listdir(model_dir)
        save_path_activations = os.path.join(save_path, "plot_data")
        os.makedirs(save_path_activations, exist_ok=True)

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
                q_values_positive,
                q_values_negative,
                q_values_positive_test,
                q_values_negative_test,
            ) = self.read_dataset(
                concept=concept,
                dataset_directory_train=dataset_directory_train,
                dataset_directory_test=dataset_directory_test,
                model_path=model_path,
                block=0,
                embedding=True,
                sensitivity=sensitivity,
                action_index=action_index,
                save_path=save_path,
                episode_number=episode_number,
            )

            self.calculate_single_cav(
                0,
                episode_number,
                positive,
                negative,
                positive_test,
                negative_test,
                save_path,
                concept,
            )

            for block in range(
                1, 3
            ):  # NOTE: This must be changed depending on the num blocks (2 blocks here)
                # load dataset
                (
                    positive,
                    negative,
                    positive_test,
                    negative_test,
                    q_values_positive,
                    q_values_negative,
                    q_values_positive_test,
                    q_values_negative_test,
                ) = self.read_dataset(
                    concept=concept,
                    dataset_directory_train=dataset_directory_train,
                    dataset_directory_test=dataset_directory_test,
                    model_path=model_path,
                    block=block - 1,
                    embedding=False,
                    sensitivity=sensitivity,
                    action_index=action_index,
                    save_path=save_path,
                    episode_number=episode_number,
                )

                # print("Block: ", block, model_path, episode_number)
                self.calculate_single_cav(
                    block,
                    episode_number,
                    positive,
                    negative,
                    positive_test,
                    negative_test,
                    save_path,
                    concept,
                )

        # Save the CAV list
        torch.save(self.cav_list, os.path.join(save_path_activations, f"{concept}.pt"))
        print(
            "CAV list saved to:", os.path.join(save_path_activations, f"{concept}.pt")
        )
        # torch.save(self.tcav_list, f"./results/tcav/tcav_list_{concept}.pt")

   
    def load_cav(self, concept: str):
        cav_list = torch.load(f"./results/car/cav_list_{concept}.pt")
        return cav_list

    def plot_cav(
        self,
        concept: str,
        tcav: bool = False,
        episode_numbers: list = None,
        save_path: str = None,
    ):

        save_path_plots = os.path.join(save_path, "plots_car")
        if not os.path.exists(save_path_plots):
            os.makedirs(save_path_plots, exist_ok=True)
        save_path_plots = os.path.join(save_path_plots, concept)

        if len(self.cav_list) == 0:
            self.cav_list = self.load_cav(concept)
        # print(self.cav_list[0])

        self.plot(concept, self.cav_list, save_path_plots)

    def plot_tcav(self, concept: str, action: int = 0):

        self.tcav_list = [
            (outer_key, int(inner_key), inner_value)
            for outer_key, inner_dict in self.tcav_list.items()
            for inner_key, inner_value in inner_dict.items()
        ]
        # self.tcav_list = list(self.tcav_list.items())
        # print(self.tcav_list)

        self.plot(concept, self.tcav_list, f"tcav_{concept}", action=action)

    def plot(self, concept: str, cav_list: list, save_path: str, action: int = 0):

        # Extract data
        blocks = np.array([t[0] for t in cav_list])
        episode_numbers = np.array([int(t[1]) for t in cav_list])
        accuracies = np.array([t[2] for t in cav_list])

        # Create a grid for interpolation
        block_lin = np.linspace(blocks.min(), blocks.max(), 3)
        episode_lin = np.linspace(episode_numbers.min(), episode_numbers.max(), 550)
        block_grid, episode_grid = np.meshgrid(block_lin, episode_lin)

        # Interpolate accuracy values onto the grid
        accuracy_grid = griddata(
            (blocks, episode_numbers),
            accuracies,
            (block_grid, episode_grid),
            method="linear",
        )

        norm = Normalize(vmin=0, vmax=1)
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
        ax.set_ylabel("Episode")
        ax.set_zlabel("Accuracy")

        # range
        ax.set_xlim(blocks.min(), blocks.max())
        ax.set_ylim(episode_numbers.min(), episode_numbers.max())
        ax.set_zlim(0, 1)

        ax.xaxis.set_major_locator(MultipleLocator(1))

        # Colorbar for accuracy
        fig.colorbar(surf, ax=ax, label="Accuracy")

        plt.savefig(save_path + f"_{action}.png")
        print(f"Plot saved to {save_path}_{action}.png")
        # plt.show()


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

        return self.total_tcav

    def get_tcav(self):
        return self.total_tcav

