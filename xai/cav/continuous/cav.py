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
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, random_split

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent.ppo.transformer_decoder_policy import TransformerPolicy
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
    model: TransformerPolicy, input, block_name: str = None, embedding: bool = True
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


def get_embedding_activations(model: TransformerPolicy, input):

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

    num_envs = 3
    sequence_length = 30
    n_embd = 196
    n_head = 8
    n_layer = 2
    dropout = 0.2
    state_dim = num_states

    action_space = 2

    # Initiate the network models
    model = TransformerPolicy(
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
        for _, states in dataset.iterrows()
    ]

    state_tensors = torch.stack(
        [torch.stack(sequence).float().to(device) for sequence in sequences]
    ).requires_grad_(requires_grad)
    # print(state_tensors.shape)
    if embedding:
        output = get_activations(model, state_tensors, embedding=embedding)
        activation = activations["embedding"]["output"]
        # print("Activation embedding:", activation)

    else:
        block_name = f"block_{block}"
        output = get_activations(model, state_tensors, block_name, embedding=embedding)
        activation = activations[block_name]["output"][0]
        # print("Activations:", activation)

    assert isinstance(activation, torch.Tensor), "Activation must be a tensor"

    return activation, output


class CAV:

    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

    def __init__(self):
        self.model = None
        self.cav_coef = None
        self.cav_list = []
        self.tcav_list = []

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
        self.model = LogisticRegression(max_iter=300)

        self.model.fit(train_data, train_labels)

        # save the model as pickle file
        save_path_models = os.path.join(save_path, "models")
        os.makedirs(save_path_models, exist_ok=True)

        concept_model_path = os.path.join(save_path_models, concept)
        os.makedirs(concept_model_path, exist_ok=True)

        save_path_models = os.path.join(concept_model_path, f"{concept}_block_{block}_episode_{episode}.pkl")
        pickle.dump(self.model, open(save_path_models, "wb"))

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
        save_path,
        concept,
    ):

        # positive_file = f"dataset/activations/positive_{concept}_activations_{block}.pt"
        # negative_file = f"dataset/activations/negative_{concept}_activations_{block}.pt"
        accuracy, cav = self.cav_model(
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
        return cav

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
        save_path_activations = os.path.join(save_path, "activations")
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
            )

            cav = self.calculate_single_cav(
                0,
                episode_number,
                positive,
                negative,
                positive_test,
                negative_test,
                save_path,
                concept,
            )
            # print("HEllo")
            # Calculate the tcav
            if sensitivity:
                self.calculate_tcav(
                    cav, positive_test, q_values_positive_test, 0, episode_number
                )

            for block in range(
                1, 3
            ):  # NOTE: This must be changed depending on the num blocks (3 blocks here)
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
                )

                # print("Block: ", block, model_path, episode_number)
                cav = self.calculate_single_cav(
                    block,
                    episode_number,
                    positive,
                    negative,
                    positive_test,
                    negative_test,
                    save_path,
                    concept,
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
        torch.save(self.cav_list, os.path.join(save_path_activations, f"{concept}.pt"))
        print(
            "CAV list saved to:", os.path.join(save_path_activations, f"{concept}.pt")
        )
        # torch.save(self.tcav_list, f"./results/tcav/tcav_list_{concept}.pt")

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
        # print(activations[0])
        # print(network_output[0])
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
        cav_list = torch.load(f"./results/cav/cav_list_{concept}.pt")
        return cav_list

    def plot_cav(
        self,
        concept: str,
        tcav: bool = False,
        episode_numbers: list = None,
        save_path: str = None,
    ):

        save_path_plots = os.path.join(save_path, "plots")
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
        block_lin = np.linspace(blocks.min(), blocks.max(), 4)
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


"""def main():
    
    model_name ="model_rose-pyramid-152"
    model_load_path = f"../../agent/dqn/models/{model_name}"
    #map_name = "map_conditional_prob_10_4"
    #map_name = "map_circular_4_5"
    map_name = "map_two_rooms_9_8"

    dataset_directory_train = f"./dataset/{model_name}/{map_name}/train"
    dataset_directory_test = f"./dataset/{model_name}/{map_name}/test"
    dataset_directory_random = f"./dataset/{model_name}/{map_name}"
    save_path = f"./results/cav/{model_name}/{map_name}"

    #concept = "random"
    #cav = CAV()
    #cav.calculate_cav(concept, dataset_directory_random, dataset_directory_random, model_load_path, sensitivity=False)
    #cav.plot_cav(concept)

    for i in range(15):
        concept = f"grid_observations_{i}"
        # grid_observation_dataset(model_name, concept)
        # cav = CAV()
        for action in range(1):
            average = 1
            pass
            analysis = Analysis(average)
            for _ in range(average):
                cav = CAV()
                cav.calculate_cav(concept, dataset_directory_train, dataset_directory_test, model_load_path, sensitivity=False, action_index=action)
                cav.plot_cav(concept)
                #tcav = cav.tcav_list
                #analysis.add_total_tcav_scores(tcav)

    #cav_list = torch.load(f"./cav_list_{concept}.pt")
    #cav.cav_list = cav_list
    #cav.plot_cav(concept)

    # analysis.calculate_average_tcav()
    # total_tcav = analysis.get_tcav()
    # Save tcav list
    # torch.save(total_tcav, f"./tcav_list_{concept}_action_{action}.pt")

    # print(analysis.total_tcav)
    # cav.tcav_list = total_tcav

    # cav.plot_tcav(concept, action=action)
    total_tcav0 = torch.load("tcav_list_goal_action_0.pt")
    total_tcav1 = torch.load("tcav_list_goal_action_1.pt")
    total_tcav2 = torch.load("tcav_list_goal_action_2.pt")

    # Loop through the tcav list and calculate the average
    average_tcav = defaultdict(lambda: defaultdict(float))
    for block in total_tcav0:
        for episode in total_tcav0[block]:
            average_tcav[block][episode] = (
                total_tcav0[block][episode]
                + total_tcav1[block][episode]
                + total_tcav2[block][episode]
            ) / 3
    cav.tcav_list = average_tcav

    cav.plot_tcav(concept, action=0)
    #cav.calculate_cav("goal", model_load_path)
    #cav.plot_cav("random")


if __name__ == "__main__":
    main()
"""
