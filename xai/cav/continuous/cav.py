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
import wandb
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, random_split

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

from agent.ppo.transformer_decoder_decoupled_policy import (
    TransformerPolicyDecoupled,  # TODO: This should be decoupled transformer
)
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
    model: TransformerPolicyDecoupled,
    input,
    block_name: str = None,
    embedding: bool = True,
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
    output, _, env_class, _ = model(input)
    # print("Block activations")
    return output, env_class


def get_embedding_activations(model: TransformerPolicyDecoupled, input):

    embedding = model.token_embedding
    embedding.register_forward_hook(get_activation("embedding"))
    output, _, env_class, _ = model(input)
    # print("Embedding activations")
    # print('q val', q_value.size())
    return output, env_class


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
    if dataset_path.__contains__("rot90"):
        state_tensors[:,:,-1] = state_tensors[:,:,-1] + 90/360

    # print(state_tensors.shape)
    if embedding:
        output, env_class = get_activations(model, state_tensors, embedding=embedding)
        activation = activations["embedding"]["output"]
        # print("Activation embedding:", activation.shape, activation)

    else:
        block_name = f"block_{block}"
        output, env_class = get_activations(
            model, state_tensors, block_name, embedding=embedding
        )
        activation = activations[block_name]["output"][0]
        # TODO: Look inside the block
        # TODO: After attention, after feed forward
        # print("Activations:", activation)

    assert isinstance(activation, torch.Tensor), "Activation must be a tensor"
    return activation, env_class


class CAV:

    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

    def __init__(self, activation_path: str = None):
        self.model = None
        self.cav_coef = None
        self.cav_list = []
        self.tcav_list = []
        self.activation_path = activation_path

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

        transformer_block = block - 1
        activation_file_train = os.path.join(
            self.activation_path,
            "train",
            concept,
            str(episode_number),
            f"activation_{concept}_block_{block}.pt",
        )
        if os.path.exists(activation_file_train) and not sensitivity:
            # Read the activation file
            positive = torch.load(activation_file_train, weights_only=True)
            output_positive = None
        else:
            positive, output_positive = create_activation_dataset(
                f"{dataset_directory_train}/{concept}_positive_train.csv",
                model_path,
                transformer_block,
                embedding=embedding,
                requires_grad=sensitivity,
                action_index=action_index,
            )

        assert isinstance(positive, torch.Tensor), "Positive must be a tensor"

        negative, output_negative = create_activation_dataset(
            f"{dataset_directory_train}/{concept}_negative_train.csv",
            model_path,
            transformer_block,
            embedding=embedding,
            requires_grad=sensitivity,
            action_index=action_index,
        )
        assert isinstance(negative, torch.Tensor), "Negative must be a tensor"

        activation_file_test = os.path.join(
            self.activation_path,
            "test",
            concept,
            str(episode_number),
            f"activation_{concept}_block_{block}.pt",
        )

        if os.path.exists(activation_file_test) and not sensitivity:
            print("Loading positive test")
            # Read the activation file
            positive_test = torch.load(activation_file_test, weights_only=True)
            output_positive_test = None
        else:
            positive_test, output_positive_test = create_activation_dataset(
                f"{dataset_directory_test}/{concept}_positive_test.csv",
                model_path,
                transformer_block,
                embedding=embedding,
                requires_grad=sensitivity,
                action_index=action_index,
            )
        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

        negative_test, output_negative_test = create_activation_dataset(
            f"{dataset_directory_test}/{concept}_negative_test.csv",
            model_path,
            transformer_block,
            embedding=embedding,
            requires_grad=sensitivity,
            action_index=action_index,
        )
        assert isinstance(negative, torch.Tensor), "Negative_test must be a tensor"

        # Make sure the activations are saved the same way as the concept plots
        if save_path is not None:
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
            save_path, f"activations/test/{concept}/{episode}"
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
    ) -> tuple[float, np.ndarray]:
        """
        Trains a logistic regression model to compute Concept Activation Vectors (CAVs)
        and evaluates its performance.

        Args:
            positive_train (torch.Tensor): Tensor containing positive training examples.
            negative_train (torch.Tensor): Tensor containing negative training examples.
            positive_test (torch.Tensor): Tensor containing positive testing examples.
            negative_test (torch.Tensor): Tensor containing negative testing examples.
            save_path (str): Directory path to save the trained model.
            concept (str): Name of the concept being evaluated.
            episode (str): Identifier for the episode.
            block (str): Identifier for the block.

        Returns:
            tuple: A tuple containing:
                - score (float): The adjusted accuracy score of the model.
                - cav_coef (np.ndarray): The coefficients of the trained logistic regression model.
        """

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
        model = LogisticRegression(penalty="l2", C=0.01, solver="lbfgs", max_iter=1000)

        model.fit(train_data, train_labels)
        if save_path is not None:
            # save the model as pickle file
            save_path_models = os.path.join(save_path, "models")
            os.makedirs(save_path_models, exist_ok=True)

            concept_model_path = os.path.join(save_path_models, concept)
            os.makedirs(concept_model_path, exist_ok=True)

            save_path_models = os.path.join(
                concept_model_path, f"{concept}_block_{block}_episode_{episode}.pkl"
            )
            pickle.dump(model, open(save_path_models, "wb"))

        # Test the model
        # score = model.score(test_data, test_labels)
        # score = accuracy_score(test_labels, model.predict(test_data))

        cav_coef = model.coef_

        score = np.mean(test_data @ cav_coef.T > 0)

        score = (score - 0.5) * 2
        # Perform relu on the score
        score = max(0, score)

        print("Score: ", score)

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
        save_path_activations = os.path.join(save_path, "plot_data")
        os.makedirs(save_path_activations, exist_ok=True)

        for model in model_list:
            model_path = os.path.join(model_dir, model)
            episode_number = model.split("_")[-1].split(".")[0]

            if episode_number not in episode_numbers:
                continue

            for block in range(
                0, 3
            ):  # NOTE: This must be changed depending on the num blocks (2 blocks here)
                # load dataset

                embedding = False
                if block == 0:
                    embedding = True
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
                    block=block,
                    embedding=embedding,
                    sensitivity=sensitivity,
                    action_index=action_index,
                    save_path=save_path,
                    episode_number=episode_number,
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


class TCAV:

    def __init__(
        self,
        activation_path: str = None,
        episode: int = 1,
        block: int = 0,
        dataset_path_train: str = None,
        dataset_path_test: str = None,
        model_dir: str = None,
    ):
        self.activation_path = activation_path
        self.cav = CAV(activation_path=activation_path)
        self.episode = episode
        self.block = block
        self.outputs = 6
        self.dataset_directory_train = dataset_path_train
        self.dataset_directory_test = dataset_path_test
        self.model_dir = model_dir
        self.tcav_list = []

    def load_cav(self, concept: str) -> LogisticRegression:
        cav_coef = pickle.load(
            open(
                os.path.join(
                    self.activation_path,
                    "..",
                    "models",
                    concept,
                    f"{concept}_block_{self.block}_episode_{self.episode}.pkl",
                ),
                "rb",
            )
        )
        print(cav_coef)
        return cav_coef

    def calculate_sensitivity(self, concept, action):

        # cav_coef = self.load_cav(concept).coef_

        fov_config = {
            "fov": math.pi / 1.5,
            "ray_length": 15,
            "number_of_rays": 40,
        }
        half_fov = fov_config["fov"] / 2
        matrix_size = calculate_fov_matrix_size(fov_config["ray_length"], half_fov)
        num_states = matrix_size[0] * matrix_size[1]
        num_states += 1

        num_envs = self.outputs
        sequence_length = 30
        n_embd = 196
        n_head = 8
        n_layer = 2
        dropout = 0.2
        state_dim = num_states

        action_space = 2

        # Initiate the network models
        """model = TransformerPolicyDecoupled(
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
        model = model.to(device)"""

        model_path = os.path.join(self.model_dir, f"policy_network_{self.episode}.pth")

        if self.block == 0:
            embedding = True
            transformer_block = 0
        else:
            embedding = False
            transformer_block = self.block - 1
        """
        # Read dataset
        (
            positive,
            negative,
            positive_test,
            negative_test,
            output_positive,
            output_negative,
            output_positive_test,
            output_negative_test,
        ) = self.cav.read_dataset(
            concept=concept,
            dataset_directory_train=self.dataset_directory_train,
            dataset_directory_test=self.dataset_directory_test,
            model_path=model_path,
            block=transformer_block,
            embedding=embedding,
            sensitivity=True,
            action_index=0,
            save_path=None,
            episode_number=self.episode,
        )

        _, cav_coef = self.cav.cav_model(
            positive,
            negative,
            positive_test,
            negative_test,
            None,
            concept,
            transformer_block,
            self.episode,
        )"""

        positive_test, output_positive_test = create_activation_dataset(
            f"{self.dataset_directory_test}/{concept}_positive_test.csv",
            model_path,
            transformer_block,
            embedding=embedding,
            requires_grad=True,
            action_index=action,
        )


        network_output = output_positive_test[
            :, :, action
        ]  # torch.sum(output_positive_test, dim=1)

        assert isinstance(
            positive_test, torch.Tensor
        ), f"Activations must be a tensor{type(positive_test)}"
        # assert all(type(a) for a in activations) == torch.tensor, f"Activations must be a tensor {type(activations)}"
        assert isinstance(
            network_output, torch.Tensor
        ), "Network output must be a tensor"

        # assert all(type(n) for n in network_output) == torch.Tensor, "Network output must be a tensor"

        outputs = torch.autograd.grad(
            outputs=network_output,
            inputs=positive_test,
            grad_outputs=torch.ones_like(network_output),
            retain_graph=True,
        )[0]

        outputs = outputs[:, -2:, :].flatten(1)
        print(outputs.shape)
        grad_flattened = outputs.view(outputs.size(0), -1).detach().cpu().numpy()


        cav_coef = self.load_cav(concept).coef_

        return np.dot(grad_flattened, cav_coef.T)

    def calculate_tcav(
        self,
        concept: str,
        action: int = 0,
    ):
        sensitivity = self.calculate_sensitivity(concept, action)
        tcav = (sensitivity > 0).mean()

        print("TCAV: ", tcav)

        return tcav


class Analysis:
    def __init__(self, average: int = 5):
        self.average = average
        self.total_tcav = {}

    def add_total_tcav_scores(
        self,
        concept,
        action,
        sensitivity,
    ):
        if concept not in self.total_tcav:
            self.total_tcav[concept] = {}
        if action not in self.total_tcav[concept]:
            self.total_tcav[concept][action] = 0
        self.total_tcav[concept][action] += sensitivity

    def calculate_average_tcav(self):
        # Calculate the average TCAV score for each concept and action
        average_tcav = {}
        for concept, actions in self.total_tcav.items():
            average_tcav[concept] = {}
            for action, scores in actions.items():
                average_tcav[concept][action] = self.total_tcav[concept][action] / self.average

        # Calculate the standard deviation of TCAV scores
        std_scores = {}
        for concept, actions in self.total_tcav.items():
            for action, scores in actions.items():
                std_scores[(concept, action)] = np.std(scores)

        return average_tcav, std_scores

    def get_tcav(self):
        return self.total_tcav

    def plot_tcav(self):

        """action_colors = {
            0: '#e41a1c',  # red
            1: '#377eb8',  # blue
            2: '#4daf4a',  # green
            3: '#984ea3',  # purple
            4: '#ff7f00',  # orange
            5: '#ffff33'   # yellow
        }

        # Create a bar plot for the TCAV scores
        # Show the standard deviation as error bars
        average_tcav, std_scores = self.calculate_average_tcav()
        concepts = list(average_tcav.keys())
        labels = sorted(list(average_tcav[concepts[0]].keys()))
        x = len(concepts)
        print("concepts", concepts, x)
        bar_width = 0.25
        print("Average TCAV scores: ", average_tcav)
        print("Standard deviation of TCAV scores: ", std_scores)
        fig, ax = plt.subplots()
        for i, (concept, actions) in enumerate(average_tcav.items()):
            for j, (action, score) in enumerate(actions.items()):
                print("Actions:", action)
                print("Concept:", concept)
                print("Score:", score)
                stds = std_scores.get((concept, action), 0)
                print("Std:", stds)

                ax.bar(i + j * bar_width, score, bar_width,
                    color=action_colors[action], yerr=stds)         #ax.errorbar(y=x + i * bar_width, x=means,yerr=stds, label=concept)

        # Set labels 
        
        ax.set_xlabel("Concepts")
        ax.set_ylabel("TCAV Score")

        ax.set_title("TCAV Scores")
        ax.legend()

        plt.show()"""
        avg_scores, std_scores = self.calculate_average_tcav()

        concepts = list(avg_scores.keys())
        actions = sorted(avg_scores[concepts[0]].keys())
        num_actions = len(actions)

        # Colors for 6 actions
        action_colors = {
            0: '#e41a1c',  # red
            1: '#377eb8',  # blue
            2: '#4daf4a',  # green
            3: '#984ea3',  # purple
            4: '#ff7f00',  # orange
            5: '#ffff33'   # yellow
        }

        x = np.arange(len(concepts))
        bar_width = 0.12

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, action in enumerate(actions):
            values = [avg_scores[concept][action] for concept in concepts]
            errors = [std_scores[(concept, action)] for concept in concepts]
            ax.bar(x + i * bar_width, values, bar_width,
                label=f'Action {action}',
                color=action_colors[action],
                yerr=errors)

        # Axis & labels
        ax.set_xlabel("Grids")
        ax.set_ylabel("TCAV Score")
        ax.set_title("TCAV Scores per Action for Each Concept")
        ax.set_xticks(x + (num_actions / 2 - 0.5) * bar_width)
        ax.set_xticklabels(concepts, rotation=90)
        ax.set_ylim(0, 1)
        ax.legend(title="Action")




        plt.tight_layout()
        plt.show()
