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
from cav import build_numpy_list_cav, create_activation_dataset
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, random_split


class RelativeCAV:

    # Get the activations of the model
    # activations = get_activations(model, _, model.blocks[0])
    # print(activations)

    def __init__(self, activation_path: str = ""):
        self.model = None
        self.cav_coef = None
        self.cav_list = []
        self.tcav_list = []
        self.activation_path = activation_path

    def read_dataset(
        self,
        concept,
        concept2,
        dataset_directory_train,
        dataset_directory_test,
        model_path,
        block,
        embedding: bool = False,
        sensitivity: bool = True,
        save_path: str = None,
        episode: int = None,
    ):
        
        dataset_exists = True

        # Check if the concept exists by checking if the concept name exists in the dataset
        activation_file_train = os.path.join(self.activation_path, "train", concept, str(episode), f"activation_{concept}_block_{block}.pt")
        if os.path.exists(activation_file_train):
            # Read the activation file
            positive = torch.load(activation_file_train, weights_only=True)
            output_positive = None
        else:
            positive, output_positive = create_activation_dataset(
                f"{dataset_directory_train}/{concept}_positive_train.csv",
                model_path,
                block-1,
                embedding=embedding,
                requires_grad=sensitivity,
            )
            dataset_exists = False

        assert isinstance(positive, torch.Tensor), "Positive must be a tensor"

        activation_file_train = os.path.join(self.activation_path, "train", concept2, str(episode), f"activation_{concept2}_block_{block}.pt")
        if os.path.exists(activation_file_train):
            # Read the activation file
            negative = torch.load(activation_file_train, weights_only=True)
            output_negative = None
        else:
            negative, output_negative = create_activation_dataset(
                f"{dataset_directory_train}/{concept2}_positive_train.csv",
                model_path,
                block-1,
                embedding=embedding,
                requires_grad=sensitivity,
            )
            dataset_exists = False

        assert isinstance(negative, torch.Tensor), "Negative must be a tensor"
        
        activation_file_test = os.path.join(self.activation_path, "test", concept, f"episode_{episode}", f"activation_{concept}_block_{block}.pt")

        if os.path.exists(activation_file_test):
            print("Loading positive test")
            # Read the activation file
            positive_test = torch.load(activation_file_test, weights_only=True)
            output_positive_test = None
        else:
            positive_test, output_positive_test = create_activation_dataset(
                f"{dataset_directory_test}/{concept}_positive_test.csv",
                model_path,
                block-1,
                embedding=embedding,
                requires_grad=sensitivity,
            )
            dataset_exists = False

        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

        activation_file_test = os.path.join(self.activation_path, "test", concept2, f"episode_{episode}", f"activation_{concept2}_block_{block}.pt")
        print(activation_file_test)
        if os.path.exists(activation_file_test):
            # Read the activation file
            print("Loading negative test")
            negative_test = torch.load(activation_file_test, weights_only=True)
            output_negative_test = None
        else:
            negative_test, output_negative_test = create_activation_dataset(
                f"{dataset_directory_test}/{concept2}_positive_test.csv",
                model_path,
                block-1,
                embedding=embedding,
                requires_grad=sensitivity,
            )
            dataset_exists = False
        assert isinstance(negative, torch.Tensor), "Negative_test must be a tensor"

        # Make sure the activations are saved the same way as the concept plots
        if not dataset_exists:
            
            self.save_activations(
                positive, positive_test, concept, save_path, block, episode
            )
            self.save_activations(
                negative, negative_test, concept2, save_path, block, episode
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
        self.model = LogisticRegression(
            penalty="l2", C=0.1, solver="lbfgs", max_iter=1000, random_state=42
        )

        self.model.fit(train_data, train_labels)

        # save the model as pickle file
        save_path_models = os.path.join(save_path, "models_relative_cav")
        os.makedirs(save_path_models, exist_ok=True)

        concept_model_path = os.path.join(save_path_models, concept)
        os.makedirs(concept_model_path, exist_ok=True)

        save_path_models = os.path.join(
            concept_model_path, f"{concept}_block_{block}_episode_{episode}.pkl"
        )
        pickle.dump(self.model, open(save_path_models, "wb"))

        # Test the model
        #score = self.model.score(test_data, test_labels)

        # Get the accuracy of the model
        #score = accuracy_score(test_labels, self.model.predict(test_data))

        # TODO: Relative CAR score
        

        cav_coef = self.model.coef_

        score = np.mean(test_data @ cav_coef.T > 0)

        score = (score - 0.5) * 2

        # Perform relu on the score
        score = max(0, score)

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
        concept2,
    ):
        print("Concept2:", concept2)
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

        concept_score = {concept2: accuracy}

        self.cav_list.append(concept_score)
        return cav

    def calculate_rel_cav(
        self,
        concept: str,
        concept2: str,
        dataset_directory_train: str,
        dataset_directory_test: str,
        model_dir: str,
        episode: int = 1,
        block: int = 0,
        save_path: str = None,
    ):
        model_list = os.listdir(model_dir)
        save_path_activations = os.path.join(save_path, "plot_data_relative_cav")
        os.makedirs(save_path_activations, exist_ok=True)

        for model in model_list:
            model_path = os.path.join(model_dir, model)
            episode_number = model.split("_")[-1].split(".")[0]

            if int(episode_number) != episode:
                continue

            if block == 0:
                embedding = True
            else:
                embedding = False

            (
                positive,
                negative,
                positive_test,
                negative_test,
                _,
                _,
                _,
                _,
            ) = self.read_dataset(
                concept=concept,
                concept2=concept2,
                dataset_directory_train=dataset_directory_train,
                dataset_directory_test=dataset_directory_test,
                model_path=model_path,
                block=block,
                embedding=embedding,
                save_path=save_path,
                episode=episode,
            )

            cav = self.calculate_single_cav(
                block,
                episode_number,
                positive,
                negative,
                positive_test,
                negative_test,
                save_path,
                concept,
                concept2
            )
           
        # Save the CAV list
        torch.save(self.cav_list, os.path.join(save_path_activations, f"{concept}.pt"))
        print(
            "CAV list saved to:", os.path.join(save_path_activations, f"{concept}.pt")
        )
        # torch.save(self.tcav_list, f"./results/tcav/tcav_list_{concept}.pt")

   

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
