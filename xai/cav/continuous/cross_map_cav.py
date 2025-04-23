import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
ppo_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../agent/ppo")
)
sys.path.append(project_root)
sys.path.append(ppo_path)

import multiprocessing
import pickle
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import torch
import umap
from cav import create_activation_dataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pandas as pd

from utils import build_numpy_list_cav


class Cross_Map_CAV:

    def __init__(self, config):
        self.source_map: str = config["source_map"]
        self.target_map: str = config["target_map"]
        self.model_name = config["model_name"]

        self.dataset_path = config["dataset_path"]
        self.results_path = config["results_path"]
        self.target_path = config["target_path"]
        self.model_path = config["model_path"]
        self.cav_model = config["cav_model"]
        self.target_cav_model = config["target_cav_model"]
        self.cos_sim = config["cos_sim"]
        self.car = config["car"]

        self.embedding = config["embedding"]
        self.block = config["block"]
        if self.block == 0:
            self.embedding = True
        self.episode = config["episode"]
        self.grid_length: int = config["grid_length"]

        # TCAV stuff
        self.sensitivity = config["sensitivity"]
        self.action_index = config["action_index"]

    def load_cav_model(self, model_path) -> LogisticRegression | SVC:
        try:
            model = pickle.load(open(model_path, "rb"))
            return model
        except FileNotFoundError:
            return None

    def read_test_dataset(self, concept):
        # See if activation dataset exists
        if os.path.exists(
            f"{self.target_path}/activations/test/{concept}/episode_{self.episode}/activation_{concept}_block_{self.block}.pt"
        ):
            positive_test = torch.load(
                f"{self.target_path}/activations/test/{concept}/episode_{self.episode}/activation_{concept}_block_{self.block}.pt"
            )
        else:
            print("Did not find actication dataset: ", f"{self.target_path}/activations/test/{concept}/episode_{self.episode}/activation_{concept}_block_{self.block}.pt")
            positive_test, _ = create_activation_dataset(
                dataset_path=f"{self.dataset_path}/{concept}_positive_test.csv",
                model_path=self.model_path,
                block=self.block - 1,
                embedding=self.embedding,
                requires_grad=self.sensitivity,
                action_index=self.action_index,
            )
        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

    
        print(f"Positive test shape: {positive_test.shape}")

        positive_test = build_numpy_list_cav(positive_test)


        test_dataset= shuffle(positive_test, random_state=42)

        return test_dataset
    

    def build_cav_list(self):
        all_cavs = self.cav_model.split("/")[:-2]
        all_cavs = "/".join(all_cavs) + "/"
        cavs = []

        for i in range(len(os.listdir(all_cavs))):
            concept = f"grid_observations_{i}"
            cav_path = os.path.join(
                all_cavs,
                f"{concept}/{concept}_block_{self.block}_episode_{self.episode}.pkl",
            )
            cav_model = self.load_cav_model(cav_path)
            if cav_model is None:
                continue
            cav_coef = cav_model.coef_
            cavs.append(cav_coef[0])
            # print(cav_model.coef_)

        for i in range(len(os.listdir(self.target_cav_model)) - 1):
            concept = f"grid_observations_{i}"
            target_cav_path = os.path.join(
                self.target_cav_model,
                f"{concept}/{concept}_block_{self.block}_episode_{self.episode}.pkl",
            )
            target_cav_model = self.load_cav_model(target_cav_path)
            if target_cav_model is None:
                continue

            cavs.append(target_cav_model.coef_[0])

        cavs = np.array(cavs)

        return cavs


    def test_cav(self, concept):
        test_data = self.read_test_dataset(concept)
        if len(test_data) == 0:
            print(f"No test dataset for {concept}")
            return 0
        cav_model = self.load_cav_model(self.cav_model)
        if cav_model is None:
            return 0
        if self.car:
            accuracy = accuracy_score(cav_model.predict(test_data), np.ones(len(test_data)))
        else:
            cav_coef = cav_model.coef_
            accuracy = np.mean(test_data @ cav_coef.T > 0)
        return max(0, (accuracy - 0.5) * 2)

    def train_reduction(self):

        cavs = self.build_cav_list()

        pca = PCA(
            48
        )  # TODO: Some statistical analysis to determine the number of components coherence score??
        pca.fit(cavs)


        print(f"Number of components: {pca.n_components_}")

        return pca

    def projection_similarity(self, cav1, cav2):
        return np.dot(cav1, cav2) / np.linalg.norm(cav2)
    
    def euclidean_distance(self,cav1, cav2):
        return np.linalg.norm(cav1 - cav2)

    def cosine_similarity(self, concept, pca = None):
        # Calculate cosine similarity between the CAVs

        # Load the cav_models for each grid
        souce_cav_model = self.load_cav_model(self.cav_model)
        if souce_cav_model is None:
            return 0

        target_cav_path = os.path.join(
            self.target_cav_model,
            f"{concept}/{concept}_block_{self.block}_episode_{self.episode}.pkl",
        )
        target_cav_model = self.load_cav_model(target_cav_path)
        if target_cav_model is None:
            return 0

        # Calculate cosine similarity
        cav_coef = souce_cav_model.coef_
        target_cav_coef = target_cav_model.coef_
        # print(f"CAV coef shape: {cav_coef}")

        # PCA and UMAP
        if pca is not None:
            cav_coef = pca.transform(cav_coef)
            target_cav_coef = pca.transform(target_cav_coef)

        

        #print(f"CAV coef shape: {cav_coef}")
        #print(f"Target CAV coef shape: {target_cav_coef}")

        #reducer = umap.UMAP(n_neighbors=15, min_dist=5, n_components=3, random_state=42)
        #cav_coef = reducer.fit_transform(cav_coef)
        #target_cav_coef = reducer.fit_transform(target_cav_coef)
        #print(f"CAV coef shape: {cav_coef}")
        #cav_coef = cav_coef.flatten()
        #target_cav_coef = target_cav_coef.flatten()

        cos_sim = cosine_similarity(cav_coef, target_cav_coef)
        #cos_sim = self.projection_similarity(cav_coef, target_cav_coef)
        #cos_sim = np.corrcoef(cav_coef, target_cav_coef)[0, 1]
        return cos_sim[0][0]

    

    def test_grids(self):

        # Change the number of grids to the number of grid observations
        num_grids = self.grid_length * self.grid_length

        if self.target_map.__contains__("horizontally") or self.target_map.__contains__(
            "vertically"
        ):
            num_grids = num_grids * 2

        accuracy_grid = {}
        if self.cos_sim:
            pca = None
            pca = self.train_reduction()
        for i in range(num_grids):
            concept = f"grid_observations_{i}"
            if self.cos_sim:
                accuracy = self.cosine_similarity(concept, pca)

            else:
                accuracy = self.test_cav(concept)
            print(f"Grid {i}: {accuracy}")
            accuracy_grid[f"grid_{i}"] = accuracy

        return accuracy_grid
    
    def save_matrix(self, accuracy_grid, grid_number: int, coordinate: tuple = (-1, -1)):
        """
        Saves a matrix representation of accuracy scores to a CSV file.
        This method processes a grid of accuracy scores, organizes them into a 
        2D matrix, and saves the resulting matrix to a CSV file. The file is 
        stored in a directory structure based on the model name, grid length, 
        source map, and target map.
        Args:
            accuracy_grid (dict): A dictionary containing accuracy scores for 
                each grid cell, where keys are in the format "grid_<index>".
            grid_number (int): The grid number being processed.
            coordinate (tuple, optional): The coordinate of the grid. Defaults 
                to (-1, -1).
        Attributes:
            grid_length (int): The base length of the grid.
            target_map (str): The target map, which determines if the grid 
                dimensions should be adjusted horizontally or vertically.
            model_name (str): The name of the model being used.
            source_map (str): The source map identifier.
            block (int): The block number in the current context.
            episode (int): The episode number in the current context.
        Saves:
            A CSV file containing the 2D matrix of accuracy scores. The file is 
            saved in the directory:
            `remapping/vectors/{model_name}/grid_length_{grid_length}/remapping_src_{source_map}_target_{target_map}/`
            with the filename:
            `grid_observations_{coordinate}_block_{block}_episode_{episode}.csv`.
        """
        
        grid_length = self.grid_length
        grid_length_horizontal = grid_length
        grid_length_vertical = grid_length
        if self.target_map.__contains__("horizontally"):
            grid_length_horizontal = grid_length * 2
        if self.target_map.__contains__("vertically"):
            grid_length_vertical = grid_length * 2

        
        if self.target_map.__contains__("horizontally"):
            # Transform the square to map to the horizontal map
            target_coordinate = (coordinate[0], coordinate[1] * 2)
        elif self.target_map.__contains__("vertically"):
            # Transform the square to map to the vertical map
            target_coordinate = (coordinate[1]*2, coordinate[0])
        elif self.target_map.__contains__("rot90") and self.model_name == "helpful-bush-1369":
            # Transform the square to map to the rotated map
            target_coordinate = (coordinate[1], -coordinate[0] + grid_length-1)


        scores = []
        for i in range(grid_length_vertical):
            row = []
            for j in range(grid_length_horizontal):
                row.append(accuracy_grid[f"grid_{i * grid_length_horizontal + j}"])
            scores.append(row)

        # Save the matrix to a file
        df = pd.DataFrame(scores)
        save_path = f"remapping/vectors/{self.model_name}/grid_length_{grid_length}/remapping_src_{self.source_map}_target_{self.target_map}/"
        if self.cos_sim:
            save_path += "cosine_sim/"
        os.makedirs(save_path, exist_ok=True)
        df.to_csv(os.path.join(save_path, f"grid_observations_{grid_number}_{coordinate[0]}_{coordinate[1]}_block_{self.block}_episode_{self.episode}_target_{target_coordinate[0]}_{target_coordinate[1]}.csv"), index=False)



    def visualize_scores(
        self, accuracy_grid, grid_number: int, coordinate: tuple = (-1, -1)
    ):

        grid_length = self.grid_length
        grid_length_horizontal = grid_length
        grid_length_vertical = grid_length
        if self.target_map.__contains__("horizontally"):
            grid_length_horizontal = grid_length * 2
        if self.target_map.__contains__("vertically"):
            grid_length_vertical = grid_length * 2

        scores = []
        for i in range(grid_length_vertical):
            row = []
            for j in range(grid_length_horizontal):
                row.append(accuracy_grid[f"grid_{i * grid_length_horizontal + j}"])
            scores.append(row)


        # Plot

        if self.target_map.__contains__("horizontally"):
            fig, ax = plt.subplots(figsize=(20, 10))
            # Transform the square to map to the horizontal map
            coordinate = (coordinate[0], coordinate[1] * 2)
        elif self.target_map.__contains__("vertically"):
            fig, ax = plt.subplots(figsize=(10, 20))
            # Transform the square to map to the vertical map
            coordinate = (coordinate[1]*2, coordinate[0])
        elif self.target_map.__contains__("rot90") and self.model_name == "helpful-bush-1369":
            fig, ax = plt.subplots(figsize=(10, 10))
            # Transform the square to map to the rotated map
            coordinate = (coordinate[1], -coordinate[0] + grid_length-1)

        else:
            fig, ax = plt.subplots(figsize=(10, 10))

        sns.heatmap(
            scores, annot=True, ax=ax, vmin=0, vmax=1
        )

        block = self.block

        if self.cos_sim:
            ax.set_title(
                f"Cosine similarity of CAVs for each grid observation for grid {grid_number}, {coordinate}"
            )
            save_path = f"remapping/heatmaps/{self.model_name}/grid_length_{grid_length}/remapping_src_{self.source_map}_target_{self.target_map}/cosine_similarity/"
        else:
            ax.set_title(
                f"Accuracy of CAVs for each grid observation for grid {grid_number}, {coordinate}"
            )
            save_path = f"remapping/heatmaps/{self.model_name}/grid_length_{grid_length}/remapping_src_{self.source_map}_target_{self.target_map}"
        rect = patches.Rectangle(
            (coordinate[1], coordinate[0]), 1, 1, linewidth=2, edgecolor="red", facecolor="none"
        )
        ax.add_patch(rect)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            f"{save_path}/grid_observations_{grid_number}_block_{block}_episode_{self.episode}.png"
        )
        # plt.show()


def worker(
    grid_number,
    source_map,
    target_map,
    model_name,
    grid_length,
    block,
    episode,
    embedding,
    car=False,
):
    cav_path = ""
    if car:
        cav_path = "_car"

        

    for i in grid_number:

        

        config = {
            "source_map": source_map,
            "target_map": target_map,
            "model_name": model_name,
            "dataset_path": f"./dataset/{model_name}/{target_map}/grid_length_{grid_length}/test",  # TODO: Change which dataset grid to use
            "results_path": f"./results/{model_name}/{source_map}/grid_length_{grid_length}",
            "target_path": f"./results/{model_name}/{target_map}/grid_length_{grid_length}",
            "model_path": f"../../../agent/ppo/models/transformers/{model_name}/actor/policy_network_{episode}.pth",
            "cav_model": f"./results/{model_name}/{source_map}/grid_length_{grid_length}/models{cav_path}/grid_observations_{i}/grid_observations_{i}_block_{block}_episode_{episode}.pkl",
            "target_cav_model": f"./results/{model_name}/{target_map}/grid_length_{grid_length}/models{cav_path}/",  # used for cosine similarity
            "cos_sim": False,
            "car": car,
            "grid_length": grid_length,
            "block": block,
            "episode": episode,
            "embedding": embedding,
            # TCAV stuff
            "sensitivity": False,
            "action_index": 0,
        }
        if car:
            config["cos_sim"] = False

        cav = Cross_Map_CAV(config)
        # Check if the plot exists
        accuracy_grid = cav.test_grids()
        grid_coordinate = (i // grid_length, i % grid_length)

        cav.save_matrix(
            accuracy_grid, grid_number=i, coordinate=grid_coordinate
        )

        cav.visualize_scores(accuracy_grid, i, grid_coordinate)


def main():

    source_map = "map_two_rooms_18_19"
    target_map = "map_two_rooms_rot90_19_2"
    model_name = "helpful-bush-1369"

    grid_length = 6

    block = 2
    episode = 1700

    car = False

    embedding = False
    if block == 0:
        embedding = True

    grids = np.arange(grid_length * grid_length)
    grids_per_worker = np.array_split(grids, multiprocessing.cpu_count() - 2)

    # print(grids_per_worker)

    q = Queue()
    processes = []
    for i in range(multiprocessing.cpu_count() - 2):
        p = Process(
            target=worker,
            args=(
                grids_per_worker[i],
                source_map,
                target_map,
                model_name,
                grid_length,
                block,
                episode,
                embedding,
                car
            ),
        )
        p.start()
        processes.append(p)

    # accuracy_grid = cav.test_grids()
    # cav.visualize_scores(accuracy_grid, grid_number)




if __name__ == "__main__":

    # analysis()

    main()
