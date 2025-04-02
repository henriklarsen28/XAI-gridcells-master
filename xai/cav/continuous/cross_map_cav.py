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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

from utils import build_numpy_list_cav


class Cross_Map_CAV:

    def __init__(self, config):
        self.source_map: str = config["source_map"]
        self.target_map: str = config["target_map"]
        self.model_name = config["model_name"]

        self.dataset_path = config["dataset_path"]
        self.results_path = config["results_path"]
        self.model_path = config["model_path"]
        self.cav_model = config["cav_model"]
        self.target_cav_model = config["target_cav_model"]
        self.cos_sim = config["cos_sim"]

        self.embedding = config["embedding"]
        self.block = config["block"]
        if self.block == 0:
            self.embedding = True
        self.episode = config["episode"]
        self.grid_length: int = config["grid_length"]

        # TCAV stuff
        self.sensitivity = config["sensitivity"]
        self.action_index = config["action_index"]

    def load_cav_model(self, model_path) -> LogisticRegression:
        try:
            model = pickle.load(open(model_path, "rb"))
            return model
        except FileNotFoundError:
            return None

    def read_test_dataset(self, concept):
        # See if activation dataset exists
        # If not, create it
        if os.path.exists(
            f"{self.results_path}/activations//{concept}_positive_test.csv"
        ):
            positive_test
        positive_test, _ = create_activation_dataset(
            dataset_path=f"{self.dataset_path}/{concept}_positive_test.csv",
            model_path=self.model_path,
            block=self.block - 1,
            embedding=self.embedding,
            requires_grad=self.sensitivity,
            action_index=self.action_index,
        )
        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

        negative_test, _ = create_activation_dataset(
            dataset_path=f"{self.dataset_path}/{concept}_negative_test.csv",
            model_path=self.model_path,
            block=self.block - 1,
            embedding=self.embedding,
            requires_grad=self.sensitivity,
            action_index=self.action_index,
        )
        assert isinstance(negative_test, torch.Tensor), "Negative_test must be a tensor"
        print(f"Positive test shape: {positive_test.shape}")

        positive_test = build_numpy_list_cav(positive_test)
        negative_test = build_numpy_list_cav(negative_test)

        positive_test_labels = np.ones(len(positive_test))
        negative_test_labels = np.zeros(len(negative_test))

        test_dataset = np.concatenate((positive_test, negative_test), axis=0)
        test_labels = np.concatenate(
            (positive_test_labels, negative_test_labels), axis=0
        )

        test_dataset, test_labels = shuffle(test_dataset, test_labels, random_state=42)

        return test_dataset, test_labels
    

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
        test_dataset, test_labels = self.read_test_dataset(concept)

        cav_model = self.load_cav_model(self.cav_model)
        accuracy = cav_model.score(test_dataset, test_labels)
        return max(0, (accuracy - 0.5) * 2)

    def train_reduction(self):

        cavs = self.build_cav_list()

        pca = PCA(
            24
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
        #cav_coef = pca.transform(cav_coef)
        #target_cav_coef = pca.transform(target_cav_coef)

        

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

    

    def analyse_pca_components(self):
        cavs = self.build_cav_list()
        variance = []
        pca_best = None
        for i in range(2, len(cavs)):
            # pca = PCA(n_components=i)
            # pca.fit(cavs)
            # explained_variance = pca.explained_variance_ratio_.sum()
            # X_reduced = PCA(n_components=i).fit_transform(cavs)
            X_umap = umap.UMAP(n_components=i, random_state=42).fit_transform(cavs)

            # Compute local variance preservation (approximation)
            variance_preserved = np.mean(np.var(X_umap, axis=0))  # Simplified metric
            # variance.append(variance_preserved)

            # print(f"Number of components: {i}, Explained variance: {explained_variance}")
            dicti = {
                "num_components": i,
                "explained_variance": variance_preserved,
            }
            variance.append(dicti)
            """if explained_variance > 0.95:
                pca_best = pca
                break"""

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        x = [i["num_components"] for i in variance]
        y = [i["explained_variance"] for i in variance]
        ax.plot(x, y)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Explained variance")
        ax.set_title("PCA explained variance")
        ax.grid()
        # save_path = f"results/{self.model_name}/remapping_src_{self.source_map}_target_{self.target_map}/grid_length_{self.grid_length}/cosine_similarity/"

        plt.show()

    def test_grids(self):

        # Change the number of grids to the number of grid observations
        num_grids = self.grid_length * self.grid_length

        if self.target_map.__contains__("horizontally") or self.target_map.__contains__(
            "vertically"
        ):
            num_grids = num_grids * 2

        accuracy_grid = {}
        pca = None
        #pca = self.train_reduction()
        # for i in range(num_grids - 1):
        for i in range(num_grids):
            concept = f"grid_observations_{i}"
            if self.cos_sim:
                accuracy = self.cosine_similarity(concept, pca)

            else:
                accuracy = self.test_cav(concept)
            print(f"Grid {i}: {accuracy}")
            accuracy_grid[f"grid_{i}"] = accuracy

        return accuracy_grid

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

        else:
            fig, ax = plt.subplots(figsize=(10, 10))

        sns.heatmap(
            scores, annot=True, ax=ax, vmin=0, vmax=1
        )  # TODO: Color map is from 0 to 1
        block = self.block

        if self.cos_sim:
            ax.set_title(
                f"Cosine similarity of CAVs for each grid observation for grid {grid_number}, {coordinate}"
            )
            save_path = f"results/{self.model_name}/remapping_src_{self.source_map}_target_{self.target_map}/grid_length_{grid_length}/cosine_similarity/"
        else:
            ax.set_title(
                f"Accuracy of CAVs for each grid observation for grid {grid_number}, {coordinate}"
            )
            save_path = f"results/{self.model_name}/remapping_src_{self.source_map}_target_{self.target_map}/grid_length_{grid_length}/"
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
):

    for i in grid_number:

        

        config = {
            "source_map": source_map,
            "target_map": target_map,
            "model_name": model_name,
            "dataset_path": f"./dataset/{model_name}/{target_map}/grid_length_{grid_length}/test",  # TODO: Change which dataset grid to use
            "results_path": f"./results/{model_name}/{source_map}/grid_length_{grid_length}/",
            "model_path": f"../../../agent/ppo/models/transformers/{model_name}/actor/policy_network_{episode}.pth",
            "cav_model": f"./results/{model_name}/{source_map}/grid_length_{grid_length}/models/grid_observations_{i}/grid_observations_{i}_block_{block}_episode_{episode}.pkl",
            "target_cav_model": f"./results/{model_name}/{target_map}/grid_length_{grid_length}/models/",  # used for cosine similarity
            "cos_sim": True,
            "grid_length": grid_length,
            "block": block,
            "episode": episode,
            "embedding": embedding,
            # TCAV stuff
            "sensitivity": False,
            "action_index": 0,
        }
        if not config["cos_sim"]:
            grid_path = f"results/{model_name}/remapping_src_{source_map}_target_{target_map}/grid_length_{grid_length}/grid_observations_{i}_block_{block}_episode_{episode}.png"
            if os.path.exists(grid_path):
                print("Skipping grid", i)
                continue
        cav = Cross_Map_CAV(config)
        # Check if the plot exists
        accuracy_grid = cav.test_grids()
        grid_coordinate = (i // grid_length, i % grid_length)
        cav.visualize_scores(accuracy_grid, i, grid_coordinate)


def main():

    source_map = "map_circular_4_19"
    target_map = "map_circular_horizontally_4_40"
    model_name = "feasible-lake-1351"
    # grid_number = 17

    grid_length = 7

    block = 1
    episode = 1000

    embedding = False
    if block == 0:
        embedding = True

    """config = {
        "source_map": source_map,
        "target_map": target_map,
        "model_name": model_name,

        "dataset_path": f"./dataset/{model_name}/{target_map}/grid_length_{grid_length}/test", # TODO: Change which dataset grid to use
        "model_path": f"../../../agent/ppo/models/transformers/{model_name}/actor/policy_network_{episode}.pth",
        "cav_model": f"./results/{model_name}/flatten/{source_map}/grid_length_{grid_length}/models/grid_observations_{grid_number}/grid_observations_{grid_number}_block_{block}_episode_{episode}.pkl",
        "grid_length": grid_length,
        "block": block,
        "episode": episode,

        # TCAV stuff
        "sensitivity": False,
        "action_index": 0,
    }"""

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
            ),
        )
        p.start()
        processes.append(p)

    # accuracy_grid = cav.test_grids()
    # cav.visualize_scores(accuracy_grid, grid_number)


def analysis():

    source_map = "map_two_rooms_18_19"
    target_map = "map_two_rooms_vertically_36_19"

    model_name = "azure-sun-1341"
    # grid_number = 17

    grid_length = 7

    block = 1
    episode = 500
    config = {
        "source_map": source_map,
        "target_map": target_map,
        "model_name": model_name,
        "dataset_path": f"./dataset/{model_name}/{target_map}/grid_length_{grid_length}/test",  # TODO: Change which dataset grid to use
        "model_path": f"../../../agent/ppo/models/transformers/{model_name}/actor/policy_network_{episode}.pth",
        "results_path": f"./results/{model_name}/{source_map}/grid_length_{grid_length}/",
        "cav_model": f"./results/{model_name}/{source_map}/grid_length_{grid_length}/models/grid_observations_0/grid_observations_0_block_{block}_episode_{episode}.pkl",
        "target_cav_model": f"./results/{model_name}/{target_map}/grid_length_{grid_length}/models/",  # used for cosine similarity
        "cos_sim": True,
        "grid_length": grid_length,
        "block": block,
        "episode": episode,
        # TCAV stuff
        "sensitivity": False,
        "action_index": 0,
    }
    cav = Cross_Map_CAV(config)
    cav.analyse_pca_components()


if __name__ == "__main__":

    # analysis()

    main()
