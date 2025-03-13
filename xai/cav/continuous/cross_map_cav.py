import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

import torch
import pickle
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from cav import create_activation_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle


from utils import build_numpy_list_cav

class Cross_Map_CAV:

    def __init__(self, config):
        self.source_map = config["source_map"]
        self.target_map = config["target_map"]

        self.dataset_path = config["dataset_path"]
        self.model_path = config["model_path"]
        self.cav_model = config["cav_model"]


        self.embedding = False
        self.block = config["block"]
        if self.block == 0:
            self.embedding = True
        self.episode = config["episode"]
        self.grid_length = config["grid_length"]
        

        # TCAV stuff
        self.sensitivity = config["sensitivity"]
        self.action_index = config["action_index"]

    def load_cav_model(self) -> LogisticRegression:
        model = pickle.load(open(self.cav_model, "rb"))
        return model

    def read_test_dataset(self, concept):
        positive_test, _ = create_activation_dataset(
            dataset_path=f"{self.dataset_path}/{concept}_positive_test.csv",
            model_path=self.model_path,
            block=self.block,
            embedding=self.embedding,
            requires_grad=self.sensitivity,
            action_index=self.action_index,
        )
        assert isinstance(positive_test, torch.Tensor), "Positive_test must be a tensor"

        negative_test, _ = create_activation_dataset(
            dataset_path=f"{self.dataset_path}/{concept}_negative_test.csv",
            model_path=self.model_path,
            block=self.block,
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
        test_labels = np.concatenate((positive_test_labels, negative_test_labels), axis=0)

        test_dataset, test_labels = shuffle(test_dataset, test_labels, random_state=42)

        return test_dataset, test_labels
    
    def test_cav(self, concept):
        test_dataset, test_labels = self.read_test_dataset(concept)

        cav_model = self.load_cav_model()
        accuracy = cav_model.score(test_dataset, test_labels)
        return accuracy

    def test_grids(self):
        # TODO: Create activation dataset on testing dataset
        #TODO: Find number of grids
        num_grids = self.grid_length * self.grid_length
        accuracy_grid = {}
        #for i in range(num_grids - 1):
        for i in range(num_grids):
            concept = f"grid_observations_{i}"
            accuracy = self.test_cav(concept)
            print(f"Grid {i}: {accuracy}")
            accuracy_grid[f"grid_{i}"] = accuracy

        return accuracy_grid

    def visualize_scores(self,accuracy_grid, grid_number):

        grid_length = self.grid_length
        scores = []
        for i in range(grid_length):
            row = []
            for j in range(grid_length):
                row.append(accuracy_grid[f"grid_{i * grid_length + j}"])
            scores.append(row)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(scores, annot=True, ax=ax)
        ax.set_title(f"Accuracy of CAVs for each grid observation for grid {grid_number}")
        save_path = f"results/remapping_src_{self.source_map}_target_{self.target_map}/grid_length_{grid_length}/models/grid_observations_{grid_number}"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/grid_observations_{grid_number}_block_{self.block}_episode_{self.episode}.png")
        plt.show()



def main():

    source_map = "map_two_rooms_18_19"
    target_map = "map_conditional_prob_11_10"
    model_name = "kind-water-1258"
    grid_number = 17

    grid_length = 7

    block = 1
    episode = 600


    config = {
        "source_map": source_map,
        "target_map": target_map,

        "dataset_path": f"./dataset/{model_name}/{target_map}/grid_length_{grid_length}/test", # TODO: Change which dataset grid to use
        "model_path": f"../../../agent/ppo/models/transformers/{model_name}/actor/policy_network_{episode}.pth",
        "cav_model": f"./results/{model_name}/flatten/{source_map}/grid_length_{grid_length}/models/grid_observations_{grid_number}/grid_observations_{grid_number}_block_{block}_episode_{episode}.pkl",
        "grid_length": grid_length,
        "block": block,
        "episode": episode,

        # TCAV stuff
        "sensitivity": False,
        "action_index": 0,
    }

    cav = Cross_Map_CAV(config)
    accuracy_grid = cav.test_grids()
    cav.visualize_scores(accuracy_grid, grid_number)


if __name__ == "__main__":
    main()