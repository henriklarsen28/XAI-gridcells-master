import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import os




def read_concept_scores(path: str):
    
    data = torch.load(path, weights_only=False)
    
    return data



def process_concept_scores(data, episode):
    
    # Split into blocks and sort out episodes that are not in the range
    data = [d for d in data if d[1] == str(episode)]

    block0 = [d[2] for d in data if d[0] == 0]
    block1 = [d[2] for d in data if d[0] == 1]
    block2 = [d[2] for d in data if d[0] == 2]

    return block0, block1, block2


def get_concept_scores(path, episode):

    dicti = {"block0": [], "block1": [], "block2": []}
    for file in os.listdir(path):
        if not file.endswith(".pt"):
            continue
        data = read_concept_scores(os.path.join(path, file))
        block0, block1, block2 = process_concept_scores(data, episode)

        

        dicti["block0"].append(block0)
        dicti["block1"].append(block1)
        dicti["block2"].append(block2)
    return dicti


def plot_concept_scores(scores_cav:dict, scores_car:dict, episode:int):

    fig, ax = plt.subplots(figsize=(10, 5))

    width = 0.35
    x = np.arange(len(scores_cav))

    y = [np.mean(data) for data in scores_cav.values()]
    stds = [np.std(data) for data in scores_cav.values()]
    ax.bar(x-width/2, y, yerr=stds, capsize=5, width=width, color='blue', label="CAV")

    y = [np.mean(data) for data in scores_car.values()]
    stds = [np.std(data) for data in scores_car.values()]
    ax.bar(x+width/2, y, yerr=stds, capsize=5,width=width, color='orange', label="CAR")


    plt.tight_layout()

    plt.ylabel("")
    plt.title("Mean Concept Scores CAV and CAR")
    plt.ylabel("Concept Score")
    ax.set_xticks(x)
    ax.set_xticklabels(scores_cav.keys())
    plt.legend()
    plt.show()





def main():

    map_name = "map_two_rooms_18_19"

    path_cav = f"helpful-bush-1369/{map_name}/grid_length_6/plot_data/"

    dicti_cav = get_concept_scores(path_cav, 1700)

    path_car = f"helpful-bush-1369/{map_name}/grid_length_6/plot_data/"
    dicti_car = get_concept_scores(path_car, 1700)

    plot_concept_scores(dicti_cav, dicti_car, 1700)

if __name__ == "__main__":
    main()




