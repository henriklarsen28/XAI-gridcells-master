import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import os
import torch



def read_csv(path: str):
    """
    Read a CSV file and return the data as a numpy array.
    """
    df = pd.read_csv(path)
    matrix = df.to_numpy()
    return matrix

def exclude_over_graph(matrix: np.ndarray, normal_dist: dict):
    

    matrix = matrix.flatten()
    # count each value
    bins = np.linspace(0, 1, 41)

    count, x_index = np.histogram(matrix, bins=bins, range=(0, 1))
    # Get the index of the values that are over the graph
    over_graph_index = np.array(
        [x_index[i] for i in range(len(count)) if count[i] > normal_dist[round(x_index[i],2)]]
    )
    if len(over_graph_index) != 0:
        print("Over graph index", over_graph_index)
        return True
    return False



def find_mean_std(matrix: np.ndarray):
    """
    Find the mean and standard deviation of the matrix.
    """
    mean = np.mean(matrix.flatten())
    std = np.std(matrix.flatten())
    return mean, std

def visualize_histogram(matrix: np.ndarray, title: str, normal_dist, peak_height: int):
    """
    Visualize the histogram of the matrix.
    """
    bins = np.linspace(0, 1, 41)
    plt.hist(matrix.flatten(), bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.xlim(0, 1)
    plt.ylim(0, peak_height)
    
    
    plt.plot(normal_dist["x"], normal_dist["y"], color="#B85450", label="Normal Distribution")
    plt.show()

def visualize_meta(cav_meta: dict):
    grid_nums = list(cav_meta.keys())
    # split the grid_num to retrieve int at the end of the string
    grid_nums = [int(grid_num.split("_")[-1]) for grid_num in grid_nums]
    means = [cav_meta[grid_num]["mean"] for grid_num in grid_nums]
    # stds = [cav_meta[grid_num]["std"] for grid_num in grid_nums]

    plt.figure(figsize=(8, 5))
    plt.bar(grid_nums, means)
    plt.xlabel('Grid Number')
    plt.ylabel('Mean Value')
    plt.title('Mean Value by Grid Number')
    plt.grid(axis='y')
    plt.show()


def read_concept_score_file(grid_num, map_name, episode, block):
    model_name = "helpful-bush-1369"
    path = f"../results/{model_name}/{map_name}/grid_length_6/plot_data/{grid_num}.pt"
    try:
        data = torch.load(path, weights_only=False)
        data = [d for d in data if d[1] == str(episode) and d[0] == block][0]
        return data[2]
    except Exception as e:
        return 0.0



def calculate_threshold(file, grid_num: str, visualize: bool = False, map_name: str = "", episode:int = 1700, block:int = 2):

    peak_height = 37  # y at x=0
    if "horizontal" in file or "vertical" in file:
        peak_height = 72 + 1
    
    epsilon = 1e-5    # very small value at x=1
    sigma = np.sqrt(-3 / (np.log(epsilon / peak_height)))
    x = np.linspace(0, 2, 500)
    normal_dist = {}
    normal_dist_dict = {
        "x": x,
        "y": peak_height * np.exp(-x**2 / (2 * sigma**2)),
    }

    for values in x:
        normal_dist[round(values,2)] = peak_height * np.exp(-values**2 / (2 * sigma**2))
    

    matrix = read_csv(file)

    # set negative values to 0
    matrix[matrix < 0] = 0

    exclude = exclude_over_graph(matrix, normal_dist)

    if visualize:
        visualize_histogram(matrix, f"Histogram of {grid_num}", normal_dist_dict, peak_height)
    
    # CAV has not high enough concept score
    score = read_concept_score_file(grid_num, map_name, episode, block)

    # exclude if the score is less than 0.5 or above the graph
    return exclude #or score < 0.5

