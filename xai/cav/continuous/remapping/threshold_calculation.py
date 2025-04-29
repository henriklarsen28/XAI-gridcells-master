import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import os



def read_csv(path: str):
    """
    Read a CSV file and return the data as a numpy array.
    """
    df = pd.read_csv(path)
    matrix = df.to_numpy()
    return matrix

def exclude_over_graph(matrix:)
    pass


def visualize_histogram(matrix: np.ndarray, title: str, normal_dist):
    """
    Visualize the histogram of the matrix.
    """
    plt.hist(matrix.flatten(), bins=50)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.xlim(0, 1.2)
    plt.ylim(0, 36)
    # Parameters
    peak_height = 36  # y at x=0
    epsilon = 1e-5    # very small value at x=1

    # Solve for sigma
    sigma = np.sqrt(-3 / (np.log(epsilon / peak_height)))

    # Define x
    x = np.linspace(-2, 2, 500)

    # Normal distribution scaled to peak_height
    y = peak_height * np.exp(-x**2 / (2 * sigma**2))
    
    
    plt.plot(x, y, color="red", label="Normal Distribution")
    plt.show()


def main():

    model_name = "helpful-bush-1369"
    grid_length = 6
    map_name = "map_circular_4_19"
    target_map = "map_circular_rot90_19_16"
    cosine_sim = False

    grid_length_horizontal = grid_length
    if target_map.__contains__("horizontally") or target_map.__contains__("vertically"):
        grid_length_horizontal = grid_length * 2

    episode = 1700
    block = 2

    path = f"vectors/{model_name}/grid_length_{grid_length}/remapping_src_{map_name}_target_{target_map}/"

    for file in os.listdir(path):
        file_new = os.path.join(path, file)
        if not file_new.endswith(".csv"):
            continue
        print(file_new)
        episode_num = int(file_new.split(".")[0].split("_")[-4])
        block_num = int(file_new.split(".")[0].split("_")[-6])

        grid_num = file.split(".")[0].split("_")[0:3]
        grid_num = "_".join(grid_num)
        if episode_num != episode or block_num != block:
            continue
        
        matrix = read_csv(file_new)

        # Visualize the histogram of the matrix
        visualize_histogram(matrix, f"Histogram of {grid_num}")


if __name__ == "__main__":
    main()
