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

def exclude_over_graph(matrix: np.ndarray, grid_num:str, normal_dist: dict):
    

    matrix = matrix.flatten()
    # count each value
    count, x_index = np.histogram(matrix, bins=20, range=(0, 1))

    # Get the index of the values that are over the graph
    over_graph_index = np.array(
        [x_index[i] for i in range(len(count)) if count[i] > normal_dist[round(x_index[i],2)]]
    )

    print("Over graph index", over_graph_index)

    if len(over_graph_index) != 0:
        return True
    return False



def find_mean_std(matrix: np.ndarray):
    """
    Find the mean and standard deviation of the matrix.
    """
    mean = np.mean(matrix.flatten())
    std = np.std(matrix.flatten())
    return mean, std

def visualize_histogram(matrix: np.ndarray, title: str, normal_dist):
    """
    Visualize the histogram of the matrix.
    """
    plt.hist(matrix.flatten(), bins=20)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.xlim(0, 1.2)
    plt.ylim(0, 36)
    # Parameters
    
    
    plt.plot(normal_dist["x"], normal_dist["y"], color="red", label="Normal Distribution")
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

def main():

    model_name = "helpful-bush-1369"
    grid_length = 6
    map_name = "map_two_rooms_18_19"

    target_map = "map_two_rooms_18_19"
    cosine_sim = True


    peak_height = 36  # y at x=0
    epsilon = 1e-5    # very small value at x=1
    sigma = np.sqrt(-3 / (np.log(epsilon / peak_height)))
    x = np.linspace(0, 2, 500)
    normal_dist = {}
    normal_dist_dict = {
        "x": x,
        "y": peak_height * np.exp(-x**2 / (2 * sigma**2)),
    }

    for values in x:
        print(values)
        normal_dist[round(values,2)] = peak_height * np.exp(-values**2 / (2 * sigma**2))
    
    print("Normal distribution", normal_dist)
    grid_length_horizontal = grid_length
    if target_map.__contains__("horizontally") or target_map.__contains__("vertically"):
        grid_length_horizontal = grid_length * 2

    episode = 1700
    block = 2

    path = f"vectors/{model_name}/grid_length_{grid_length}/remapping_src_{map_name}_target_{target_map}/"

    cav_meta = {}
    excluded_grid_nums = []
    if cosine_sim:
        path += "cosine_sim/"

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
        #visualize_histogram(matrix, f"Histogram of {grid_num}")
        """mean, std = find_mean_std(matrix)
        cav_meta[grid_num] = {
            "mean": mean
        }"""
        # Exclude values over the graph
        exclude = exclude_over_graph(matrix, grid_num, normal_dist)

        if exclude:
            excluded_grid_nums.append(grid_num)
            print(f"Excluded grid number: {grid_num}")
            continue

    # Save the list of excluded grid numbers to a CSV file
    if cosine_sim:
        file_name = f"excluded_grid_nums_episode_{episode}_block_{block}_cosine_sim.csv"
        
    else:
        file_name = f"excluded_grid_nums_episode_{episode}_block_{block}.csv"

    save_path = os.path.join(path,"excluded_grid_nums/")
    os.makedirs(save_path, exist_ok=True)

    excluded_grid_nums_path = os.path.join(save_path, file_name)

    excluded_grid_nums_df = pd.DataFrame(excluded_grid_nums, columns=["Excluded Grid Numbers"])
    excluded_grid_nums_df.to_csv(excluded_grid_nums_path, index=False)
    #visualize_meta(cav_meta)

if __name__ == "__main__":
    main()
