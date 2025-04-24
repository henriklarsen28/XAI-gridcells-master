import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


import re

# Example: your_dict = {...}  # <- the dictionary you provided
# For the sake of clarity, here we assume your_dict is already defined.



def generate_random_grid_data(grid_size):
    # for testing
    grid_data = {
        f"grid_observations_{i}": random.random() for i in range(grid_size * grid_size)
    }
    grid_matrix = np.array(list(grid_data.values())).reshape(grid_size, grid_size)
    print("Grid data as matrix:")
    print(grid_data)
    return grid_data


def get_diagonal(grid_size, grid_data):
    diagonal_indices = [f"{i}_{i}" for i in range(grid_size**2)]

    diagonal_values = [grid_data[f"grid_observations_{i}"] for i in diagonal_indices]
    return diagonal_values


def get_column_excluding_diagonal(grid_size, grid_data):
    all_columns = []
    for col in range(grid_size**2):
        column_values = []
        for row in range(grid_size**2):
            idx = f"{row}_{col}"
            #idx = row * grid_size + col
            if row != col:  # exclude diagonal
                column_values.append(grid_data[f"grid_observations_{idx}"])
        #print(np.mean([v for v in column_values if v is not None]))
        all_columns.append(column_values)
    return all_columns


def heat_map(grid_data, grid_size):
    grid_matrix = np.array(list(grid_data.values())).reshape(grid_size**2, grid_size**2)

    i_lower = np.tril_indices_from(grid_matrix, k=1)
    grid_matrix[i_lower[1], i_lower[0]] = grid_matrix[i_lower[0], i_lower[1]]

    plt.imshow(grid_matrix, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Grid Data Heat Map")
    plt.show()


def plot_std(std_diagonal, std_column):
    plt.bar(["Diagonal", "Column"], [std_diagonal, std_column])
    plt.ylabel("Standard Deviation")
    plt.title("Standard Deviation Comparison")
    plt.show()


def summarize_grids(grid_data):
    summaries = []
    # make a summary of the grid data as a table with grid size as row and diagonal mean and std as columns
    # Get the diagonal values

    for grid_size, grid_data in grid_data.items():
        diagonal_values = get_diagonal(grid_size, grid_data)
        columns = get_column_excluding_diagonal(grid_size, grid_data)

        all_column_values = [val for col in columns for val in col]  # flatten list

        mean_diagonal = np.mean([v for v in diagonal_values if v is not None])
        std_diagonal = np.std([v for v in diagonal_values if v is not None])

        mean_column = np.mean([cv for cv in all_column_values if cv is not None])
        std_column = np.std([cv for cv in all_column_values if cv is not None])

        summary = {
            "grid_size": grid_size,
            "mean_diagonal_values": mean_diagonal,
            "std_diagonal_values": std_diagonal,
            "mean_column_values": mean_column,
            "std_column_values": std_column
        }

        summaries.append(summary)

        print(f"\nGrid Size: {grid_size} x {grid_size}")
        print(f"Mean diagonal: {mean_diagonal:.4f}, Std diagonal: {std_diagonal:.4f}")
        print(
            f"Mean columns (excluding diag): {mean_column:.4f}, Std columns: {std_column:.4f}"
        )

        #plot_std(std_diagonal, std_column)
        heat_map(grid_data, grid_size)


def read_cav_list(dir_path, grid_length=5):

    files = os.listdir(dir_path)
    cav_dicti = {}

    for file in files:
        if not file.endswith(".pt"):
            continue
        file_path = os.path.join(dir_path, file)
        cav_data = torch.load(file_path)
        # Merge or accumulate data for this file
        for outer_key, value_list in cav_data.items():
            if outer_key not in cav_dicti:
                cav_dicti[outer_key] = {}
            
            # Flatten the inner dictionary
            for inner_dict in value_list:
                for k, v in inner_dict.items():                    
                    cav_dicti[outer_key][k] = v

    
    # Fill in the missing outer dictionary keys with None
    for i in range(grid_length**2):
        if f"grid_observations_{i}" not in cav_dicti:
            cav_dicti[f"grid_observations_{i}"] = {}


    # Fill in the missing keys with None

    for key in cav_dicti.keys():
        for i in range(grid_length**2):
            if f"grid_observations_{i}" not in cav_dicti[key]:
                cav_dicti[key][f"grid_observations_{i}"] = 0


    # Flatten the dictionary with new keys without nesting
    flattened_dict = {}
    for outer_key, inner_dict in cav_dicti.items():
        for inner_key, value in inner_dict.items():
            in_key = inner_key.split("_")[2]
            new_key = f"{outer_key}_{in_key}"
            flattened_dict[new_key] = value

    def extract_AB(key):
        # Extract two integers A and B from keys like "grid_observations_A_B"
        matches = re.findall(r'(\d+)', key)
        if len(matches) >= 2:
            return int(matches[-2]), int(matches[-1])
        else:
            return float('inf'), float('inf')  # fallback for unexpected format

    # Sort the items based on A, then B
    sorted_items = sorted(flattened_dict.items(), key=lambda item: extract_AB(item[0]))

    # Optionally convert back to a dictionary
    sorted_dict = dict(sorted_items)

    # To print or use
    #for k, v in sorted_dict.items():
    #    print(f"{k}: {v}")

    return sorted_dict

def box_and_whisker_plot():
    data = {"5x5":	[0.1950,0.2224,	0.2359,	0.2091],
            "6x6":	[0.1227,0.2275,	0.2234,	0.2156],
            "7x7":	[0.1455,0.2926,	0.2658,	0.3011]
            }
    

    fig, ax = plt.subplots()
    means = [np.mean(v) for v in data.values()]
    stds = [np.std(v) for v in data.values()]
    ax.bar(data.keys(), means, yerr=stds, capsize=5)
    ax.errorbar(data.keys(), means, yerr=stds, fmt='o', color='black', capsize=5)
    ax.set_title("Box and Whisker Plot")
    ax.set_ylabel("Values")
    ax.set_xlabel("Grid Size")
    plt.show()


def main():
    data = {}
    #grid_size = 5

    #data[grid_size] = read_cav_list(dir_path, grid_length=grid_size)
    # Generate random grid data
    grid_sizes = [5,6,7]  # Example grid sizes
    for grid_size in grid_sizes:
        #grid_data = generate_random_grid_data(grid_size)
        dir_path = f"./results/helpful-bush-1369/map_circular_4_19/grid_length_{grid_size}/cav_list/"

        data[grid_size] = read_cav_list(dir_path, grid_length=grid_size)

    summarize_grids(data)


if __name__ == "__main__":
    #main()
    box_and_whisker_plot()

