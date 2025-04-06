import random
import numpy as np
import matplotlib.pyplot as plt

def generate_random_grid_data(grid_size):
    # for testing
    grid_data = {f"grid_{i}": random.random() for i in range(grid_size*grid_size)}
    grid_matrix = np.array(list(grid_data.values())).reshape(grid_size, grid_size)
    print("Grid data as matrix:")
    print(grid_matrix)
    return grid_data

def get_diagonal(grid_size, grid_data):
    diagonal_indices = [i * (grid_size+1) for i in range(grid_size)]
    diagonal_values = [grid_data[f"grid_{i}"] for i in diagonal_indices]
    return diagonal_values


def get_column_excluding_diagonal(grid_size, grid_data):
    all_columns = []
    for col in range(grid_size):
        column_values = []
        for row in range(grid_size):
            idx = row * grid_size + col
            if idx != row * grid_size + row:  # exclude diagonal
                column_values.append(grid_data[f"grid_{idx}"])
        all_columns.append(column_values)
    return all_columns

def plot_std(std_diagonal, std_column):
    plt.bar(['Diagonal', 'Column'], [std_diagonal, std_column])
    plt.ylabel('Standard Deviation')
    plt.title('Standard Deviation Comparison')
    plt.show()


def summarize_grids(grid_data):
    summaries = []
    # make a summary of the grid data as a table with grid size as row and diagonal mean and std as columns
    # Get the diagonal values

    for grid_size, grid_data in grid_data.items():
        diagonal_values = get_diagonal(grid_size, grid_data)
        columns = get_column_excluding_diagonal(grid_size, grid_data)
        
        all_column_values = [val for col in columns for val in col] # flatten list

        mean_diagonal = np.mean(diagonal_values)
        std_diagonal = np.std(diagonal_values)
        
        mean_column = np.mean(all_column_values)
        std_column = np.std(all_column_values)

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
        print(f"Mean columns (excluding diag): {mean_column:.4f}, Std columns: {std_column:.4f}")

        # plot_std(std_diagonal, std_column)


def main():
    data = {}

    # Generate random grid data
    grid_sizes = [4, 5, 6, 7]  # Example grid sizes
    for grid_size in grid_sizes:
        grid_data = generate_random_grid_data(grid_size)
        data[grid_size] = grid_data

    summarize_grids(data)


if __name__ == "__main__":
    main()