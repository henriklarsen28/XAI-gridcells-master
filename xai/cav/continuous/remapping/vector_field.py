import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d


# Read output
class VectorField:

    def __init__(self, grid_length: int, grid_length_horizontal: int = None):
        self.grid_length = grid_length
        self.grid_length_horizontal = grid_length_horizontal
        self.vectors = {}
        self.threshold_grids = 1  # Check the grids around the target

        """
        0 , 0 , 0.3
        0 , t , 0
        0 , 1 , 0
        """

    def add_vector(self, path: str):

        coordinate = path.split("/")[-1].split("_")[3:5]
        coordinate = (int(coordinate[0]), int(coordinate[1]))

        target_coordinate = path.split("/")[-1].split(".")[0].split("_")[-2:]
        target_coordinate = (int(target_coordinate[0]), int(target_coordinate[1]))
        print("Target", coordinate, target_coordinate)

        df = pd.read_csv(path)
        matrix = df.to_numpy()
        best_grid = self._get_best_grid(matrix, target_coordinate=target_coordinate)
        if best_grid is None:
            return
        vector, coordinate_transformed = self._create_vector(coordinate, best_grid)

        self.vectors[coordinate_transformed] = vector

    def _create_vector(self, coordinate_a, coordinate_b):
        # Create a vector from coordinate_a to coordinate_b
        coordinate_a = (
            -coordinate_a[0],
            coordinate_a[1],
        )
        coordinate_b = (
            -coordinate_b[0],
            coordinate_b[1],
        )
        vector = np.array(coordinate_b) - np.array(coordinate_a)
        return vector, coordinate_a

    def _get_best_grid(self, matrix, target_coordinate=(-1,-1)):
        
        kernel = np.ones((3, 3))
        result = convolve2d(matrix, kernel, mode="same", fillvalue=0)
        result2 = self.select_grid_close_to_target(result, target_coordinate)
        # Get the max index
        max_index = np.unravel_index(np.argmax(result2), result.shape)
        # check if the max index is above a threshold
        threshold = 0.5
        if result[max_index] < threshold:
            return None

        return tuple(max_index)

    def select_grid_close_to_target(self, matrix, target_coordinate):

        neighbours = []
        for i in range(-self.threshold_grids, self.threshold_grids + 1, 1):
            for j in range(-self.threshold_grids, self.threshold_grids + 1, 1):

                print("I,J", i, j)
                if (
                    target_coordinate[0] + i < 0
                    or target_coordinate[1] + j < 0
                    or target_coordinate[0] + i >= matrix.shape[0]
                    or target_coordinate[1] + j >= matrix.shape[1]
                ):
                    print("Out of bounds")
                    continue

                
                neighbour_coordinate = (
                    target_coordinate[0] + i,
                    target_coordinate[1] + j,
                )
                neighbours.append(neighbour_coordinate)

        neighbours = np.array(neighbours)
        # check if the neighbours are in the matrix
        max_value = [[neighbour[0], neighbour[1]] for neighbour in neighbours]
        max_value = np.array(max_value)
        #print(max_value)
        #print(max_value[:, 0])
        mask = np.zeros_like(matrix)
        mask[max_value[:, 0], max_value[:, 1]] = True

        # Ensure mask has the same shape as matrix
        mask = mask.astype(bool)
        matrix = np.where(mask, matrix, 0)
        print("Neighbour: ", neighbours)
        print(matrix)

        # max_index = np.unravel_index(np.argmax(max_value), max_value.shape)
        # loop through the neighbours and find

        return matrix

    def plot_field(self):
        # Plot the vector field

        for coordinate, vector in self.vectors.items():
            color = np.random.rand(
                3,
            )  # Generate a random color
            plt.quiver(
                coordinate[1],
                coordinate[0],
                vector[1],
                vector[0],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
            )
            plt.text(
                coordinate[1] + vector[1] / 2,
                coordinate[0] + vector[0] / 2,
                f"{coordinate}",
                color=color,
                fontsize=8,
            )

        plt.xlim(-0.5, self.grid_length_horizontal)
        plt.ylim(-self.grid_length_horizontal, 0.5)
        # Line on origin
        plt.axhline(0, color="black", lw=1)
        plt.axvline(0, color="black", lw=1)
        plt.grid()
        plt.show()

    def plot_field_centered(self):
        for coordinate, vector in self.vectors.items():
            color = np.random.rand(
                3,
            )  # Generate a random color
            plt.quiver(
                0,
                0,
                vector[1],
                vector[0],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=color,
            )
            plt.text(
                vector[1],
                vector[0],
                f"{coordinate}",
                color=color,
                fontsize=8,
            )

        plt.xlim(-self.grid_length_horizontal, self.grid_length_horizontal)
        plt.ylim(-self.grid_length_horizontal, self.grid_length_horizontal)
        # Line on origin
        plt.axhline(0, color="black", lw=1)
        plt.axvline(0, color="black", lw=1)
        plt.grid()
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
    if cosine_sim:
        path += "cosine_sim/"

    vector_field = VectorField(
        grid_length=grid_length, grid_length_horizontal=grid_length_horizontal
    )
    for file in os.listdir(path):
        file = os.path.join(path, file)
        if not file.endswith(".csv"):
            continue
        print(file)
        episode_num = int(file.split(".")[0].split("_")[-4])
        block_num = int(file.split(".")[0].split("_")[-6])
        if episode_num != episode or block_num != block:
            continue

        vector_field.add_vector(file)

    vector_field.plot_field()
    vector_field.plot_field_centered()


if __name__ == "__main__":
    main()
