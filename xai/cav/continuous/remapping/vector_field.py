import numpy as np
import pandas as pd
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

import os


# Read output
class VectorField:

    def __init__(self, grid_length: int):
        self.grid_length = grid_length
        self.vectors = {}

    def add_vector(self, path: str):

        coordinate = path.split("/")[-1].split("_")[3:5]
        coordinate = (int(coordinate[0]), int(coordinate[1]))

        df = pd.read_csv(path)
        matrix = df.to_numpy()
        best_grid = self._get_best_grid(matrix)
        vector = self._create_vector(coordinate, best_grid)

        self.vectors[coordinate] = vector

    def _create_vector(self, coordinate_a, coordinate_b):
        # Create a vector from coordinate_a to coordinate_b
        coordinate_a = (
            self.grid_length - coordinate_a[0],
            coordinate_a[1],
        )
        coordinate_b = (
            self.grid_length - coordinate_b[0],
            coordinate_b[1],
        )
        vector = np.array(coordinate_b) - np.array(coordinate_a)
        return vector

    def _get_best_grid(self, matrix):

        kernel = np.ones((2, 2))
        result = convolve2d(matrix, kernel, mode="same", fillvalue=0)

        # Get the max index
        max_index = np.unravel_index(np.argmax(result), result.shape)

        return tuple(max_index)

    def plot_field(self):
        # Plot the vector field

        for coordinate, vector in self.vectors.items():
            color = np.random.rand(3,)  # Generate a random color
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



        plt.xlim(0, self.grid_length)
        plt.ylim(-self.grid_length, 0)
        # Line on origin
        plt.axhline(0, color="black", lw=1)
        plt.axvline(0, color="black", lw=1)
        plt.grid()
        plt.show()


def main():

    model_name = "helpful-bush-1369"
    grid_length = 7
    map_name = "map_circular_4_19"
    target_map = "map_circular_rot90_19_16"

    path = f"vectors/{model_name}/grid_length_{grid_length}/remapping_src_{map_name}_target_{target_map}/"
    vector_field = VectorField(grid_length=grid_length)
    for file in os.listdir(path):
        
        file = os.path.join(path, file)
        if not file.endswith(".csv"):
            continue



        vector_field.add_vector(file)
        # vector_field.get_best_grid()
    vector_field.plot_field()


if __name__ == "__main__":
    main()
