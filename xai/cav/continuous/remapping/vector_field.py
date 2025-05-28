import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from threshold_calculation import calculate_threshold


# Read output
class VectorField:

    def __init__(
        self, grid_length: int, grid_length_horizontal: int = None, target: bool = False, name: str = "", cosine_sim: bool = False, car: bool = False, exclude: bool = False, episode: int = 0, block: int = 0
    ):
        self.grid_length = grid_length
        self.grid_length_horizontal = grid_length_horizontal
        self.vectors = {}
        self.threshold_grids_vetical = 1  # Check the grids around the target
        self.threshold_grids_horizontal = 1
        self.target = target

        self.kernel_size = (2,2)
        self.threshold = 0.5

        self.cosine_sim=cosine_sim
        self.car=car
        self.exclude=exclude
        self.name = name
        self.episode = episode
        self.block = block

        """
        0 , 0 , 0.3
        0 , t , 0
        0 , 1 , 0
        """

    def random_color_generator(self):
        # Generate a random color
        color = np.random.rand(3)
        return color


    def add_vector(self, path: str):

        coordinate = path.split("/")[-1].split("_")[3:5]
        coordinate = (int(coordinate[0]), int(coordinate[1]))

        target_coordinate = path.split("/")[-1].split(".")[0].split("_")[-2:]
        target_coordinate = (int(target_coordinate[0]), int(target_coordinate[1]))

        df = pd.read_csv(path)
        matrix = df.to_numpy()
        best_grid = self._get_best_grid(matrix, target_coordinate=target_coordinate)
        if best_grid is None:
            return
        vector, coordinate_transformed = self._create_vector(coordinate, best_grid)
        color = self.random_color_generator()
        self.vectors[coordinate_transformed] = (vector, color)

    def _create_vector(self, coordinate_a, coordinate_b):
        # Create a vector from coordinate_a to coordinate_b
        coordinate_a = (
            -coordinate_a[0],
            coordinate_a[1], # NOTE: 
        )
        coordinate_b = (
            -coordinate_b[0],
            coordinate_b[1],
        )
        vector = np.array(coordinate_b) - np.array(coordinate_a)
        return vector, coordinate_a

    def _get_best_grid(self, matrix, target_coordinate=(-1, -1)):

        kernel = np.ones(self.kernel_size)

        result = convolve2d(matrix, kernel, mode="same", fillvalue=0)
        if self.target:
            result2 = self.select_grid_close_to_target(result, target_coordinate)
            # Get the max index
            max_index = np.unravel_index(np.argmax(result2), result.shape)
        else:
            # Get the max index
            max_index = np.unravel_index(np.argmax(result), result.shape)
        # check if the max index is above a threshold
        if result[max_index] < self.threshold:
            return None

        return tuple(max_index)

    def select_grid_close_to_target(self, matrix, target_coordinate):

        neighbours = []
        for i in range(-self.threshold_grids_vetical, self.threshold_grids_vetical + 1, 1):
            for j in range(-self.threshold_grids_horizontal, self.threshold_grids_horizontal + 1, 1):

                if (
                    target_coordinate[0] + i < 0
                    or target_coordinate[1] + j < 0
                    or target_coordinate[0] + i >= matrix.shape[0]
                    or target_coordinate[1] + j >= matrix.shape[1]
                ):
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

        mask = np.zeros_like(matrix)
        mask[max_value[:, 0], max_value[:, 1]] = True

        # Ensure mask has the same shape as matrix
        mask = mask.astype(bool)
        matrix = np.where(mask, matrix, 0)
        # max_index = np.unravel_index(np.argmax(max_value), max_value.shape)
        # loop through the neighbours and find

        return matrix

    def plot_field(self):
        # Plot the vector field

        plt.figure(figsize=(8, 8))
        plt.title(f"Vector Field")

        for coordinate, (vector, color) in self.vectors.items():
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
                f"{vector}",
                color=color,
                fontsize=8,
            )

        plt.xlim(-0.5, self.grid_length_horizontal)
        plt.ylim(-self.grid_length_horizontal, 0.5)
        # Line on origin
        plt.axhline(0, color="black", lw=1)
        plt.axvline(0, color="black", lw=1)
        plt.grid()

        sim_method = "cos-sim-" if self.cosine_sim else "car-" if self.car else ""
        excluded = "excluded" if self.exclude else ""
        threshold = str(self.threshold).replace(".", "")
        target_str = "-target" if self.target else ""


        name = f"{self.name}-{self.episode}-{sim_method}{self.kernel_size[0]}x{self.kernel_size[1]}-{threshold}tr-{excluded}{target_str}"
        save_path = f"vector_fields/{self.name}-{self.episode}/{self.kernel_size[0]}x{self.kernel_size[1]}"
        os.makedirs(save_path, exist_ok=True)

        save_path = os.path.join(save_path, name)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.show()

    def plot_field_centered(self):
        # Plot the vector field centered
        plt.figure(figsize=(8, 8))
        plt.title("Vector Field Centered")
        #plt.title("Place field movement within defined boundaries")

        for coordinate, (vector, color) in self.vectors.items():
    
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
                f"{vector}",
                color=color,
                fontsize=8,
            )
        
        plt.xlim(-self.grid_length_horizontal, self.grid_length_horizontal)
        plt.ylim(-self.grid_length_horizontal, self.grid_length_horizontal)
        # Line on origin
        plt.axhline(0, color="black", lw=1)
        plt.axvline(0, color="black", lw=1)
        plt.grid()


        sim_method = "cos-sim-" if self.cosine_sim else "car-" if self.car else ""
        excluded = "excluded" if self.exclude else "included"
        threshold = str(self.threshold).replace(".", "")

        target_str = "-target" if self.target else ""
    
        name = f"{self.name}-{self.episode}-{sim_method}{self.kernel_size[0]}x{self.kernel_size[1]}-{threshold}tr-{excluded}{target_str}-centered"
        save_path = f"vector_fields/{self.name}-{self.episode}/{self.kernel_size[0]}x{self.kernel_size[1]}"
        os.makedirs(save_path, exist_ok=True)

        save_path = os.path.join(save_path, name)
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")

        plt.show()


def read_excluded_files(path, block: int, episode: int, cos_sim: bool) -> list:
    path = os.path.join(path, "excluded_grid_nums")

    files = os.listdir(path)

    if cos_sim:
        files = [f for f in files if "cosine_sim" in f]

    files = [f for f in files if f"block_{block}" in f]
    files = [f for f in files if f"episode_{episode}" in f]
    files = [f for f in files if f.endswith(".csv")]

    df = pd.read_csv(os.path.join(path, files[0]))
    excluded_grid_nums = df["Excluded Grid Numbers"].tolist()
    return excluded_grid_nums


def create_vector_field(method, exclude:bool, episode, source_map, target_map, target: bool = False):

    model_name = "helpful-bush-1369"
    grid_length = 6

    cosine_sim = False
    car = False

    
    if method == "CAV (Cos-sim)":
        cosine_sim = True
        car = False
    elif method == "CAR":
        cosine_sim = False
        car = True
    

    grid_length_horizontal = grid_length
    if target_map.__contains__("horizontally") or target_map.__contains__("vertically"):
        grid_length_horizontal = grid_length * 2

    block = 2

    path = f"vectors/{model_name}/grid_length_{grid_length}/remapping_src_{source_map}_target_{target_map}/"
    if cosine_sim:
        path += "cosine_sim/"

    elif car:
        path += "car/"

    target_name = target_map.split("_")[1]
    source_name = source_map.split("_")[1]
    if "two" in source_name:
        source_name = "two-rooms"
    if "two" in target_map:
        target_name = "two-rooms"

    if "vertical" in source_map:
        name = source_name + "-vertical"
    elif "horizontal" in source_map:
        name = source_name + "-horizontal"
    elif "rot" in source_map:
        name = source_name + "-rot"
    else:
        name = source_name

    if "vertical" in target_map:
        name += target_name + "-vertical"
    elif "horizontal" in target_map:
        name += target_name + "-horizontal"
    elif "rot" in target_map:
        name += target_name + "-rot"
    else:
        name += target_name

    vector_field = VectorField(
        grid_length=grid_length,
        grid_length_horizontal=grid_length_horizontal,
        target=target,
        name = name,
        cosine_sim=cosine_sim,
        car=car,
        exclude=exclude,
        episode=episode,
        block=block,
    )

    # excluded_grid_nums = read_excluded_files(path, block, episode, cosine_sim)

    for file in os.listdir(path):
        file = os.path.join(path, file)
        if not file.endswith(".csv"):
            continue
        episode_num = int(file.split(".")[0].split("_")[-4])
        block_num = int(file.split(".")[0].split("_")[-6])
        if episode_num != episode or block_num != block:
            continue

        grid_num = file.split("/")[-1].split("_")[0:3]
        grid_num = "_".join(grid_num)

        if calculate_threshold(file, grid_num, False, source_map, episode, block) and exclude:
            print("Excluded grid", file)
            continue

        vector_field.add_vector(file)

    #vector_field.plot_field()
    #vector_field.plot_field_centered()

    return vector_field


if __name__ == "__main__":
    create_vector_field()
