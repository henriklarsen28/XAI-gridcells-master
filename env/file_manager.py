import numpy as np
from PIL import Image
import random as rd
import re

def build_grid_layout(env_map: np.array, num_cells: int):
    """
    Builds a grid layout for the environment map.

    Parameters:
    - env_map (list of lists): The environment map representing the maze.
    - num_cells (int): The number of cells in the grid layout.

    Returns:
    - grid_layout (list of lists): start_row, end_row, start_col, end_col
    """

    if num_cells in [4,5,6]:
        total_rows, total_cols = env_map.shape

        total_rows = total_rows - 1
        total_cols = total_cols - 1

        grid_layout = {}
        grid_id = 0

        row_splits = np.linspace(1, total_rows, num_cells + 1, dtype=int)
        col_splits = np.linspace(1, total_cols, num_cells + 1, dtype=int)
        print("row_splits", row_splits)
        print("col_splits", col_splits)
        for i in range(num_cells):
            for j in range(num_cells):
                row_start = row_splits[i]
                row_end = row_splits[i + 1]
                col_start = col_splits[j]
                col_end = col_splits[j + 1]

                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        grid_layout[(row, col)] = grid_id

                grid_id += 1

        return grid_layout, grid_id
    else:
        total_rows, total_cols = env_map.shape
        # Calculate heights and widths accounting for remainder
        subgrid_height = (total_rows + num_cells - 1) // num_cells
        subgrid_width = (total_cols + num_cells - 1) // num_cells

        grid_layout = {}
        grid_id = 0
        # Create indices for subgrids
        for row_start in range(0, total_rows, subgrid_height):
            for col_start in range(0, total_cols, subgrid_width):
                row_end = min(row_start + subgrid_height, total_rows)
                col_end = min(col_start + subgrid_width, total_cols)
                # Assign grid index to each cell within the subgrid
                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        grid_layout[(row, col)] = grid_id
                grid_id += 1
        
        num_cells = grid_id + 1

        return grid_layout, num_cells

def build_grid_layout_horizontal_stretch(env_map: np.array, num_cells_vertical: int):
    if num_cells_vertical in [4,5,6]:
        total_rows, total_cols = env_map.shape

        total_rows = total_rows - 1
        total_cols = total_cols - 1

        grid_layout = {}
        grid_id = 0
        num_cells_horizontal = num_cells_vertical * 2

        row_splits = np.linspace(1, total_rows, num_cells_vertical + 1, dtype=int)
        col_splits = np.linspace(1, total_cols, num_cells_horizontal + 1, dtype=int)
        print("row_splits", row_splits)
        print("col_splits", col_splits)
        for i in range(num_cells_vertical):
            for j in range(num_cells_horizontal):
                row_start = row_splits[i]
                row_end = row_splits[i + 1]
                col_start = col_splits[j]
                col_end = col_splits[j + 1]

                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        grid_layout[(row, col)] = grid_id

                grid_id += 1

        return grid_layout, grid_id
    else:
        total_rows, total_cols = env_map.shape
        #Remove the borders of the map
        # Calculate heights and widths accounting for remainder
        subgrid_height = (total_rows + num_cells_vertical - 1) // num_cells_vertical
        subgrid_width = subgrid_height
        #subgrid_width = (total_cols + num_cells - 1) // num_cells
        
        grid_layout = {}
        grid_id = 0
        # Create indices for subgrids
        for row_start in range(0, total_rows, subgrid_height):
            for col_start in range(0, total_cols, subgrid_width):
                row_end = min(row_start + subgrid_height, total_rows)
                col_end = min(col_start + subgrid_width, total_cols)
                # Assign grid index to each cell within the subgrid
                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        grid_layout[(row, col)] = grid_id
                grid_id += 1
        
        num_cells = grid_id + 1

        return grid_layout, num_cells

def build_grid_layout_vertical_stretch(env_map: np.array, num_cells_horizontal: int):

    if num_cells_horizontal in [4,5,6]:
        total_rows, total_cols = env_map.shape

        total_rows = total_rows # NOTE: -1 for tworooms vertical
        total_cols = total_cols - 1

        grid_layout = {}
        grid_id = 0
        num_cells_vertical = num_cells_horizontal * 2

        row_splits = np.linspace(1, total_rows, num_cells_vertical + 1, dtype=int)
        col_splits = np.linspace(1, total_cols, num_cells_horizontal + 1, dtype=int)
        print("row_splits", row_splits)
        print("col_splits", col_splits)
        for i in range(num_cells_vertical):
            for j in range(num_cells_horizontal):
                row_start = row_splits[i]
                row_end = row_splits[i + 1]
                col_start = col_splits[j]
                col_end = col_splits[j + 1]

                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        grid_layout[(row, col)] = grid_id

                grid_id += 1

        return grid_layout, grid_id
    else:
        total_rows, total_cols = env_map.shape
        #Remove the borders of the map
        # Calculate heights and widths accounting for remainder
        subgrid_width = (total_cols + num_cells_horizontal - 1) // num_cells_horizontal
        subgrid_height = subgrid_width
        #subgrid_width = (total_cols + num_cells - 1) // num_cells
        
        grid_layout = {}
        grid_id = 0
        # Create indices for subgrids
        for row_start in range(0, total_rows, subgrid_height):
            for col_start in range(0, total_cols, subgrid_width):
                row_end = min(row_start + subgrid_height, total_rows)
                col_end = min(col_start + subgrid_width, total_cols)
                # Assign grid index to each cell within the subgrid
                for row in range(row_start, row_end):
                    for col in range(col_start, col_end):
                        grid_layout[(row, col)] = grid_id
                grid_id += 1
        
        num_cells = grid_id + 1

        return grid_layout, num_cells

def extract_goal_coordinates(map_path: str):
    """
    Extracts the goal coordinates from the maze filename.

    Returns:
        tuple: The goal coordinates extracted from the maze filename.
    """
    # Extract the goal coordinates from the maze filename
    match = re.search(r"(\d+)_(\d+)\.csv$", map_path)
    if match:
        goal = (int(match.group(1)), int(match.group(2)))
    else:
        goal = None
    #print("Goal:", self.goal, "in maze file:", self.maze_file)
    return goal

def extract_goal_coordinates(map_path: str):
    """
    Extracts the goal coordinates from the maze filename.

    Returns:
        tuple: The goal coordinates extracted from the maze filename.
    """
    # Extract the goal coordinates from the maze filename
    match = re.search(r"(\d+)_(\d+)\.csv$", map_path)
    if match:
        goal = (int(match.group(1)), int(match.group(2)))
    else:
        goal = None
    #print("Goal:", self.goal, "in maze file:", self.maze_file)
    return goal

def get_map(map_path):

    map = map_path
    # check if map_load_path is a list, pick random from list
    if isinstance(map_path, list):
        map = rd.choice(map_path)
        print("Random map: ", map)
        
    return map

def read_map(map_file):
    
    with open(map_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    return lines


def build_map(map_file: str) -> np.array:
    assert map_file is not None, "Map file is not defined"
    lines = read_map(map_file)
    env_map = []
    for line in lines:
        env_map.append([int(x) for x in line.strip().split(",")])

    env_map = np.array(env_map)

    return env_map


def symbol_orientation(orientation: int = 0):
    """
    Returns the symbol corresponding to the given orientation.

    Parameters:
    - orientation (int): The orientation value. Default is 0.

    Returns:
    - symbol (str): The symbol corresponding to the given orientation.
    """
    symbol = {0: ",", 1: "-", 2: ".", 3: "+"}
    return symbol[orientation]

def main():

    # Load and print the map

    map_file = "map_v1/map.csv"
    map_env = build_map(map_file)
    print(map_env)


if __name__ == "__main__":
    main()