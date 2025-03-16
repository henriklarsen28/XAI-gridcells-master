import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import numpy as np
import csv
from env.file_manager import build_map, extract_goal_coordinates
from env.random_maps import save_map

def stretch_map_horizontally(map_path: str, stretch_factor=1):
    """
    Stretches the given map in the specified dimension.

    Parameters:
    - map_path (str): The path to the map file.
    - dim (int): The dimension to stretch the map. Default is 1 (horizontally), 0 is vetically.

    Returns:
    - stretched_map (np.array): The stretched map.
    """
    # Read the map
    map_array = build_map(map_path)

    # Stretch the map
    # Define how much to stretch

    # Process each row to stretch open spaces only
    stretched_rows = []
    for row in map_array:
        new_row = []
        for i, cell in enumerate(row):
            # Stretch 0s but keep 1s and 2s untouched
            if cell == 0:
                new_row.extend([0] * stretch_factor)
            else:
                if i == 0 and row[i + 1] != 1:
                    new_row.append(cell)
                    new_row.extend([0] * (stretch_factor - 1))
                elif row[i - 1] == 1:
                    new_row.extend([1] * (stretch_factor - 1))
                    new_row.append(cell)

                elif row[i - 1] == 0:
                    new_row.extend([0] * (stretch_factor - 1))
                    new_row.append(cell)

                elif row[i - 1] == 2:
                    new_row.pop()
                    new_row.append(0)

                    new_row.extend([2] * (stretch_factor - 1))
                    new_row.append(cell)
                    goal_pos = i * stretch_factor
                else:
                    new_row.extend([0] * (stretch_factor - 1))
                    new_row.append(cell)


        stretched_rows.append(new_row)

    stretched_map = np.array(stretched_rows)


    # TODO: Update goal_position
    old_goal = extract_goal_coordinates(map_path)
    new_goal = (old_goal[0], goal_pos)
    print("New_goal:", new_goal)

    return stretched_map, new_goal

def stretch_map_vertically(map_path: str, stretch_factor=1):
    """
    Stretches the given map vertically while keeping walls (1s) and goals (2s) intact.

    Parameters:
    - map_path (str): The path to the map file.
    - stretch_factor (int): How many times to stretch each row vertically. Default is 1 (no stretch).

    Returns:
    - stretched_map (np.array): The stretched map.
    """
    # Read the map
    map_array = build_map(map_path)

    # Stretch vertically
    stretched_map = []
    for i, row in enumerate(map_array):
        if all(cell == 1 for cell in row):
            # If it's a full wall row, copy it once
            stretched_map.append(row)
        else:
            # Stretch non-wall rows by repeating them
            # Check if the wall above is wall or not and the wall below
            for _ in range(stretch_factor):
                stretched_map.append(row)

    stretched_map = np.array(stretched_map)

    # Loop through the map to find duplicate walls and goals
    for i, row in enumerate(stretched_map):
        for j, cell in enumerate(row):
            if cell == 1:
                if i > 0 and stretched_map[i - 1][j] == 1:
                    stretched_map[i][j] = 1
                if i < len(stretched_map) - 1 and stretched_map[i + 1][j] == 1:
                    # If the cell below is a wall, make the current cell a wall
                    stretched_map[i][j] = 1
                else:
                    # If the cell below is not a wall, make the current cell empty
                    stretched_map[i][j] = 0
            elif cell == 2:
                if i < len(stretched_map) - 1 and stretched_map[i + 1][j] == 2:
                    stretched_map[i][j] = 0

    # Add border walls
    stretched_map[0,:] = 1
    stretched_map[-1,:] = 1

    # Update goal position
    old_goal = extract_goal_coordinates(map_path)
    new_goal = (old_goal[0] * stretch_factor-2, old_goal[1])
    print("New goal:", new_goal)

    return stretched_map, new_goal



def main():
    map_path = "goal/large/map_circular_4_19.csv"
    dst_file = map_path.split("/")[-1]

    folder = "goal/stretched/"
    os.makedirs(folder, exist_ok=True)
    map_name = "map_two_rooms"
    
    dst_file = os.path.join(folder, dst_file)
    with open(map_path, "rb") as src, open(dst_file, "wb") as dst:
        dst.write(src.read())

    stretched_map, goal = stretch_map_horizontally(map_path, stretch_factor=2)

    
    save_map(stretched_map, goal, map_name+"_horizontally", folder)

    stretched_map, goal = stretch_map_vertically(map_path, stretch_factor=2)
    save_map(stretched_map, goal, map_name+"_vertically", folder)

if __name__ == "__main__":
    main()
