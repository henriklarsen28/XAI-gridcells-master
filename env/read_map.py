import numpy as np
from PIL import Image


def read_map(map_file):
    with open(map_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    return lines


def build_map(map_file):
    lines = read_map(map_file)
    map = []
    for line in lines:
        map.append([int(x) for x in line.strip().split(",")])

    map = np.array(map)

    return map


def symbol_orientation(orientation: int = 0):
    symbol = {0: ",", 1: "-", 2: ".", 3: "+"}
    return symbol[orientation]


def show_map(env_map: np.array, position: tuple, orientation: int = 0):

    # Walls are black if the value is -1 and white if the value is 1
    orientation_symbol = symbol_orientation(orientation)
    colors = {
        "1": (0, 0, 0),  # Border
        "0": (255, 255, 255),  # Walkable
        "2": (0, 255, 0),  # Goal
        ",": (255, 0, 0),  # Up
        "-": (0, 255, 255),  # Right
        ".": (0, 0, 255),  # Down
        "+": (255, 255, 0),  # Left
    }

    scale = 20
    cell_width = 10  # Set the width of a single cell

    width = env_map.shape[1]
    height = env_map.shape[0]

    img = Image.new("RGB", (width * scale, height * scale), (255, 255, 255))

    pixels = img.load()

    for i in range(height):
        for j in range(width):
            for y in range(scale):
                for x in range(scale):
                    pixels[j * scale + x, i * scale + y] = colors[env_map[i, j]]

    print(colors[orientation_symbol])
    print(position)
    for y in range(scale):
        for x in range(scale):

            pixels[position[1] * scale + x, position[0] * scale + y] = colors[
                orientation_symbol
            ]
    print(pixels)
    img.show()

    # print the size of the map
    print("Map size: ", env_map.shape)


def main():
<<<<<<< HEAD
    map_file = "map_v1/map.csv"
    map = build_map(map_file)
    show_map(map)
=======

    map_file = "map_v1/map.csv"
    map_env = build_map(map_file)
    print(map_env)
    show_map(map_env, (27, 10))
>>>>>>> 1a98982 (✨ feat: Prints dot and can move it. Needs to return available actions when a step is done)


if __name__ == "__main__":
    main()
