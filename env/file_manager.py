import numpy as np
from PIL import Image


def read_map(map_file):
    with open(map_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    return lines


def build_map(map_file):

    lines = read_map(map_file)
    env_map = []
    for line in lines:
        env_map.append(line.strip().split(","))

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


def show_map(env_map: np.array, position: tuple, orientation: int = 0):
    """
    Displays the environment map with the current position and orientation highlighted.

    Args:
        env_map (np.array): The environment map represented as a numpy array.
        position (tuple): The current position in the map.
        orientation (int, optional): The current orientation. Defaults to 0.

    Returns:
        None
    """

    # Walls are black if the value is -1 and white if the value is 1
    orientation_symbol = symbol_orientation(orientation)
    colors = {
        "1": (0, 0, 0),  # Border
        "0": (255, 255, 255),  # Walkable
        "2": (0, 255, 0),  # Goal
        ",": (255, 0, 0),  # Up (red)
        "-": (0, 255, 255),  # Right (cyan)
        ".": (0, 0, 255),  # Down (blue)
        "+": (255, 255, 0),  # Left (yellow)
    }

    scale = 20

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


def main():
    # Load and print the map
    map_file = "map_v1/map.csv"
    map_env = build_map(map_file)
    print(map_env)
    show_map(map_env, (27, 10))


if __name__ == "__main__":
    main()
