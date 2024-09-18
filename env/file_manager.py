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
