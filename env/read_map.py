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


def show_map(env_map: np.array):

    # Walls are black if the value is -1 and white if the value is 1

    colours = {"1":(0, 0, 0), "0":(255, 255, 255), "2":(0,255,0)}

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
                    pixels[j * scale + x, i * scale + y] = colours[env_map[i, j]]

    img.show()

    # print the size of the map
    print("Map size: ", env_map.shape)


def main():
    map_file = "map_v1/map.csv"
    map = build_map(map_file)
    show_map(map)


if __name__ == "__main__":
    main()
