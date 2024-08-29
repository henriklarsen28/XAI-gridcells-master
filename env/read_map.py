import numpy as np
from PIL import Image


def read_map(map_file):
    with open(map_file, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        print(lines)
    return lines


def build_map(map_file):

    lines = read_map(map_file)
    map = []
    for line in lines:
        map.append(line.strip().split(","))

    map = np.array(map)

    return map


def show_map(env_map: np.array):

    # Walls are black if the value is -1 and white if the value is 1

    colours = {"1":(0, 0, 0), "0":(255, 255, 255), "2":(0,255,0)}

    scale = 20

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


def main():

    map_file = "../Samfundet_map_1.csv"
    map = build_map(map_file)
    print(map)
    show_map(map)


if __name__ == "__main__":
    main()
