import numpy as np


def generate_random_maps(env_size: tuple):

    # Rules for the random map generation
    # 1. The map should have a border of walls
    # The map should have two rooms
    # The rooms should be connected by a corridor, with maybe a cooridoor around
    matrix = np.zeros(env_size)

    # Create the border walls
    matrix[0, :] = 1
    matrix[-1, :] = 1
    matrix[:, 0] = 1
    matrix[:, -1] = 1
    print(matrix)
    # Generate a random map
    # 1. Generate random room sizes
    # Define how many rooms there should be
    room_count = env_size[0] // 10

    room_sizes = []
    for i in range(room_count):
        room_size = np.random.randint(env_size[1]//5, env_size[1]//2, 2)
        print(room_sizes)
        room_sizes.append(room_size)
    # 2. Generate random room positions

    # 3. Generate random corridor positions

    # 4. Generate random corridor sizes

    # 5. Generate random wall positions


def generate_circular_maps(env_size: tuple):
    assert env_size[0] == env_size[1], "The environment size should be square"
    assert env_size[0] % 2 == 1, "The environment size should be odd"
    matrix = np.ones(env_size)
    center = (env_size[0] // 2, env_size[1] // 2)

    for r in range(1, env_size[0] - 1):
        for c in range(1, env_size[1] - 1):
            if (r - center[0]) ** 2 + (c - center[1]) ** 2 <= center[0] ** 2:
                matrix[r, c] = 0

    print(matrix)


def main():
    env_size = (21, 21)
    #generate_circular_maps(env_size)
    generate_random_maps(env_size)


if __name__ == "__main__":
    main()
