import random as rd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt


def generate_random_maps(env_size: tuple):

    # Rules for the random map generation
    # 1. The map should have a border of walls
    # The map should have two rooms
    # The rooms should be connected by a corridor, with maybe a cooridoor around
    matrix = np.zeros(env_size)

    # Create the border walls
    matrix = add_border(matrix)
    print(matrix)
    # Generate a random map
    # 1. Generate random room sizes
    # Define how many rooms there should be
    room_count = env_size[0] // 10

    rooms = []
    for i in range(room_count):
        room_size = np.random.randint(max(5,env_size[1] // 5), max(7,env_size[1] // 2), 2)

        # Generate room matrises
        room = np.zeros(room_size)
        room = add_border(room)
        room = add_doors(room, 1)
        # Create a door in the room
        rooms.append(room)
        print(room)

    #print(rooms)
    # 2. Generate random room positions
    # Find a position for the centre of the room
    for room in rooms:
        room_position = np.random.randint(1, env_size[0] - room.shape[0]), np.random.randint(1, env_size[1] - room.shape[1])
        print(room_position)
        # Check if the room fits in the map

        

        # Check if the room overlaps with another room
        # If it does, generate a new room position



        # Generate room matrises
        room = np.zeros(room_size)
        room = add_border(room)

        # Create a door in the room
        rooms.append(room)

    # print(rooms)
    # 2. Generate random room positions
    # Find a position for the top left corner of the room
    corners_taken = []
    for room in rooms:
        taken = True
        while taken:

            room_position = np.random.randint(
                0, env_size[0] - room.shape[0]
            ), np.random.randint(
                0, env_size[1] - room.shape[1]
            )  # (x, y) substract the room size to make sure it fits in the map
            room_border = [
                (i, j)
                for i in range(room_position[0], room_position[0] + room.shape[0])
                for j in range(room_position[1], room_position[1] + room.shape[1])
            ]
            room_corners = [
                (room_position[0], room_position[1]),
                (room_position[0] + room.shape[0], room_position[1]),
                (room_position[0], room_position[1] + room.shape[1]),
                (room_position[0] + room.shape[0], room_position[1] + room.shape[1]),
            ]
            if not any(border in corners_taken for border in room_border):
                taken = False

        matrix[
            room_position[0] : room_position[0] + room.shape[0],
            room_position[1] : room_position[1] + room.shape[1],
        ] = room

        corners_taken.extend(
            [
                (i, j)
                for i in range(room_position[0], room_position[0] + room.shape[0])
                for j in range(room_position[1], room_position[1] + room.shape[1])
            ]
        )

        # Add doors to the rooms
        matrix = add_doors(matrix, 2, room_corners, room.shape)

        # change matrix to integer values
        matrix = matrix.astype(int)
        
    # 3. Generate random corridor positions

    # 4. Generate random corridor sizes

    # 5. Generate random wall positions

    return matrix


def add_border(matrix: np.array):
    matrix[0, :] = 1
    matrix[-1, :] = 1
    matrix[:, 0] = 1
    matrix[:, -1] = 1
    return matrix


def add_doors(
    matrix: np.array, number_of_doors: int, room_corners: list, room_size: tuple
):

    sides_with_door = np.random.randint(0, 3, number_of_doors)

    height, width = room_size
    # print(room_corners) # TODO: Fix the corners so that they are correct
    for side in sides_with_door:
        if side == 0:
            # Top
            door_width = max(width - 4, np.random.randint(1, width // 2))
            door_position = np.random.randint(
                room_corners[0][1] + 1, (width - door_width) + room_corners[0][1]
            )
            matrix[room_corners[0][0], door_position : (door_position + door_width)] = 0
        elif side == 1:
            # Right
            door_height = max(height - 4, np.random.randint(1, height // 2))

            door_position = np.random.randint(
                room_corners[2][0], room_corners[2][0] + (height - door_height)
            )
            matrix[
                door_position : (door_position + door_height), room_corners[2][1] - 1
            ] = 0

        elif side == 2:
            # Bottom
            door_width = max(width - 4, np.random.randint(1, width // 2))

            door_position = np.random.randint(
                room_corners[1][1], (width - door_width) + room_corners[1][1]
            )
            matrix[
                room_corners[1][0] - 1, door_position : door_position + door_width
            ] = 0

        elif side == 3:
            # Left
            door_height = max(height - 4, np.random.randint(1, height // 2))
            door_position = np.random.randint(
                room_corners[0][0] + 1, (height - door_height) + room_corners[0][0]
            )
            matrix[
                door_position : (door_position + door_height), room_corners[0][1]
            ] = 0

    return matrix


def generate_random_map_conditional_prob(env_size: tuple, prob: float):
    matrix = np.zeros(env_size, dtype=int)

    for i in range(1, env_size[0] - 1):
        for j in range(1, env_size[1] - 1):
            # Calculate the number of walls around
            neighbor_walls = sum(
                [matrix[i - 1, j], matrix[i + 1, j], matrix[i, j - 1], matrix[i, j + 1]]
            )
            # Reduce the sum if there is a wall diagonally
            diagonal_walls = sum(
                [
                    matrix[i - 1, j - 1],
                    matrix[i - 1, j + 1],
                    matrix[i + 1, j - 1],
                    matrix[i + 1, j + 1],
                ]
            )

            # Increase the probability of generating a wall if there are many walls around
            adjusted_prob = prob + 0.15 * neighbor_walls - 0.05 * diagonal_walls

            if rd.random() < adjusted_prob:
                matrix[i, j] = 1

    # Ytterkantene som vegger
    matrix = add_border(matrix)
    return matrix


def add_border(matrix: np.array):
    matrix[0, :] = 1
    matrix[-1, :] = 1
    matrix[:, 0] = 1
    matrix[:, -1] = 1
    return matrix


def add_doors(matrix: np.array, number_of_doors: int):
    sides_with_door = np.random.randint(0, 3, number_of_doors)

    height, width = matrix.shape

    for side in sides_with_door:
        if side == 0:
            # Top
            door_width = max(width-4, np.random.randint(1, width // 2))
            door_position = np.random.randint(1, width - door_width)
            matrix[0, door_position : door_position + door_width] = 0

        elif side == 1:
            # Right
            door_height = max(height-4, np.random.randint(1, height // 2))
            door_position = np.random.randint(1, height - door_height)
            matrix[door_position : door_position + door_height, -1] = 0

        elif side == 2:
            # Bottom
            door_width = max(width-4, np.random.randint(1, width // 2))
            door_position = np.random.randint(1, width - door_width)
            matrix[-1, door_position : door_position + door_width] = 0

        elif side == 3:
            # Left
            door_height = max(height-4, np.random.randint(1, height // 2))
            door_position = np.random.randint(1, height - door_height)
            matrix[door_position : door_position + door_height, 0] = 0

    return matrix

def generate_random_map_conditional_prob(env_size: tuple, prob: float):
    matrix = np.zeros(env_size, dtype=int)
    
    for i in range(1, env_size[0]-1):
        for j in range(1, env_size[1]-1):
            # Calculate the number of walls around
            neighbor_walls = sum([
                matrix[i-1, j], matrix[i+1, j],
                matrix[i, j-1], matrix[i, j+1]
            ])
            # Reduce the sum if there is a wall diagonally
            diagonal_walls = sum([
                matrix[i-1, j-1], matrix[i-1, j+1],
                matrix[i+1, j-1], matrix[i+1, j+1]
            ])
            
            # Increase the probability of generating a wall if there are many walls around
            adjusted_prob = prob + 0.15 * neighbor_walls - 0.05 * diagonal_walls
            
            if rd.random() < adjusted_prob:
                matrix[i, j] = 1

    # Ytterkantene som vegger
    matrix = add_border(matrix)
    return matrix

def generate_circular_maps(env_size: tuple):
    assert env_size[0] == env_size[1], "The environment size should be square"
    assert env_size[0] % 2 == 1, "The environment size should be odd"
    matrix = np.ones(env_size)
    center = (env_size[0] // 2, env_size[1] // 2)

    for r in range(1, env_size[0] - 1):
        for c in range(1, env_size[1] - 1):
            if (r - center[0]) ** 2 + (c - center[1]) ** 2 <= center[0] ** 2:
                matrix[r, c] = 0

    # matrix = add_goal(matrix)

    return matrix

def combine_maps(matrix1: np.array, matrix2: np.array):
    matrix = np.logical_or(matrix1, matrix2).astype(int)
    return matrix


def save_map(matrix: np.array, true_goal: tuple, path: str, folder: str):
    x = str(true_goal[1])
    y = str(true_goal[0])
    path = folder + path + "_" + y + "_" + x + ".csv"
    print("Saving map to", path)
    with open(path, "w") as f:
        for row in matrix:
            f.write(",".join([str(x) for x in row]) + "\n")
    
def add_goal(matrix: np.array):

    goal_position = np.random.randint(1, matrix.shape[0] - 1), np.random.randint(1, matrix.shape[1] - 1)
    matrix[goal_position] = 2
    print("Checking if goal is reachable from", goal_position)
    # check if the goal is reachable
    # if not, generate a new goal
    for i in range(1, matrix.shape[0] - 1):
        for j in range(1, matrix.shape[1] - 1):
            if matrix[i, j] == 0:
                # check if the goal is reachable
                if not is_reachable(matrix, goal_position):
                    matrix[goal_position] = 0
                    return add_goal(matrix)
    
    print("Goal is reachable from", goal_position)                
    return matrix, goal_position

def is_reachable(matrix: np.array, goal_position: tuple):

    # check if the goal is accessible by checking if it surrounded by at least four free cells
    i, j = goal_position
    if sum([matrix[i - 1, j], matrix[i + 1, j], matrix[i, j - 1], matrix[i, j + 1]]) <= 4:
        return True
    return False


def main():
    env_size = (21, 21)
    n_goals = 3

    # delete all files in the folder random_generated_maps/goal
    
    folder = "random_generated_maps/goal"
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    # Room 1
    matrix1 = generate_random_map_conditional_prob(env_size, 0.15)
    matrix2 = generate_circular_maps(env_size)
    matrix = combine_maps(matrix1, matrix2)
    
    goals = []
    for _ in range(n_goals):
        matrix, goal = add_goal(matrix)
        goals.append(goal)
    # choose random goal position
    true_goal = rd.choice(goals)
    save_map(matrix, true_goal, "/map_circular", folder)

    # Room 2
    matrix = generate_random_maps(env_size)
    goals = []
    for _ in range(n_goals):
        matrix, goal_position = add_goal(matrix)
        goals.append(goal_position)
    true_goal = rd.choice(goals)
    save_map(matrix, true_goal,"/map_two_rooms", folder)

    # Room 3
    matrix = generate_random_map_conditional_prob(env_size, 0.2)
    goals = []
    for _ in range(n_goals):
        matrix, goal_position = add_goal(matrix)
        goals.append(goal_position)
    true_goal = rd.choice(goals)
    save_map(matrix, true_goal,"/map_conditional_prob", folder)


if __name__ == "__main__":
    main()
