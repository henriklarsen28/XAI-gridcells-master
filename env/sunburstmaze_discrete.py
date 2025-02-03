import copy
import math
import random as rd
import re
from collections import deque

import gymnasium as gym
import numpy as np
import pandas as pd
import pygame
from gymnasium import spaces
from PIL import Image

from utils.calculate_fov import calculate_fov_matrix_size, step_angle

from .file_manager import build_map, get_map
from .maze_game import Maze

checkpoints = [
    {"coordinates": [(19, 9), (19, 10), (19, 11)], "visited": False},
    {"coordinates": [(13, 9), (13, 10), (13, 11)], "visited": False},
    {"coordinates": [(7, 9), (7, 10), (7, 11)], "visited": False},
    {"coordinates": [(5, 13), (6, 13), (7, 13)], "visited": False},
    {"coordinates": [(13, 16), (13, 17), (13, 18), (13, 19)], "visited": False},
]


def action_encoding(action: int) -> str:

    action_dict = {0: "forward", 1: "left", 2: "right"}

    return action_dict[action]


class SunburstMazeDiscrete(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(
        self,
        maze_file=None,
        render_mode=None,
        max_steps_per_episode=200,
        random_start_position=None,
        # random_goal_position=None,
        rewards=None,
        observation_space=None,
        fov=math.pi / 2,
        ray_length=10,
        number_of_rays=100,
    ):
        self.map_file = maze_file
        self.maze_file = get_map(maze_file)
        self.initial_map = build_map(self.maze_file)
        self.env_map = copy.deepcopy(self.initial_map)
        self.height = self.env_map.shape[0]
        self.width = self.env_map.shape[1]
        self.random_start_position = random_start_position
        # self.random_goal_position = random_goal_position
        self.rewards = rewards
        self.observation_space = observation_space
        self.render_mode = render_mode

        # calculate the size of the observation space for rewards of viewing the map
        self.map_observation_size = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.env_map[y][x] == 0:
                    self.map_observation_size += 1
                '''if self.env_map[y][x] == 2:
                    self.goal = (y, x)'''
        print("height:", self.height, "width:", self.width, "map_observation_size:", self.map_observation_size)
        

        # Three possible actions: forward, left, right

        self._action_to_direction = {
            "forward": self.move_forward,
            "left": self.turn_left,
            "right": self.turn_right,
        }

        self.orientation = 0  # 0 = Up, 1 = Right, 2 = Down, 3 = Left

        self.position = None
        self.goal = None

        # Episode step settings
        self.max_steps_per_episode = max_steps_per_episode
        self.steps_current_episode = 0

        # Pygame and rendering
        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        ), f"Invalid render mode: {render_mode}"
        self.render_mode = render_mode
        self.render_maze = None
        self.window = None
        self.clock = None

        ## Visited checkpoints
        # self.visited_squares = []
        self.last_position = None
        self.last_moves = []

        self.fov = fov
        self.half_fov = self.fov / 2
        self.ray_length = ray_length
        self.number_of_rays = number_of_rays
        self.matrix_size = calculate_fov_matrix_size(self.ray_length, self.half_fov)
        self.step_angle = step_angle(self.fov, self.number_of_rays)
        self.matrix_middle_index = int(self.matrix_size[1] / 2)

        self.wall_rays = set()
        self.observed_squares = set()
        self.observed_squares_map = set()
        self.observed_red_wall = set()
        self.goal_observed_square = set()

        self.q_variance = 0
        self.past_actions = deque(maxlen=10)

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Discrete(
            self.matrix_size[0] * self.matrix_size[1]
        )

        self.q_values = []
        self.goal_in_sight = False

    '''def goal_position(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.env_map[y][x] == 2:
                    if self.random_goal_position is True:
                        self.env_map[y][x] = 0
                        position = self.random_position()
                        self.env_map[position[0]][position[1]] = 2
                        return position
                    return (y, x)
        return None'''

    def select_start_position(self) -> tuple:
        """
        Selects the start position for the maze.

        Returns:
            tuple: The coordinates of the selected start position.
        """

        if self.random_start_position is True:
            position = self.random_position()
            self.orientation = rd.randint(0, 3)
        else:
            # position = (10, 13)
            position = (self.height - 2, 10)  # Bottom left for the small maze
        # print("Starting at random position: ", random_position)
        return position

    def random_position(self):

        # move the goal to a random position
        position = (rd.randint(0, self.height - 1), rd.randint(0, self.width - 1))
        # Check if the position is not a wall
        while int(self.env_map[position[0]][position[1]]) == 1:
            position = (
                rd.randint(0, self.height - 1),
                rd.randint(0, self.width - 1),
            )

        return position

    def _get_info(self):
        return {
            "legal_actions": self.legal_actions(),
            "orientation": self.orientation,
            "goal_in_sight": self.goal_in_sight,
        }

    def _get_observation(self):
        """
        Generates an observation of the current environment state.

        This method performs ray casting to obtain a matrix representation of the environment,
        flattens the matrix, and appends the agent's orientation to the resulting array.

        Returns:
            np.ndarray: A flattened array representing the environment state with the agent's orientation.
        """
        matrix = self.ray_casting()
        matrix = matrix.flatten()

        # Get the matrix of marked squares without rendering
        return np.array([*matrix, self.orientation])

    def extract_goal_coordinates(self):
        """
        Extracts the goal coordinates from the maze filename.

        Returns:
            tuple: The goal coordinates extracted from the maze filename.
        """
         # Extract the goal coordinates from the maze filename
        match = re.search(r'(\d+)_(\d+)\.csv$', self.maze_file)
        if match:
            self.goal = (int(match.group(1)), int(match.group(2)))
        else:
            self.goal = None
        print("Goal:", self.goal, "in maze file:", self.maze_file)
        return self.goal

    def reset(self, seed=None, options=None) -> tuple:

        super().reset(seed=seed)

        self.past_actions.clear()

        self.visited_squares = []
        self.viewed_squares = set()

        self.maze_file = get_map(self.map_file)
        self.initial_map = build_map(self.maze_file)
        self.env_map = copy.deepcopy(self.initial_map)
        self.position = self.select_start_position()
        self.goal = self.extract_goal_coordinates()

        self.steps_current_episode = 0

        self.reset_checkpoints()
        self.last_moves = []
        observation = self._get_observation()

        # Render the maze
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            framerate = self.metadata["render_fps"]

            self.render_maze = Maze(
                self.render_mode,
                self.initial_map,
                self.env_map,
                self.width,
                self.height,
                framerate,
                self.position,
                self.goal,
                self.orientation,
                self.observed_squares_map,
                self.wall_rays,
            )
        return observation, self._get_info()

    def reset_checkpoints(self):
        for checkpoint in checkpoints:
            checkpoint["visited"] = False

    def ray_casting(self):
        self.observed_squares = set()
        self.wall_rays = set()
        self.observed_squares_map = set()
        self.goal_observed_square = set()
        self.observed_red_wall = set()

        agent_angle = self.orientation * math.pi / 2  # 0, 90, 180, 270

        start_angle = agent_angle - self.half_fov
        for _ in range(self.number_of_rays + 1):
            for depth in range(self.ray_length):
                x = round(self.position[0] - depth * math.cos(start_angle))
                y = round(self.position[1] + depth * math.sin(start_angle))

                if self.env_map[x][y] == 1:
                    self.wall_rays.add((x, y))
                    break

                if self.env_map[x][y] == -1:  # Red wall
                    marked_x, marked_y = self.find_relative_position_in_matrix(x, y)
                    self.observed_red_wall.add((marked_x, marked_y))
                    break

                self.find_relative_position_in_matrix(x, y)

                self.observed_squares_map.add((x,y))
            start_angle += self.step_angle

        matrix = self.calculate_fov_matrix()
        return matrix

    def find_relative_position_in_matrix(self, x2, y2):
        x, y = self.position

    def find_relative_position_in_matrix(self, x2, y2):
        x, y = self.position

        if self.orientation == 0:
            marked_x = self.matrix_middle_index + y - y2
            marked_x = self.matrix_middle_index + y - y2
            marked_y = x - x2
        if self.orientation == 1:
            marked_x = self.matrix_middle_index + x2 - x
            marked_y = y2 - y

        if self.orientation == 2:
            marked_x = self.matrix_middle_index + y - y2
            marked_y = x2 - x

        if self.orientation == 3:
            marked_x = self.matrix_middle_index + x2 - x
            marked_y = y - y2

        # Add the goal square
        if (marked_x, marked_y) == self.goal:
            self.goal_observed_square.add((marked_x, marked_y)) # TODO: Not sure if this should be changed since we have three goals?

        self.observed_squares.add((marked_x, marked_y))

    def calculate_fov_matrix(self):
        matrix = np.zeros(calculate_fov_matrix_size(self.ray_length, self.half_fov))

        # Create a matrix with the marked squares from the marked_2 set
        for square in self.observed_squares:
            x, y = square
            matrix[y, x] = 1

        for square in self.observed_red_wall:
            x, y = square
            matrix[y, x] = -1

        # Mark the goal square
        if len(self.goal_observed_square) == 1:
            x, y = self.goal_observed_square.pop()
            matrix[y, x] = 2

        #df = pd.DataFrame(matrix)
        #df.to_csv("matrix.csv")

        # if self.orientation == 2 or self.orientation == 3:
        #     matrix = np.rot90(matrix, 2)
        #     matrix = np.roll(matrix, 1, axis=0)

        # if self.orientation == 3:
        #     matrix = np.roll(matrix, 1, axis=0)

        return matrix

    def can_move_forward(self) -> bool:
        """
        Determines whether the agent can move forward in the maze.
        Returns:
            bool: True if the agent can move forward, False otherwise.
        """

        # Get the coordinates of the cell in front of the agent
        if self.orientation == 0:
            next_position = (self.position[0] - 1, self.position[1])
        elif self.orientation == 1:
            next_position = (self.position[0], self.position[1] + 1)
        elif self.orientation == 2:
            next_position = (self.position[0] + 1, self.position[1])
        elif self.orientation == 3:
            next_position = (self.position[0], self.position[1] - 1)
        else:
            raise ValueError("Invalid orientation")

        # Check if the cell in front of the agent is a wall
        if int(self.env_map[next_position[0]][next_position[1]]) == 1 or int(self.env_map[next_position[0]][next_position[1]]) == -1:
            return False

        return True

    def next_to_wall(self) -> list: # TODO: Can be removed
        """
        Determines which directions the agent is next to a wall.

        Returns:
            list: A list of binary values indicating whether the agent is next to a wall in each direction.
                  The order of the values corresponds to [front, right, back, left].
                  A value of 1 indicates that the agent is next to a wall in that direction,
                  while a value of 0 indicates that the agent is not next to a wall in that direction.
        """

        next_to_wall = [0, 0, 0, 0]

        # TODO: Clean this up
        # Check if the cell in front of the agent is a wall
        if int(self.env_map[self.position[0] - 1][self.position[1]]) == 1:
            next_to_wall[0] = 1
        # Check if the cell to the right of the agent is a wall
        if int(self.env_map[self.position[0]][self.position[1] + 1]) == 1:
            next_to_wall[1] = 1
        # Check if the cell behind the agent is a wall
        if int(self.env_map[self.position[0] + 1][self.position[1]]) == 1:
            next_to_wall[2] = 1
        # Check if the cell to the left of the agent is a wall
        if int(self.env_map[self.position[0]][self.position[1] - 1]) == 1:
            next_to_wall[3] = 1
        return next_to_wall

    def legal_actions(self) -> list:
        """
        Returns a list of legal actions that the agent can take.

        Returns:
            list: A list of legal actions. The agent can always turn left or right.
            If the agent can move forward, "forward" is also included in the list.
        """

        # The agent can always turn left or right
        actions = ["left", "right"]

        if self.can_move_forward():
            actions.append("forward")

        return actions

    def move_forward(self):
        """
        Moves the agent forward in the grid based on its current orientation.
        The agent's position is updated according to its orientation:
        - If the orientation is 0 (Up), the agent's position is decremented by 1 in the y-axis.
        - If the orientation is 1 (Right), the agent's position is incremented by 1 in the x-axis.
        - If the orientation is 2 (Down), the agent's position is incremented by 1 in the y-axis.
        - If the orientation is 3 (Left), the agent's position is decremented by 1 in the x-axis.
        """

        if self.orientation == 0:  # Up
            self.position = (self.position[0] - 1, self.position[1])

        if self.orientation == 1:  # Right
            self.position = (self.position[0], self.position[1] + 1)

        if self.orientation == 2:  # Down
            self.position = (self.position[0] + 1, self.position[1])

        if self.orientation == 3:  # Left
            self.position = (self.position[0], self.position[1] - 1)

    def turn_left(self):
        """
        Turns the agent to the left.

        This method updates the orientation of the agent by subtracting 1 from the current orientation and
        taking the modulo 4 to ensure the orientation stays within the range of 0 to 3.

        Parameters:
            None

        Returns:
            None
        """
        self.orientation = (self.orientation - 1) % 4

    def turn_right(self):
        """
        Turns the agent to the right.

        This method updates the orientation of the agent by incrementing it by 1 and taking the modulo 4.
        The modulo operation ensures that the orientation stays within the range of 0 to 3, representing the four
        cardinal directions (north, east, south, west).
        """
        self.orientation = (self.orientation + 1) % 4

    def step(self, action: int):
        """
        Takes a step in the environment based on the given action.
        Parameters:
            action (int): The action to take in the environment.
        Returns:
            observation (list): The current observation of the environment, legal actions.
            reward (float): The reward obtained from the environment.
            turnicated (bool):
            terminated (bool): Whether the episode is terminated or not.
            info (dict): Additional information about the environment.
        """
        self.past_actions.append(
            (self.position, action, self.q_variance, self.orientation)
        )
        self.last_moves.append(self.position)
        # Used if the action is invalid
        reward = self.reward()
        observation = self._get_observation()
        terminated = self.is_goal()
        # terminated = self.view_of_maze_complete()
        info = self._get_info()

        if self.steps_current_episode >= self.max_steps_per_episode:
            print("Reached max steps")
            self.steps_current_episode = 0
            return (
                observation,
                self.rewards["max_steps_reached"],
                True,
                True,
                self._get_info(),
            )

        action = action_encoding(action)
        self.steps_current_episode += 1

        # Walking into a wall
        if action not in self.legal_actions():
            # print("Hit a wall")
            return observation, self.rewards["hit_wall"], False, False, info

        # Perform the action
        self._action_to_direction[action]()

        # Updated values
        reward = self.reward()
        
        # terminated = self.is_goal()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        if 2 in observation[:-1]:
            self.goal_in_sight = True # TODO: Not sure if this should be changed since we have three goals?

        return observation, reward, terminated, False, info

    def is_goal(self):
        """
        Checks if the current position is a goal position.
        Returns:
            bool: True if the current position is a goal position, False otherwise.
        """
        if int(self.env_map[self.position[0]][self.position[1]]) == 2 and self.position == self.goal:
            print("Correct goal reached!")
            return True
        return False
    
    def is_false_goal(self):
        """
        Checks if the current position is a goal position.
        Returns:
            bool: True if the current position is a goal position, False otherwise.
        """
        if int(self.env_map[self.position[0]][self.position[1]]) == 2 and self.position != self.goal:
            return True
        return False
    
    def view_of_maze_complete(self):
        if len(self.viewed_squares) == self.map_observation_size:
            return True
        return False

    def has_not_moved(self, position):
        # Only keep the last 50 moves

        if len(self.last_moves) < 10:
            return False
        self.last_moves = self.last_moves[-11:]
        if all(last_move == position for last_move in self.last_moves):
            # print("Has not moved from position: ", position)
            return True
        return False

    def number_of_squares_visible(self):
        # Return the number of squares that are 1 in the observation
        observation = self._get_observation()
        return len([square for square in observation if square == 1])

    # TODO: Gets stuck at wall, q-values for other actions are negative
    def reward(self):
        """
        Calculates the reward for the current state.
        Returns:
            int: The reward value.
        """

        if self.is_goal():
            print("Goal reached!")
            return self.rewards["is_goal"]
        # Penalize for just rotating in place without moving
        current_pos = self.position
        
        if self.has_not_moved(self.position):
            return self.rewards["has_not_moved"]
        # Update the last position
        self.last_position = current_pos

        # for checkpoint in checkpoints:
        #     if self.position in checkpoint["coordinates"] and not checkpoint["visited"]:
        #         checkpoint["visited"] = True
        #         print("Checkpoint visited: ", self.position)
        #         return 20
        # if self.decreased_steps_to_goal():
        #    return 0.00 #+ self.distance_to_goal_reward()
        '''reward = (
            self.rewards["number_of_squares_visible"] * self.number_of_squares_visible()
        )'''
        reward = 0
        if self.goal_in_sight:
            reward += self.rewards["goal_in_sight"]# # TODO: Not sure if this should be changed since we have three goals?

        if self.position not in self.visited_squares:
            self.visited_squares.append(self.position)
            reward +=  self.rewards["new_square"]# + reward # + self.distance_to_goal_reward()

        reward += self.rewards["penalty_per_step"]# + reward

        # Add reward for increasing the number of viewed squares
        viewed_squares_original = len(self.viewed_squares)
        self.viewed_squares.update(self.observed_squares_map)
        if viewed_squares_original < len(self.viewed_squares):
            # reward_new_squares = math.exp(len(self.viewed_squares) / self.map_observation_size)/math.exp(1) - 0.3
            # reward_new_squares = math.pow(1, len(self.viewed_squares) / self.map_observation_size)
            reward_new_squares = (len(self.viewed_squares) - viewed_squares_original) / self.map_observation_size
            reward += reward_new_squares
            print("Reward for viewing new squares: ", reward_new_squares)

        
        if self.is_false_goal():
            return self.rewards["is_false_goal"]

        return reward

    def render_q_value_overlay(self, q_values):
        """
        Renders the Q-values as an overlay on the maze.

        Args:
            q_values (np.ndarray): The Q-values to render as an overlay.

        Returns:
            None
        """
        self.render_maze.draw_q_values(q_values)

    def render(self):
        self._render_frame()

    def _render_frame(self):
        if self.render_mode == "rgb_array":
            return self.render_rgb_array()

        elif self.render_mode == "human":
            self.render_maze.draw_frame(
                self.env_map,
                self.position,
                self.orientation,
                self.observed_squares_map,
                self.wall_rays,
                self.q_values,
                self.past_actions,
            )

    def render_rgb_array(self):

        self.render_maze.render_mode = "rgb_array"
        frame = np.asarray(
            self.render_maze.draw_frame(
                self.env_map,
                self.position,
                self.orientation,
                self.observed_squares_map,
                self.wall_rays,
                [],
                []
            )
        )
        self.render_maze.render_mode = self.render_mode
        return frame

    def close(self):  # TODO: Not tested
        if self.window is not None:
            pygame.display.quit()

    def create_gif(self, gif_path: str, frames: list):
        """
        Creates a GIF from a list of frames.

        Args:
            frames (list): A list of frames to be included in the GIF.
            gif_path (str): The path to save the GIF file.
            duration (int): The duration of each frame in milliseconds.

        Returns:
            None
        """
        images = [Image.fromarray(frame, mode="RGB") for frame in frames]
        images[0].save(
            gif_path, save_all=True, append_images=images[1:], duration=100, loop=0
        )
        return gif_path