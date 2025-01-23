import copy
import math
import random as rd
from collections import deque

import time


import gymnasium as gym
import numpy as np
import pandas as pd
import pygame
from gymnasium import spaces
from PIL import Image

from utils.calculate_fov import calculate_fov_matrix_size, step_angle

from ..file_manager import build_map
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


class SunburstMazeContinuous(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(
        self,
        maze_file=None,
        render_mode=None,
        max_steps_per_episode=200,
        random_start_position=None,
        rewards=None,
        observation_space=None,
        fov=math.pi / 2,
        ray_length=10,
        number_of_rays=100,
    ):
        self.map_file = maze_file
        self.initial_map = build_map(maze_file)
        self.env_map = copy.deepcopy(self.initial_map)
        self.height = self.env_map.shape[0]
        self.width = self.env_map.shape[1]
        self.random_start_position = random_start_position
        self.rewards = rewards
        self.observation_space = observation_space
        self.render_mode = render_mode
        
        self.map_observation_size = 0
        for y in range(self.height):
            for x in range(self.width):
                if self.env_map[y][x] == 0:
                    self.map_observation_size += 1
        print("height:", self.height, "width:", self.width, "map_observation_size:", self.map_observation_size)


        self.orientation = 0  # Between 0 and 360 degrees, should probably be radians
        self.velocity_x = 0
        self.velocity_y = 0
        self.position = None

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
        self.visited_squares = []
        self.last_position = None
        self.last_moves = []

        self.viewed_squares = set()
        
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

        self.q_variance = 0
        self.past_actions = deque(maxlen=10)
        # Define the action space. Rotation and acceleration
        self.action_space = spaces.Box(low=np.array([-30.0, 0.0]), high=np.array([30.0,1.0]), dtype=np.float32)

        # TODO: Change how the observation space is defined
        y = self.matrix_size[0]
        x = self.matrix_size[1]
        # Observation space, position y, x and velocity
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), high=np.array([y, x, 1.0]), dtype=np.float32)


    def select_start_position(self) -> tuple:
        """
        Selects the start position for the maze.

        Returns:
            tuple: The coordinates of the selected start position.
        """

        if self.random_start_position is True:
            position = self.random_position()
            self.orientation = rd.randint(0, 360)
            self.orientation = 0
        else:
            # position = (10, 13)
            position = (self.height - 2, 10)  # Bottom left for the small maze
        # print("Starting at random position: ", random_position)
        return position

    def random_position(self):

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
            #"legal_actions": self.legal_actions(),
            "orientation": self.orientation
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

    def reset(self, seed=None, options=None) -> tuple:

        super().reset(seed=seed)

        self.past_actions.clear()

        # Reset visited and observed squares
        self.visited_squares = []
        self.viewed_squares = set()

        self.env_map = copy.deepcopy(self.initial_map)
        self.position = self.select_start_position()

        self.steps_current_episode = 0

        self.reset_checkpoints()
        self.last_moves = []
        observation = self._get_observation()

        # Render the maze
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            framerate = self.metadata["render_fps"]

            self.render_maze = Maze(
                self.render_mode,
                self.map_file,
                self.env_map,
                self.width,
                self.height,
                framerate,
                self.position,
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

        agent_angle = math.radians(self.orientation) # 0, 90, 180, 270
        start_angle = agent_angle - self.half_fov
        for _ in range(self.number_of_rays + 1):
            for depth in range(self.ray_length):
                x = round(self.position[0] - depth * math.cos(start_angle))
                y = round(self.position[1] + depth * math.sin(start_angle))

                if self.env_map[x][y] == 1:
                    self.wall_rays.add((x, y))
                    break

                self.find_relative_position_in_matrix(x, y)
                self.observed_squares_map.add((x,y))
            start_angle += self.step_angle
        
       # print("Observed squares: ", self.observed_squares)
        # print("Observed squares map: ", self.observed_squares_map)

        matrix = self.calculate_fov_matrix()
        #time.sleep(1)
        return matrix

    def find_relative_position_in_matrix(self, x2, y2):
        x, y = self.position
        x = math.ceil(x)
        y = int(y)
        marked_x = self.matrix_middle_index + y - y2 - 1
        marked_y = x - x2 - 1

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

        df = pd.DataFrame(matrix)
        df.to_csv("matrix.csv")

        # if self.orientation == 2 or self.orientation == 3:
        #     matrix = np.rot90(matrix, 2)
        #     matrix = np.roll(matrix, 1, axis=0)

        # if self.orientation == 3:
        #     matrix = np.roll(matrix, 1, axis=0)

        return matrix

    
    def is_collision(self, x, y):
        # Convert continuous coordinates to grid indices
        grid_x = math.floor(x)
        grid_y = math.floor(y)
        # Check if the agent is out of bounds or hits a wall
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return True
        return self.env_map[grid_y, grid_x] == 1
    
    def is_goal(self):
        """
        Checks if the current position is a goal position.
        Returns:
            bool: True if the current position is a goal position, False otherwise.
        """
        if int(self.env_map[round(self.position[0])][round(self.position[1])]) == 2:
            return True
        return False


    def limit_velocity(self):
        """
        Limits the velocity of the agent to a maximum value of 1.
        """
        if self.velocity_x > 1:
            self.velocity_x = 1
        if self.velocity_y > 1:
            self.velocity_y = 1

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
        rotation, velocity = action

        # Clip the actions
        velocity = np.clip(velocity, self.action_space.low[1], self.action_space.high[1])
        rotation = np.clip(rotation, self.action_space.low[0], self.action_space.high[0])


        self.orientation += rotation
        self.orientation = self.orientation % 360

        # Find the x and y components of the velocity
        self.velocity_x = velocity * math.sin(math.radians(self.orientation))
        self.velocity_y = -velocity * (math.cos(math.radians(self.orientation)))
        

        self.limit_velocity()


        position_y = self.position[0] + self.velocity_y
        position_x = self.position[1] + self.velocity_x
        

        if self.is_collision(position_x, position_y):
            print("Collision")
            return self._get_observation(), self.rewards["hit_wall"], False, False, self._get_info()

        self.position = (position_y, position_x)
        
        observation = self._get_observation()
        
        if self.is_goal():
            return observation, self.rewards["goal_reached"] + self.reward(), True, True, self._get_info()
        """self.past_actions.append(
            (self.position, action, self.q_variance, self.orientation)
        )
        self.last_moves.append(self.position)"""
        # Used if the action is invalid
        
        reward = self.reward()
        observation = None

        terminated = self.view_of_maze_complete()
        info = self._get_info()

        if self.is_goal():
            return observation, self.rewards["goal_reached"] + reward, True, True, self._get_info()

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

        self.steps_current_episode += 1
        # Updated values
        observation = self._get_observation()

        if self.render_mode == "human":
            self.render()

        #print("Observed squares map in step: ", self.observed_squares_map)
        #print('Number of squares in total', self.height * self.width)

        
   
        #print ("Length of viewed squares: ",  len(self.viewed_squares))
        #print("Proporition of viewed squares: ", len(self.viewed_squares) / self.map_observation_size)

        
        return observation, reward, terminated, False, info

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

        # Penalize for just rotating in place without moving
        current_pos = self.position

        reward = 0

        if self.position not in self.visited_squares:
            self.visited_squares.append(self.position)
            reward +=  self.rewards["new_square"]

        reward += self.rewards["penalty_per_step"]

        # Add reward for increasing the number of viewed squares
        viewed_squares_original = len(self.viewed_squares)
        self.viewed_squares.update(self.observed_squares_map)
        if viewed_squares_original < len(self.viewed_squares):
            reward += len(self.viewed_squares) / self.map_observation_size

        return reward

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
                [],
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
