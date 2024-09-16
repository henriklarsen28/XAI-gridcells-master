import random as rd

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .file_manager import build_map
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

    def __init__(self, maze_file=None, render_mode=None, max_steps_per_episode=1000):
        self.map_file = maze_file
        self.env_map = build_map(maze_file)
        self.height = self.env_map.shape[0]
        self.width = self.env_map.shape[1]

        # Three possible actions: forward, left, right
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Discrete(3)  # Orientation + next_to_wall

        self._action_to_direction = {
            "forward": self.move_forward,
            "left": self.turn_left,
            "right": self.turn_right,
        }
        self.orientation = 0  # 0 = Up, 1 = Right, 2 = Down, 3 = Left

        self.position = self.select_start_position()
        # self.last_position = None

        self.max_steps_per_episode = max_steps_per_episode
        self.steps_current_episode = 0

        ## Rendering
        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        ), f"Invalid render mode: {render_mode}"

        self.render_mode = render_mode
        self.render_maze = None
        if self.render_mode == "human":
            framerate = self.metadata["render_fps"]

            self.render_maze = Maze(
                self.map_file,
                self.env_map,
                self.width,
                self.height,
                framerate,
                self.position,
                self.orientation,
            )

        self.window = None
        self.clock = None

        ## Visited checkpoints
        self.visited_squares = []
        self.last_position = None
        self.last_moves = []

    def select_start_position(self) -> tuple:
        """
        Selects the start position for the maze.

        Returns:
            tuple: The coordinates of the selected start position.
        """
        random_position = (rd.randint(0, self.height - 1), rd.randint(0, self.width - 1))
        # Check if the position is not a wall
        while int(self.env_map[random_position[0]][random_position[1]]) == 1:
            random_position = (rd.randint(0, self.height - 1), rd.randint(0, self.width - 1))
            
        # print("Starting at random position: ", random_position)
        return random_position
    
    def _get_info(self):

        return {"legal_actions": self.legal_actions(), "orientation": self.orientation}

    def _get_observation(self):

        return np.array([*self.position, self.orientation])

    def reset(self, seed=None, options=None) -> tuple:

        super().reset(seed=seed)

        self.visited_squares = []
        self.env_map = build_map(self.map_file)
        self.position = self.select_start_position()
        self.reset_checkpoints()
        self.steps_current_episode = 0
        self.last_moves = []
        return self._get_observation()

    def reset_checkpoints(self):
        for checkpoint in checkpoints:
            checkpoint["visited"] = False

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
        if int(self.env_map[next_position[0]][next_position[1]]) == 1:
            return False

        return True

    def next_to_wall(self) -> list:
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

        # self.last_position = self.position

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
        self.last_moves.append(self.position)
        # Used if the action is invalid
        reward = self.reward(action)
        observation = self._get_observation()
        terminated = self.is_goal()
        info = self._get_info()

        if self.steps_current_episode >= self.max_steps_per_episode:
            print("Reached max steps")
            self.steps_current_episode = 0
            return observation, 0, True, True, self._get_info()
        
        action = action_encoding(action)
        self.steps_current_episode += 1

        # Stop if the agent is stuck and not moving
        if self.has_not_moved(self.position):
            # print("Not moving")
            return observation, reward, terminated, False, info

        # Walking into a wall
        if action not in self.legal_actions():
            # print("Hit a wall")
            return observation, -1, False, False, info
        self._action_to_direction[action]()

        # Updated values
        reward = self.reward(action)
        observation = self._get_observation()
        terminated = self.is_goal()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def is_goal(self):
        """
        Checks if the current position is a goal position.
        Returns:
            bool: True if the current position is a goal position, False otherwise.
        """
        if int(self.env_map[self.position[0]][self.position[1]]) == 2:
            return True
        return False

    def reward(self, action):
        """
        Calculates the reward for the current state.
        Returns:
            int: The reward value.
        """
        if self.is_goal():
            print("Goal reached!")
            return 100
        
        # TODO: Penalize for just rotating in place without moving
        current_pos = self.position
        if self.has_not_moved(self.position):
            return -0.5
        
        # Update the last position
        self.last_position = current_pos

        # for checkpoint in checkpoints:
        #     if self.position in checkpoint["coordinates"] and not checkpoint["visited"]:
        #         checkpoint["visited"] = True
        #         print("Checkpoint visited: ", self.position)
        #         return 20

        if self.position not in self.visited_squares:
            self.visited_squares.append(self.position)
            return 0.5

        return 0

    def render(self):
        if self.render_mode == "rgb_array" or self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        self.render_maze.draw_frame(self.env_map, self.position, self.orientation)

    def close(self):  # TODO: Not tested
        if self.window is not None:
            pygame.display.quit()


    def has_not_moved(self, position):
        # Only keep the last 100 moves

        if len(self.last_moves) < 100:
            return False
        

        self.last_moves = self.last_moves[-100:]
        if all(last_move == position for last_move in self.last_moves):
            return True