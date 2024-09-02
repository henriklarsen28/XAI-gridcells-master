import random as rd

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from env.file_manager import build_map, show_map


def action_encoding(action: int) -> str:

    action_dict = {0: "forward", 1: "left", 2: "right"}

    return action_dict[action]


class SunburstMazeDiscrete(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, maze_file=None):
        self.map_file = maze_file
        self.map = build_map(maze_file)
        self.height = self.map.shape[0]
        self.width = self.map.shape[1]

        self.window_size = (self.width * 20, self.height * 20)  # 20 X scale

        # Three possible actions: forward, left, right
        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.height, self.width), dtype=np.uint8
        )

        self._action_to_direction = {
            "forward": self.move_forward,
            "left": self.turn_left,
            "right": self.turn_right,
        }
        self.orientation = 0  # 0 = Up, 1 = Right, 2 = Down, 3 = Left

        self.position = self.select_start_position()
        # self.last_position = None

        assert (
            render_mode is None or render_mode in self.metadata["render.modes"]
        ), f"Invalid render mode: {render_mode}"
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def select_start_position(self) -> tuple:
        # TODO: Maybe implement random selection
        return (26, 10)

    def _get_info(self):

        return {"position": self.position, "orientation": self.orientation}

    def reset(self, seed=None, options=None) -> tuple:

        super().reset(seed=seed)
        # self.last_position = None
        self.map = build_map(self.map_file)
        self.position = self.select_start_position()
        return self.legal_actions, self._get_info()

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
        if int(self.map[next_position[0]][next_position[1]]) == 1:
            return False

        return True

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

    def step(self, action):
        """
        Takes an action and performs the corresponding movement in the environment.

        Parameters:
            action (int): The action to be performed. 0 represents moving forward, 1 represents turning left, and 2 represents turning right.

        Returns:
            None
        """
        action = action_encoding(action)
        self._action_to_direction[action]()

        terminated = self.is_goal()
        reward = self.reward()
        observation = self.legal_actions()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info, False, info

    def is_goal(self):
        """
        Checks if the current position is a goal position.
        Returns:
            bool: True if the current position is a goal position, False otherwise.
        """
        if int(self.map[self.position[0]][self.position[1]]) == 2:
            return True
        return False

    def reward(self):
        """
        Calculates the reward for the current state.
        Returns:
            int: The reward value.
        """
        if self.is_goal():
            return 100

        return -1

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        raise NotImplementedError

    # TODO: Remove this later
    def show_map(self):
        show_map(self.map, self.position, orientation=self.orientation)

    def close(self):  # TODO: Not tested
        if self.window is not None:
            pygame.display.quit()


def main():
    # Test the environment with random actions for 20 steps
    import time

    env = SunburstMazeDiscrete("map_v1/map.csv")
    env.show_map()

    available_actions = env.reset()
    print(available_actions)
    for _ in range(20):
        action = rd.choice(available_actions)
        print(action)
        available_actions = env.step(action)

        env.show_map()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
