from read_map import build_map, show_map


class SunburstMazeDiscrete:

    def __init__(self, map_file: str):
        self.map = build_map(map_file)
        self.action_space = list(range(3))
        self.orientation = 0  # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.position = self.select_start_position()

    def select_start_position(self) -> tuple:
        # TODO: Maybe implement random selection
        return (27, 10)

    def reset(self) -> tuple:
        """
        Resets the environment to its initial state.
        Returns:
            tuple: The starting position of the agent.
        """

        self.position = self.select_start_position()
        return self.position

    def step(self, action):
        """
        Takes an action and performs the corresponding movement in the environment.

        Parameters:
            action (int): The action to be performed. 0 represents moving forward, 1 represents turning left, and 2 represents turning right.

        Returns:
            None
        """

        if action == 0:  # Forward
            self.move_forward()
        if action == 1:  # Left
            self.turn_left()
        if action == 2:  # Right
            self.turn_right()

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

        This method updates the orientation of the agent by subtracting 1 from the current orientation and taking the modulo 4 to ensure the orientation stays within the range of 0 to 3.

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
        The modulo operation ensures that the orientation stays within the range of 0 to 3, representing the four cardinal directions (north, east, south, west).
        """
        self.orientation = (self.orientation + 1) % 4

    def show_map(self):
        show_map(self.map)


def main():
    env = SunburstMazeDiscrete("map_v1/map.csv")
    print(env.reset())
    env.show_map()


if __name__ == "__main__":
    main()
