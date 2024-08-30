from read_map import build_map, show_map
import random as rd

def action_encoding(action: int) -> str:

    action_dict = {
        0: "forward",
        1: "left",
        2: "right"
    }
    

    raise ValueError("Invalid action")

class SunburstMazeDiscrete:

    def __init__(self, map_file: str):
        self.map_file = map_file
        self.map = build_map(map_file)
        self.action_space = ["Forward", "Left", "Right"]
        self.orientation = 0  # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.position = self.select_start_position()
        #self.last_position = None

    def select_start_position(self) -> tuple:
        # TODO: Maybe implement random selection
        return (26, 10)

    def reset(self) -> tuple:
        """
        Resets the environment to its initial state.
        Returns:
            tuple: The starting position of the agent.
        """
        #self.last_position = None
        self.map = build_map(self.map_file)
        self.position = self.select_start_position()
        return self.legal_actions()
    
    def can_move_forward(self) -> bool:

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

        if self.map[next_position[0]][next_position[1]] == 1:
            return False
        
        return True
    
    def legal_actions(self) -> list:


        # The agent can always turn left or right
        actions = ["left", "right"]

        # Check if the agent can move forward
        if self.can_move_forward():
            actions.append("forward")

        return actions


    def step(self, action):
        """
        Takes an action and performs the corresponding movement in the environment.

        Parameters:
            action (int): The action to be performed. 0 represents moving forward, 1 represents turning left, and 2 represents turning right.

        Returns:
            None
        """

        if action == "forward":
            self.move_forward()
        if action == "left":
            self.turn_left()
        if action == "right":
            self.turn_right()

        return self.legal_actions()

    def move_forward(self):
        """
        Moves the agent forward in the grid based on its current orientation.
        The agent's position is updated according to its orientation:
        - If the orientation is 0 (Up), the agent's position is decremented by 1 in the y-axis.
        - If the orientation is 1 (Right), the agent's position is incremented by 1 in the x-axis.
        - If the orientation is 2 (Down), the agent's position is incremented by 1 in the y-axis.
        - If the orientation is 3 (Left), the agent's position is decremented by 1 in the x-axis.
        """
        
        #self.last_position = self.position

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
        show_map(self.map, self.position, orientation=self.orientation)


def main():

    import time

    env = SunburstMazeDiscrete("map_v1/map.csv")
    env.show_map()
    
    #actions = [0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
