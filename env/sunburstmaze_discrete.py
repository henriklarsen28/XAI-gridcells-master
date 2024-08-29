from read_map import build_map, show_map


class SunburstMazeDiscrete:

    def __init__(self, map_file: str):
        self.map = build_map(map_file)
        self.action_space = list(range(3))
        self.orientation = 0: # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.position = self.select_start_position()

    def select_start_position(self) -> tuple:
        # TODO: Maybe implement random selection
        return (27,10)

    def reset(self):

        self.position = self.select_start_position()
        return self.position
    

    def step(self,action):

        if action == 0: # Forward
            self.move_forward()
        if action == 1: # Left
            self.turn_left()
        if action == 2: # Right
            self.turn_right()

    def move_forward(self):

        if self.orientation == 0: # Up
            self.position = (self.position[0] - 1, self.position[1])

        if self.orientation == 1: # Right
            self.position = (self.position[0], self.position[1] + 1)
        
        if self.orientation == 2: # Down
            self.position = (self.position[0] + 1, self.position[1])

        if self.orientation == 3: # Left
            self.position = (self.position[0], self.position[1] - 1)
    
    def rotate_left(self):
        self.orientation = (self.orientation - 1) % 4

    def rotate_right(self):
        self.orientation = (self.orientation + 1) % 4

    def show_map(self):
        show_map(self.map)


def main():
    env = SunburstMazeDiscrete("map_v1/map.csv")
    print(env.reset())
    env.show_map()


if __name__ == "__main__":
    main()
