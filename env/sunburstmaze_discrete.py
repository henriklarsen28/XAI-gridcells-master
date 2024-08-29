from read_map import build_map, show_map


class SunburstMazeDiscrete:

    def __init__(self, map_file: str):
        self.map = build_map(map_file)
        self.action_space = list(range(3))
        self.position = (0, 0)

    def select_start_position(self) -> tuple:
        # TODO: Implement
        return (0, 0)

    def reset(self):

        self.position = self.select_start_position()
        return self.position

    def show_map(self):
        show_map(self.map)


def main():
    env = SunburstMazeDiscrete("map_v1/map.csv")

    print(env.map.__repr__())
    print(env.reset())
    env.show_map()


if __name__ == "__main__":
    main()
