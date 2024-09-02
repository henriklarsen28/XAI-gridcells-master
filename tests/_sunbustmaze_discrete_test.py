import sys

sys.path.append("..")


import unittest

from env import SunburstMazeDiscrete


class TestSunbustMazeDiscrete(unittest.TestCase):

    def test_can_move_forward_start_pos(self):

        env = SunburstMazeDiscrete(maze_file="../env/map_v1/map.csv")
        env.reset()
        env.position = (26, 10)
        env.orientation = 0

        env.show_map()
        self.assertTrue(env.can_move_forward())
        env.orientation = 1
        self.assertTrue(env.can_move_forward())
        env.orientation = 2
        self.assertFalse(env.can_move_forward())
        env.orientation = 3
        self.assertTrue(env.can_move_forward())

    def test_can_move_forward_start_room_top_left(self):

        env = SunburstMazeDiscrete(maze_file="../env/map_v1/map.csv")
        env.reset()
        env.position = (21, 6)
        env.orientation = 0

        env.show_map()
        self.assertFalse(env.can_move_forward())
        env.orientation = 1
        self.assertTrue(env.can_move_forward())
        env.orientation = 2
        self.assertTrue(env.can_move_forward())
        env.orientation = 3
        self.assertFalse(env.can_move_forward())

    def test_step_forward_start_pos(self):

        env = SunburstMazeDiscrete(maze_file="../env/map_v1/map.csv")
        env.reset()
        env.position = (26, 10)
        env.orientation = 0

        env.show_map()
        env.step(0)

        self.assertEqual(env.position, (25, 10))
        env.show_map()

    def test_move_forward_clear_path(self):

        env = SunburstMazeDiscrete(maze_file="../env/map_v1/map.csv")
        env.reset()
        env.position = (24, 10)
        env.orientation = 0

        env.show_map()
        env.move_forward()

        self.assertEqual(env.position, (23, 10))
        env.show_map()

    def test_is_goal_true(self):

        env = SunburstMazeDiscrete(maze_file="../env/map_v1/map.csv")
        env.reset()

        env.position = (26, 17)

        env.show_map()

        self.assertTrue(env.is_goal())


if __name__ == "__main__":
    unittest.main()
