import math
import os

import numpy as np
import pygame
from collections import deque
from PIL import Image

white = (255, 255, 255)
black = (0, 0, 0)
green = (91, 240, 146)
red = (255, 0, 0)
grey = (174, 174, 174)


class Maze:

    def __init__(
        self,
        render_mode: str,
        map_file: str,
        env_map: np.array,
        width: int,
        height: int,
        framerate: int,
        position: tuple,
        orientation: int,
        observed_squares_map: set,
        wall_rays: set,
    ):
        self.render_mode = render_mode
        self.map_file = map_file
        self.env_map = env_map
        self.width = width
        self.height = height
        self.position = position
        self.orientation = orientation
        self.cell_size = 30
        screen_width = width * self.cell_size
        screen_height = height * self.cell_size

        if render_mode == "rgb_array":
            os.environ["SDL_VIDEODRIVER"] = "dummy"  # run in headless mode (no display)
            pygame.init()
            pygame.display.set_mode((1, 1))
            self.win = pygame.Surface((screen_width, screen_height))
        else:
            pygame.init()
            pygame.display.set_caption("First Game")
            self.win = pygame.display.set_mode((screen_width, screen_height))


        # load sprite
        script_dir = os.path.dirname(__file__)
        sprite_file = os.path.join(script_dir, "images", "orange_mouse_single.png")
        sprite = pygame.image.load(sprite_file).convert_alpha()

        self.spriteWidth, self.spriteHeight = sprite.get_size()
        sprite = pygame.transform.scale(
            sprite, (self.spriteWidth * 0.35, self.spriteHeight * 0.35)
        )

        self.sprite_right = sprite
        self.sprite_left = pygame.transform.flip(sprite, True, False)
        self.sprite_down = pygame.transform.rotate(self.sprite_left, 90)
        self.sprite_up = pygame.transform.rotate(sprite, 90)
        
        
        self.clock = pygame.time.Clock()
        self.framerate = framerate
        self.draw_frame(self.env_map, position, orientation, observed_squares_map, wall_rays)


    def select_sprite(self, orientation: int):
        """
        Selects the appropriate sprite based on the given orientation.

        Parameters:
        - orientation (int): The orientation of the sprite. Must be a value between 0 and 3.

        Returns:
        - sprite (Sprite): The selected sprite based on the given orientation.
        """
        if orientation == 0:
            return self.sprite_up
        elif orientation == 1:
            return self.sprite_right
        elif orientation == 2:
            return self.sprite_down
        elif orientation == 3:
            return self.sprite_left

    # set up the maze
    def draw_maze(self, env_map: np.array) -> None:
        """
        Draws the maze based on the given environment map.

        Parameters:
        - env_map (list of lists): The environment map representing the maze.

        Returns:
        - None
        """
        for y in range(self.height):
            for x in range(self.width):
                if env_map[y][x] == 1:
                    pygame.draw.rect(
                        self.win,
                        black,
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )
                elif env_map[y][x] == 2:
                    pygame.draw.rect(
                        self.win,
                        green,
                        (
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size,
                        ),
                    )

    def draw_sprite(self, position: tuple, orientation: int) -> None:
        """
        Draws a sprite on the game window at the specified position and orientation.

        Parameters:
        - position (tuple): The position of the sprite in the game grid.
        - orientation (str): The orientation of the sprite.

        Returns:
        - None
        """
        sprite = self.select_sprite(orientation)

        self.win.blit(
            sprite,
            (
                position[1] * self.cell_size,
                position[0] * self.cell_size,
                self.spriteWidth,
                self.spriteHeight,
            ),
        )

    def calculate_square_ahead(self, position: tuple, orientation: int):

        if orientation == 0:
            return (position[0] - 1, position[1])
        elif orientation == 1:
            return (position[0], position[1] + 1)
        elif orientation == 2:
            return (position[0] + 1, position[1])
        elif orientation == 3:
            return (position[0], position[1] - 1)
        return position

    """def draw_raycast(self, position: tuple, orientation):
        agent_angle = orientation * math.pi / 2  # 0, 90, 180, 270
        position_ahead = self.calculate_square_ahead(position, orientation)
        ray_shift_x = 0
        ray_shift_y = 0
        if orientation == 0:
            ray_shift_y = 20
            ray_shift_x = 40
        elif orientation == 1:
            ray_shift_y = 0
            ray_shift_x = 20
        elif orientation == 2:
            ray_shift_y = 20
            ray_shift_x = 0
        elif orientation == 3:
            ray_shift_y = 40
            ray_shift_x = 20

        start_angle = agent_angle - HALF_FOV
        for ray in range(NUMBER_OF_RAYS):
            for depth in range(RAY_LENGTH):
                x = int(position[0] - depth * math.cos(start_angle))
                y = int(position[1] + depth * math.sin(start_angle))

                if self.env_map[x][y] == 1:
                    pygame.draw.line(
                        self.win,
                        (255, 0, 0),
                        (
                            (position_ahead[1] * self.cell_size) + ray_shift_y,
                            (position_ahead[0] * self.cell_size) + ray_shift_x,
                        ),
                        (y * self.cell_size + 15, x * self.cell_size + 15),
                    )
                    break

                if orientation == 0 or orientation == 2:
                    x_2 = int(MATRIX_MIDDLE_INDEX + depth * math.sin(start_angle))
                    y_2 = 0 + math.ceil(depth * math.cos(start_angle))
                if orientation == 1 or orientation == 3:
                    y_2 = int(depth * math.sin(start_angle))
                    x_2 = MATRIX_MIDDLE_INDEX - math.ceil(depth * math.cos(start_angle))

                # Change position based on orientaion
                marked_square = (x, y)
                # print("Position in matrix: ", x_2, y_2)
                self.marked_squares.add(marked_square)
                self.marked_2.add((x_2, y_2))
            start_angle += STEP_ANGLE"""

    def draw_rays(self, position: tuple, orientation: int, wall_rays: set):
        agent_angle = orientation * math.pi / 2
        position_ahead = self.calculate_square_ahead(position, orientation)

        ray_shift_x = 0
        ray_shift_y = 0
        if orientation == 0:
            ray_shift_y = 20
            ray_shift_x = 40
        elif orientation == 1:
            ray_shift_y = 0
            ray_shift_x = 20
        elif orientation == 2:
            ray_shift_y = 20
            ray_shift_x = 0
        elif orientation == 3:
            ray_shift_y = 40
            ray_shift_x = 20

        for x, y in wall_rays:
            pygame.draw.line(
                self.win,
                (255, 0, 0),
                (
                    (position_ahead[1] * self.cell_size) + ray_shift_y,
                    (position_ahead[0] * self.cell_size) + ray_shift_x,
                ),
                (y * self.cell_size + 15, x * self.cell_size + 15),
            )

    def draw_marked_blocks(self, observed_squares_map: set):
        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        surface.set_alpha(128)
        # surface.fill((255, 255, 255))
        for square in observed_squares_map:
            surface.fill(
                (255, 255, 255),
            )
            self.win.blit(
                surface, (square[1] * self.cell_size, square[0] * self.cell_size)
            )

    def draw_rays(self, position: tuple, orientation: int, wall_rays: set):
        agent_angle = orientation * math.pi / 2
        position_ahead = self.calculate_square_ahead(position, orientation)

        ray_shift_x = 0
        ray_shift_y = 0
        if orientation == 0:
            ray_shift_y = 20
            ray_shift_x = 40
        elif orientation == 1:
            ray_shift_y = 0
            ray_shift_x = 20
        elif orientation == 2:
            ray_shift_y = 20
            ray_shift_x = 0
        elif orientation == 3:
            ray_shift_y = 40
            ray_shift_x = 20

        for x,y in wall_rays:
            pygame.draw.line(
                self.win,
                (255, 0, 0),
                (
                    (position_ahead[1] * self.cell_size) + ray_shift_y,
                    (position_ahead[0] * self.cell_size) + ray_shift_x,
                ),
                (y * self.cell_size + 15, x * self.cell_size + 15),
            )

    def draw_triangle(self, position, orientation, color=green):

        triangle_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        triangle_surface.set_alpha(100)

        width = 2


        # Calculate coordinates of the triangle

        '''if orientation == 0:
            triangle_coordinates = [(0, 0),
                                    (self.cell_size, 0),
                                    (self.cell_size / 2, self.cell_size / 2)]
        elif orientation == 1:
            triangle_coordinates = [(self.cell_size, 0),
                                    (self.cell_size, self.cell_size),
                                    (self.cell_size / 2, self.cell_size / 2)]

        elif orientation == 2:
            triangle_coordinates = [(0, self.cell_size),
                                    (self.cell_size, self.cell_size),
                                    (self.cell_size / 2, self.cell_size / 2)]
        
        elif orientation == 3:
            triangle_coordinates = [(0, 0),
                                    (0, self.cell_size),
                                    (self.cell_size / 2, self.cell_size / 2)]'''
        
        if orientation == 2:
            triangle_coordinates = [(0, 0),
                                    (self.cell_size, 0),
                                    (self.cell_size / 2, self.cell_size / 2)]
        elif orientation == 3:
            triangle_coordinates = [(self.cell_size, 0),
                                    (self.cell_size, self.cell_size),
                                    (self.cell_size / 2, self.cell_size / 2)]

        elif orientation == 0:
            triangle_coordinates = [(0, self.cell_size),
                                    (self.cell_size, self.cell_size),
                                    (self.cell_size / 2, self.cell_size / 2)]
        
        elif orientation == 1:
            triangle_coordinates = [(0, 0),
                                    (0, self.cell_size),
                                    (self.cell_size / 2, self.cell_size / 2)]

        pygame.draw.polygon(
            triangle_surface,
            color,
            [
                triangle_coordinates[0],
                triangle_coordinates[1],
                triangle_coordinates[2],
            ]
        )


        self.win.blit(triangle_surface, (position[1] * self.cell_size, position[0] * self.cell_size))

    def draw_q_values(self, q_values):
        
        

        for q_value in q_values:
            for position, value in q_value.items():
                for orientation in range(4
                ):
                    saturation = 255 * (1-value[orientation])
                    if value[orientation] > 0.255:
                        color = (0,0,saturation)
                    else:
                        color = grey
                        # color = (0,0,255-saturation)
                    self.draw_triangle((position[0], position[1]), orientation, color)

    def draw_action_tail(self, last_ten_actions):
        orientation = 0
        for position, action, q_variance, orientation in last_ten_actions:
            arrow_orientation = 0
            if action == 0:
                arrow_orientation = orientation
            elif action == 1:
                arrow_orientation = (orientation + 4 - 1) % 4
            elif action == 2:
                arrow_orientation = (orientation + 1) % 4
            self.draw_triangle((position[0], position[1]), arrow_orientation, (255*(1-q_variance), 0, 0))

    def draw_frame(
        self,
        env_map: np.array,
        position: tuple,
        orientation: int,
        observed_squares_map: set,
        wall_rays: set,
        q_values: list = [],
        last_ten_actions = deque(maxlen=10),
    ):
        """
        Draws a frame of the maze game.

        Args:
            env_map (np.array): The environment map representing the maze.
            position (tuple): The current position of the player.
            orientation (int): The orientation of the player.

        Returns:
            returns matrix of the fov
        """
        self.marked_squares = set()
        self.marked_2 = set()
        self.win.fill(grey)  # fill screen before drawing
        self.draw_maze(env_map)
        self.draw_action_tail(last_ten_actions)
        #self.draw_rays(position, orientation, wall_rays)
        self.draw_marked_blocks(observed_squares_map)

        if self.render_mode == "human":
            self.draw_q_values(q_values)
            self.draw_sprite(position, orientation)
            pygame.display.flip()
        self.clock.tick(self.framerate)

        if self.render_mode == "rgb_array":
            self.draw_sprite(position, orientation)
            rgb_array = pygame.surfarray.array3d(self.win)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            return rgb_array
