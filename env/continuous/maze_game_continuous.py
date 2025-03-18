import math
import os
from collections import deque

import numpy as np
import pygame
from PIL import Image

white = (255, 255, 255)
black = (0, 0, 0)
green = (91, 240, 146)
red = (220, 20, 60)
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
        goal: tuple,
        orientation: int,
        observed_squares_map: set,
        wall_rays: set,
        q_values: list = [],
    ):
        self.render_mode = render_mode
        self.map_file = map_file
        self.env_map = env_map
        self.width = width
        self.height = height
        self.position = position
        self.goal_position = goal
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
        sprite_file = os.path.join(script_dir, "../images", "orange_mouse_single.png")
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
        rect = self.sprite_up.get_rect()
        sprite = pygame.transform.rotate(self.sprite_up, -orientation)
        rotated_rect = sprite.get_rect(center=rect.center)

        
        return sprite, rotated_rect, rotated_rect.center

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
                    if (y, x) == self.goal_position:
                        # print("Goal position: ", (x, y))
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
                    else:
                            # print("False goal position: ", (x, y))
                            pygame.draw.rect(
                            self.win,
                            red,
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
        sprite, rotated_rect, center = self.select_sprite(orientation)
        self.win.blit(
            sprite,
            (
                (position[1] * self.cell_size) + rotated_rect[0] - center[0],
                (position[0] * self.cell_size) + rotated_rect[1] - center[1],
                rotated_rect[2],
                rotated_rect[3]
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

    def draw_rays(self, position: tuple, orientation: int, wall_rays: set):
        agent_angle = orientation * math.pi / 2
        #position_ahead = self.calculate_square_ahead(position, orientation)

        ray_shift_x = 0
        ray_shift_y = 0
        """if orientation == 0:
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
"""
        for x, y in wall_rays:
            pygame.draw.line(
                self.win,
                (255, 0, 0),
                (
                    (position[1] * self.cell_size) + ray_shift_y,
                    (position[0] * self.cell_size) + ray_shift_x,
                ),
                (y * self.cell_size + 15, x * self.cell_size + 15), # TODO: Fix the offset to match toril
            )

    def draw_marked_blocks(self, observed_squares_map: set):
        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        surface.set_alpha(128)

        for square in observed_squares_map:
            surface.fill(
                (255, 255, 255),
            )
            self.win.blit(
                surface, (square[1] * self.cell_size, square[0] * self.cell_size)
            )

    def draw_triangle(self, position, orientation, color=green):

        triangle_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
        triangle_surface.set_alpha(100)

        # Calculate coordinates of the triangle
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
    # TODO: Draw dots behind the agent to show the last 100 actions
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

    def render_grid_overlay(self, grid: dict, color_map: dict = None):
        
         # Ensure color_map is not None
        if color_map is None:
            color_map = {}
            for grid_id in set(grid.values()):
                # Default to random colors with random alpha (semi-transparent)
                color_map[grid_id] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255), 128)

        # Render each cell using the color associated with its grid ID
        for position, grid_id in grid.items():
            color = color_map[grid_id]  # Retrieve the RGBA color
            
            # Create a new surface with per-pixel alpha to support transparency
            cell_surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            cell_surface.fill(color)  # Fill the surface with the RGBA color
            
            # Blit the transparent surface onto the main window
            self.win.blit(cell_surface, (position[1] * self.cell_size, position[0] * self.cell_size))
            
    def draw_frame(
        self,
        env_map: np.array,
        position: tuple,
        orientation: int,
        observed_squares_map: set,
        wall_rays: set,
        q_values: list = [],
        last_ten_actions = deque(maxlen=10),
        grid_pos_to_id: dict = {},
        grid_id_to_color: dict = {}
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
        self.render_grid_overlay(grid_pos_to_id, grid_id_to_color)

        self.draw_maze(env_map)
        #self.draw_action_tail(last_ten_actions)
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
