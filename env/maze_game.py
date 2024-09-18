import os

import numpy as np
import pygame

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
grey = (192, 192, 192)


class Maze:

    def __init__(
        self,
        map_file: str,
        env_map: np.array,
        width: int,
        height: int,
        framerate: int,
        position: tuple,
        orientation: int,
    ):
        pygame.init()
        self.map_file = map_file
        self.env_map = env_map
        self.width = width
        self.height = height

        self.cell_size = 30
        screen_width = width * self.cell_size
        screen_height = height * self.cell_size

        self.win = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("First Game")

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
        self.draw_frame(self.env_map, position, orientation)

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

    def draw_frame(self, env_map: np.array, position: tuple, orientation: int):
        """
        Draws a frame of the maze game.

        Args:
            env_map (np.array): The environment map representing the maze.
            position (tuple): The current position of the player.
            orientation (int): The orientation of the player.

        Returns:
            None
        """
        self.win.fill(white)  # fill screen before drawing
        self.draw_maze(env_map)
        self.draw_sprite(position, orientation)
        pygame.display.flip()
        self.clock.tick(self.framerate)
