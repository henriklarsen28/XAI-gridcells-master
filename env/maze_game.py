import os

import numpy as np
import pygame
import math

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
grey = (192, 192, 192)



NUMBER_OF_RAYS = 50
RAY_LENGTH = 30 # VIEW_DISTANCE BLOCKS
FIELD_OF_VIEW = math.pi / 2 # 180 degrees
HALF_FOV = FIELD_OF_VIEW / 2
STEP_ANGLE = FIELD_OF_VIEW / NUMBER_OF_RAYS


MATRIX_WIDTH = math.ceil(math.sin(HALF_FOV) * RAY_LENGTH)

MATRIX_WIDTH = MATRIX_WIDTH if MATRIX_WIDTH % 2 == 1 else MATRIX_WIDTH + 1
MATRIX_MIDDLE = math.ceil(MATRIX_WIDTH / 2)

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

        self.marked_squares = set()

        self.win = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Toril's maze")

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
        
    def calculate_fov_matrix(self):
        matrix = np.zeros((MATRIX_WIDTH, RAY_LENGTH))

        print("Shape: ", matrix.shape)
        print("Middle: ", MATRIX_MIDDLE)
        print("Matrix: ", matrix)

        import matplotlib.pyplot as plt
        x, y = zip(*self.marked_2)
        # Plotting the points
        plt.scatter(x,y, marker='o', color='b')

        # Adding labels and title
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.title('Plot of Coordinates')

        # Display the plot
        plt.show()


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

    def calculate_square_aheat(self, position: tuple, orientation: int):

        if orientation == 0:
            return (position[0] - 1, position[1])
        elif orientation == 1:
            return (position[0], position[1] + 1)
        elif orientation == 2:
            return (position[0] + 1, position[1])
        elif orientation == 3:
            return (position[0], position[1] - 1)
        return position


    def draw_raycast(self, position: tuple, orientation):
        agent_angle = orientation * math.pi / 2 # 0, 90, 180, 270
        position_ahead = self.calculate_square_aheat(position, orientation)
        ray_shift_x = 0
        ray_shift_y = 0
        if orientation == 0:
            ray_shift_y = 20
            ray_shift_x= 40
        elif orientation == 1:
            ray_shift_y = 0
            ray_shift_x = 20
        elif orientation == 2:
            ray_shift_y = 20
            ray_shift_x = 0
        elif orientation == 3:
            ray_shift_y = 40
            ray_shift_x = 20
        self.marked_2 = set()

        start_angle = agent_angle - HALF_FOV
        for ray in range(NUMBER_OF_RAYS):
            for depth in range(RAY_LENGTH):
                x = int(position[0] - depth * math.cos(start_angle))
                y = int(position[1] + depth * math.sin(start_angle))
                
                if self.env_map[x][y] == 1:
                    pygame.draw.line(self.win, (255,0,0), ((position_ahead[1] * self.cell_size) + ray_shift_y, (position_ahead[0] * self.cell_size) + ray_shift_x), (y * self.cell_size+15, x * self.cell_size+15))
                    break

                x_2 = int(MATRIX_MIDDLE + depth * math.cos(start_angle))
                y_2 = int(RAY_LENGTH - depth * math.sin(start_angle))

                marked_square = (x, y)
                print("Position in matrix: ", x_2, y_2)
                self.marked_squares.add(marked_square)
                self.marked_2.add((x_2, y_2))
            start_angle += STEP_ANGLE


    def draw_marked_blocks(self):
        for square in self.marked_squares:
            pygame.draw.rect(
                self.win,
                (0,0,255),
                (
                    square[1] * self.cell_size,
                    square[0] * self.cell_size,
                    self.cell_size,
                    self.cell_size,
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
        self.marked_squares = set()
        self.win.fill(white)  # fill screen before drawing
        self.draw_maze(env_map)
        self.draw_sprite(position, orientation)
        self.draw_raycast(position, orientation)
        self.draw_marked_blocks()
        
        print("Marked squares: ", self.marked_squares)
        print("Marked 2: ", self.marked_2)
        print("Position: ", position)
        pygame.display.flip()
        self.calculate_fov_matrix()
        self.clock.tick(self.framerate)
