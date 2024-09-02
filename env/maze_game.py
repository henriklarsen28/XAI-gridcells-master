import os
import random

import pygame
#from register_env import build_map, show_map




# mazeWidth = screenWidth // cellSize
# mazeHeight = screenHeight // cellSize
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
grey = (192, 192, 192)




cellSize = 30


X = 10  # start position
Y = 2
vel = 10  # how fast the object moves



"""
def get_sprite(sheet, frame, width, height):
    # Calculate the x and y coordinates on the sprite sheet
    x = (frame % 10) * width + 2
    y = (frame // 10) * height + 2
    # Create a new surface for the sprite
    sprite = pygame.Surface((width, height), pygame.SRCALPHA)
    # Blit the sprite onto the surface
    sprite.blit(sheet, (0, 0), (x, y, width, height))
    return sprite



sprites = []
for i in range(3):  # Change 10 to the number of sprites you want to extract
    sprite = get_sprite(sprite_sheet, i, spriteWidth, spriteHeight)
    # make sprite smaller
    sprite = pygame.transform.scale(sprite, (spriteWidth * 0.5, spriteHeight * 0.5))
    sprites.append(sprite)
"""



"""# Animation variables
current_frame = 0
frame_count = len(sprite)"""

"""
class Player:
    def __init__(self):
        self.x = X
        self.y = Y

    def move(self, dx, dy, maze):
        new_x = self.x + dx
        new_y = self.y + dy
        if (
            0 <= new_x < mazeWidth
            and 0 <= new_y < mazeHeight
            and maze[new_y][new_x] != 1
        ):
            self.x = new_x
            self.y = new_y

    def draw(self):
        # pygame.draw.rect(screen, GREEN, (self.x * cellSize, self.y * cellSize, cellSize, cellSize))
        win.blit(
            sprite, (self.x * cellSize, self.y * cellSize, spriteWidth, spriteHeight)
        )
"""

class Maze:

    def __init__(self, map_file, env_map, width, height, framerate, position, orientation):
        pygame.init()
        self.map_file = map_file
        self.env_map = env_map
        self.width = width
        self.height = height

        cellSize = 30
        screenWidth = width * cellSize
        screenHeight = height * cellSize

        self.win = pygame.display.set_mode((screenWidth, screenHeight))
        pygame.display.set_caption("First Game")

        # load sprite
        script_dir = os.path.dirname(__file__)
        sprite_file = os.path.join(script_dir, "images", "orange_mouse_single.png")
        sprite = pygame.image.load(sprite_file).convert_alpha()

        self.spriteWidth, self.spriteHeight = sprite.get_size()
        sprite = pygame.transform.scale(sprite, (self.spriteWidth * 0.5, self.spriteHeight * 0.5))

        self.sprite_right = sprite
        self.sprite_left = pygame.transform.flip(sprite, True, False)
        self.sprite_down = pygame.transform.rotate(self.sprite_left, 90)
        self.sprite_up = pygame.transform.rotate(sprite, 90)

        self.clock = pygame.time.Clock()
        self.framerate = framerate
        self.draw_frame(self.env_map, position, orientation)


    def select_sprite(self, orientation):
        if orientation == 0:
            return self.sprite_up
        elif orientation == 1:
            return self.sprite_right
        elif orientation == 2:
            return self.sprite_down
        elif orientation == 3:
            return self.sprite_left

    # set up the maze
    def draw_maze(self, env_map):
        print(env_map.size)
        for y in range(self.height):
            for x in range(self.width):
                if env_map[y][x] == 1:
                    pygame.draw.rect(
                        self.win, black, (x * cellSize, y * cellSize, cellSize, cellSize)
                    )
                elif env_map[y][x] == 2:
                    pygame.draw.rect(
                        self.win, green, (x * cellSize, y * cellSize, cellSize, cellSize)
                    )

    def draw_sprite(self, position, orientation):
        sprite = self.select_sprite(orientation)

        self.win.blit(
            sprite,
            (
                position[0] * cellSize,
                position[1] * cellSize,
                self.spriteWidth,
                self.spriteHeight,
            ),
        )

    def draw_frame(self, env_map, position, orientation):
        self.win.fill(white)  # fill screen before drawing
        self.draw_maze(env_map)
        self.draw_sprite(position, orientation)
        pygame.display.flip()
        self.clock.tick(self.framerate)


"""player = Player()
run = True
while run:
    # pygame.time.delay(100)  # delays the game so it doesn't run too fast
    for event in pygame.event.get():  # event from user
        if event.type == pygame.QUIT:  # if user quits
            run = False
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        # flip sprite
        sprite = sprite_left
        player.move(-1, 0, env)
        print(player.x, player.y)

    if keys[pygame.K_RIGHT]:
        sprite = sprite_right
        player.move(1, 0, env)
        print(player.x, player.y)

    if keys[pygame.K_UP]:
        sprite = sprite_up
        player.move(0, -1, env)
        print(player.x, player.y)

    if keys[pygame.K_DOWN]:
        sprite = sprite_down
        player.move(0, 1, env)
        print(player.x, player.y)

    win.fill(white)  # fill screen before drawing
    #draw_maze(win, env)
    player.draw()

    # Draw the current sprite
    # win.blit(sprite, (x, y, spriteWidth, spriteHeight))
    # Update the frame
    # current_frame = (current_frame + 1) % frame_count

    # pygame.draw.rect(win, (255, 0, 0), (x, y, width, height)) # draw rectangle
    # pygame.display.update()
    # Update the display
    pygame.display.flip()

pygame.quit()
"""