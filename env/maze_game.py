import os
import random

import pygame
from read_map import build_map, show_map
from sunburstmaze_discrete import SunburstMazeDiscrete

pygame.init()

# mazeWidth = screenWidth // cellSize
# mazeHeight = screenHeight // cellSize
white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
grey = (192, 192, 192)


script_dir = os.path.dirname(__file__)
map_file = os.path.join(script_dir, "map_v1", "map.csv")
    
env = build_map(map_file)   
mazeHeight = env.shape[0]
mazeWidth = env.shape[1]

# window properties
cellSize = 30
screenWidth = mazeWidth * cellSize
screenHeight = mazeHeight * cellSize

print("screenWidth", screenWidth)
print('screenHeight', screenHeight)
print("mazeWidth", mazeWidth)
print("mazeHeight", mazeHeight)

win = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("First Game")


X = 10  # start position
Y = 2
vel = 10  # how fast the object moves

# load sprite
script_dir = os.path.dirname(__file__)
sprite_file = os.path.join(script_dir, "images", "orange_mouse_single.png")
sprite = pygame.image.load(sprite_file).convert_alpha()

# get sprite dinmensions
spriteWidth, spriteHeight = sprite.get_size()
sprite = pygame.transform.scale(sprite, (spriteWidth * 0.5, spriteHeight * 0.5))

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

sprite_right = sprite
sprite_left = pygame.transform.flip(sprite, True, False)
sprite_down = pygame.transform.rotate(sprite_left, 90)
sprite_up = pygame.transform.rotate(sprite, 90)

"""# Animation variables
current_frame = 0
frame_count = len(sprite)"""


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


# set up the maze
def draw_maze(screen, maze):
    for y in range(mazeHeight):
        for x in range(mazeWidth):
            if maze[y][x] == 1:
                pygame.draw.rect(
                    screen, black, (x * cellSize, y * cellSize, cellSize, cellSize)
                )
            elif maze[y][x] == 2:
                pygame.draw.rect(
                    screen, green, (x * cellSize, y * cellSize, cellSize, cellSize)
                )


player = Player()
run = True
while run:
    pygame.time.delay(100)  # delays the game so it doesn't run too fast
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
    draw_maze(win, env)
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
