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
sprite_left = pygame.tr