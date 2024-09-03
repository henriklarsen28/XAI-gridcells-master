import sys

import pygame

sys.path.append("..")

from env import SunburstMazeDiscrete
from env.sunburstmaze_discrete import action_encoding


def perform_action(action, env, legal_actions):
    """
    Perform the given action in the environment.

    Args:
        action (str): The action to be performed.
        env (Environment): The environment object.
        legal_actions (list): The list of legal actions.

    Returns:
        list: The updated list of legal actions.
    """
    #action = action_encoding(action)
    print("Action: ", action, env.orientation, env.position)
    
    legal_actions, reward, _, _, _, _ = env.step(action)
    #env.show_map()
    return legal_actions, env


def play_with_keyboard():
    """
    Plays the game using keyboard inputs.

    This function initializes the game environment, displays the map, and allows the user to control the agent using keyboard inputs.
    The user can press the 'q' key to quit the game, 'w' key to move the agent forward, 'a' key to turn the agent left, and 'd' key to turn the agent right.
    The function continuously checks for keyboard events and performs the corresponding action based on the key pressed.
    The legal actions are updated after each action is performed.

    Returns:
        None
    """

    env = SunburstMazeDiscrete(maze_file="../env/map_v1/map_closed_doors.csv", render_mode="human")

    pygame.init()
    legal_actions, _ = env.reset()
    print(legal_actions)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                    break
                if event.key == pygame.K_w:
                    action = 0
                elif event.key == pygame.K_a:
                    action = 1
                elif event.key == pygame.K_d:
                    action = 2
            else:
                action = None

            if action is not None:
                legal_actions, env = perform_action(action, env, legal_actions)
                if env.is_goal():
                    print("Goal reached!")
                    running = False
                    break
    print("Exiting...")


if __name__ == "__main__":
    play_with_keyboard()
