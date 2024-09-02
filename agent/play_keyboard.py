import sys

import pygame

sys.path.append("..")

from env import SunburstMazeDiscrete


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
    if action in legal_actions:
        print("Action: ", action)
        legal_actions = env.step(action)
        env.show_map()
    else:
        print("Illegal action")
        print("Legal actions: ", legal_actions)
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

    env = SunburstMazeDiscrete(maze_file="../env/map_v1/map.csv", render_mode="human")

    pygame.init()
    legal_actions = env.reset()
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
                    action = "forward"
                elif event.key == pygame.K_a:
                    action = "left"
                elif event.key == pygame.K_d:
                    action = "right"
            else:
                action = None
            legal_actions, env = perform_action(action, env, legal_actions)
    print("Exiting...")


if __name__ == "__main__":
    play_with_keyboard()
