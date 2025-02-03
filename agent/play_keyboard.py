import math
import sys

import pygame

sys.path.append("..")

from env import SunburstMazeDiscrete
#from env.continuous.sunburstmaze_continuous import action_encoding


def perform_action(action: int, env: SunburstMazeDiscrete, legal_actions: list):
    """
    Perform the given action in the environment.

    Args:
        action (str): The action to be performed.
        env (Environment): The environment object.
        legal_actions (list): The list of legal actions.

    Returns:
        list: The updated list of legal actions.
    """
    # action = action_encoding(action)
    print("Action: ", action, env.orientation, env.position)

    observation, reward, goal, _, info = env.step(action)
    print("Observation: \n", observation)
    # env.show_map()
    return info["legal_actions"], env


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

    config = {
        "rewards": {
            "is_goal": 200,
            "hit_wall": -0.1,
            "has_not_moved": -0.1,
            "new_square": 0.2,
            "penalty_per_step": -0.1,
            "goal_in_sight": -0.1,
            "is_false_goal": -0.1,
        },
        # TODO
        "observation_space": {
            "position": True,
            "orientation": True,
            "steps_to_goal": True,
            "last_known_steps": 5,
        },
        "fov": math.pi / 1.5,
        "ray_length": 10,
        "number_of_rays": 100,
    }

    env = SunburstMazeDiscrete(
        maze_file="../env/random_generated_maps/goal/map_circular_16_18.csv",
        render_mode="human",
        rewards=config["rewards"],
        observation_space=config["observation_space"],
        fov=config["fov"],
        ray_length=config["ray_length"],
        number_of_rays=config["number_of_rays"],
    )

    pygame.init()
    observation, _ = env.reset()
    #print(observation)
    legal_actions = env.legal_actions()
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
