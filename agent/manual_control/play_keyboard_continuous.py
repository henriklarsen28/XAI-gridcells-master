import math
import sys
import pygame
import time
import numpy as np
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)


from env import SunburstMazeContinuous
from env.sunburstmaze_discrete import action_encoding


def perform_action(action: int, env: SunburstMazeContinuous):
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
    print("Action: ", action, env.orientation, env.position, env.velocity_x, env.velocity_y)

    observation, reward, goal, _, info = env.step(action)
    print(observation)
    #time.sleep(0.05)
    # env.show_map()
    return goal, env


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
        "ray_length": 15,
        "number_of_rays": 30,
        "random_start_position": True,
        "random_goal_position": True,

    }

    env = SunburstMazeContinuous(
        maze_file=f"{project_root}/env/random_generated_maps/goal/stretched/map_two_rooms_horizontally_18_40.csv",
        render_mode="human",
        rewards=config["rewards"],
        random_start_position=config["random_start_position"],
        max_steps_per_episode=2000,
        fov=config["fov"],
        ray_length=config["ray_length"],
        number_of_rays=config["number_of_rays"],
        grid_length=7
    )

    pygame.init()
    
    observation, _ = env.reset()
    print(env.goal)
    env.orientation = 90
    env.render()
    
    #print(observation)
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
                    action = np.array([0,0.4])
                elif event.key == pygame.K_a:
                    action = np.array([-0.4,0.1])
                elif event.key == pygame.K_d:
                    action = np.array([0.4,0.1])
            else:
                action = None


            if action is not None:
                done, env = perform_action(action, env)
                if done:
                    print("Episode finished.")
                    running = False
                    break
                if env.is_goal():
                    print("Goal reached!")
                    running = False
                    break
    print("Exiting...")


if __name__ == "__main__":
    play_with_keyboard()
