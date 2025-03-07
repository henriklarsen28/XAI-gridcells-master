import math
import os
import sys

import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from build_dataset import build_csv_dataset
from cav import CAV, Analysis

from env import SunburstMazeContinuous
from xai.cav.process_data import find_model_files


def main():
    fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 15,
        "number_of_rays": 40,
    }

    config = {
        # MODEL PATHS
        "model_path": "../../../agent/ppo/models/transformers/ppo/model_icy-violet-1223/actor",
        "model_name": "icy-violet-1223",  # NOTE: make sure to update
        "model_episodes": [575, 825, 1050],  # NOTE: for eval_policy
        # PPO
        "policy_load_path": None,
        "critic_load_path": None,
        # ENVIRONMENT
        "env_name": "map_circular_4_19",  # TODO: change to the correct env name
        "env_path": "../../../env/random_generated_maps/goal/large/map_circular_4_19.csv",  # TODO: Change to the correct path for what the model was trained on
        # "env_path": "../../../env/map_v0/map_open_doors_horizontal.csv",
        "grid_length": 7,  # 7 x 7 grid
        "cav": {
            "dataset_max_length": 1500,
            "episode_numbers": ["100", "200", "300", "400", "500", "600", "700"],
        },
        # RENDERING
        "train_mode": False,
        "map_path_train": None,
        "render": True,
        "render_mode": "human",
        # HYPERPARAMETERS
        "loss_function": "mse",
        "learning_rate": 3e-4,
        "batch_size": 3000,
        "mini_batch_size": 64,
        "n_mini_batches": 5,
        "optimizer": "adam",
        "PPO": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_updates_per_iteration": 10,  # hard update of the target model
            "clip": 0.2,
            "clip_grad_normalization": 0.5,
            "policy_kl_range": 0.0008,
            "policy_params": 5,
            "normalize_advantage": True,
        },
        "map_path": None,
        "max_steps_per_episode": 250,
        "random_start_position": True,
        "random_goal_position": False,
        "rewards": {
            "is_goal": 5,
            "hit_wall": -0.001,
            "has_not_moved": -0.005,
            "new_square": 0.0,
            "max_steps_reached": -0.025,
            "penalty_per_step": -0.00002,
            "number_of_squares_visible": 0,
            "goal_in_sight": 0.001,
            "is_false_goal": -0.01,
            # and the proportion of number of squares viewed (set in the env)
        },
        "observation_space": {
            "position": True,
            "orientation": True,
            "last_known_steps": 0,
            "salt_and_pepper_noise": 0,
        },
        "save_interval": 25,
        "render_fps": 15,
        "fov": fov_config["fov"],
        "ray_length": fov_config["ray_length"],
        "number_of_rays": fov_config["number_of_rays"],
        "transformer": {
            "sequence_length": 30,
            "n_embd": 128,
            "n_head": 6,
            "n_layer": 2,
            "dropout": 0.2,
            "decouple_positional_embedding": False,
        },
        "entropy": {"coefficient": 0.015, "min": 0.0001, "step": 1_000},
    }

    device = torch.device("cpu")

    # print("config", config["PPO"])

    env = SunburstMazeContinuous(
        maze_file=config["env_path"],
        max_steps_per_episode=config["max_steps_per_episode"],
        render_mode=config["render_mode"],
        random_start_position=config["random_start_position"],
        random_goal_position=config["random_goal_position"],
        rewards=config["rewards"],
        fov=fov_config["fov"],
        ray_length=fov_config["ray_length"],
        number_of_rays=fov_config["number_of_rays"],
        grid_length=config["grid_length"],
    )

    # BUILD DATASET
    model_name = config["model_name"]
    model_files = find_model_files(config["model_path"], config["model_episodes"])
    grid_length = "grid_length_" + str(config["grid_length"])
    dataset_path = os.path.join(
        "./dataset/", model_name, config["env_name"], grid_length
    )
    dataset_subfolder = "raw_data"

    episode_numbers = config["cav"]["episode_numbers"]

    # CAV
    map_name = config["env_name"]
    dataset_directory_random = f"./dataset/{model_name}/{map_name}"
    save_path = f"./results/{model_name}/{map_name}/{grid_length}"
    dataset_directory_train = f"{dataset_path}/train"
    dataset_directory_test = f"{dataset_path}/test"

    if not os.path.exists(dataset_directory_train):
        os.makedirs(dataset_directory_train, exist_ok=True)
    if not os.path.exists(dataset_directory_test):
        os.makedirs(dataset_directory_test, exist_ok=True)

    # Number of grid observations in the environment
    grid_size = env.num_cells

    # Build the dataset
    build_csv_dataset(
        env=env,
        device=device,
        config=config,
        actor_model_paths=model_files,
        dataset_path=dataset_path,
        dataset_subfolder=dataset_subfolder,
        grid_size=grid_size,
    )

    # Train CAV for grid observations
    for i in range(grid_size):
        print("CAVing for grid observation", i)
        concept = f"grid_observations_{i}"
        # grid_observation_dataset(model_name, concept)
        # cav = CAV()
        for action in range(1):
            average = 1
            pass
            analysis = Analysis(average)
            for _ in range(average):
                cav = CAV()
                # check if the concept exists by checking if the concept name exists in the dataset
                filename = f"{concept}_positive_train.csv"
                if filename not in os.listdir(dataset_directory_train):
                    print(
                        f"Concept {concept} does not exist in the dataset. Skipping..."
                    )
                    continue
                cav.calculate_cav(
                    concept=concept,
                    dataset_directory_train=dataset_directory_train,
                    dataset_directory_test=dataset_directory_test,
                    model_dir=config["model_path"],
                    sensitivity=False,
                    action_index=action,
                    episode_numbers=episode_numbers,
                    save_path=save_path,
                )
                cav.plot_cav(
                    concept=concept,
                    episode_numbers=episode_numbers,
                    save_path=save_path,
                )


if __name__ == "__main__":
    main()
