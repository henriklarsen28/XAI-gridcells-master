from agent.ppo.ppo import PPO_agent
from env import SunburstMazeContinuous
from xai.cav.process_data import find_model_files
from xai.cav.continuous.build_dataset import build_csv_dataset

import math
import torch
import os

def main():
    config = {
    
            # MODEL PATHS
            "model_path": "../../../agent/ppo/models/transformers/expert-durian-1146/actor",
            "model_name": "expert-durian-1146", # TODO: change to the correct model name
            "model_episodes": [100, 150, 200], # TODO: change to the correct model episodes
    
            # PPO
            "policy_load_path": None,
            "critic_load_path": None,
            
            # ENVIRONMENT
            "env_name": "map_two_rooms_18_19", #TODO: change to the correct env name
            "env_path": "../../../env/random_generated_maps/goal/large/map_two_rooms_18_19.csv", # TODO: Change to the correct path for what the model was trained on
            # "env_path": "../../../env/map_v0/map_open_doors_horizontal.csv",
            "grid_length": 4,
    
    
            "cav": {
                "dataset_max_length" : 1500
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
    
    fov_config = {
                "fov": math.pi / 1.5,
                "ray_length": 15,
                "number_of_rays": 40,
                }
    
    print("config", config["PPO"])
    
    env = SunburstMazeContinuous(
        maze_file=config['env_path'],
        max_steps_per_episode=config["max_steps_per_episode"],
        render_mode=config["render_mode"],
        random_start_position=config["random_start_position"],
        random_goal_position=config["random_goal_position"],
        rewards=config["rewards"],
        fov=fov_config["fov"],
        ray_length=fov_config["ray_length"],
        number_of_rays=fov_config["number_of_rays"],
        grid_length=config["grid_length"]
    )
    
    agent = PPO_agent(
        env=env,
        device=device,
        config=config
    )

    # BUILD DATASET
    #model_path = os.path.join(config["model_path"], config["model_name"])
    model_name =config["model_name"]
    model_files = find_model_files(config["model_path"], config["model_episodes"])
    dataset_path = os.path.join('./dataset/', model_name, config["env_name"])
    dataset_subfolder = 'raw_data'

    # CAV 
    model_load_path = f"../../agent/dqn/models/{model_name}"
    #map_name = "map_conditional_prob_10_4"
    #map_name = "map_circular_4_5"
    map_name = "map_two_rooms_9_8"

    dataset_directory_train = f"./dataset/{model_name}/{map_name}/train"
    dataset_directory_test = f"./dataset/{model_name}/{map_name}/test"
    dataset_directory_random = f"./dataset/{model_name}/{map_name}"
    save_path = f"./results/cav/{model_name}/{map_name}"

    dataset_directory_train = f"{dataset_path}/train"
    dataset_directory_test = f"{dataset_path}/test"

    if not os.path.exists(dataset_directory_train):
        os.makedirs(dataset_directory_train, exist_ok=True)
    if not os.path.exists(dataset_directory_test):
        os.makedirs(dataset_directory_test, exist_ok=True)

    print("Model files:", model_files)

    # Build the dataset
    build_csv_dataset(
        env=env, 
        device=device, 
        config=config,
        model_files=model_files, 
        dataset_path=dataset_path, 
        dataset_subfolder=dataset_subfolder, 
        config=config)

    
    # Train CAV for grid observations
    for i in range(15):
        concept = f"grid_observations_{i}"
        # grid_observation_dataset(model_name, concept)
        # cav = CAV()
        for action in range(1):
            average = 1
            pass
            analysis = Analysis(average)
            for _ in range(average):
                cav = CAV()
                cav.calculate_cav(concept, dataset_directory_train, dataset_directory_test, model_load_path, sensitivity=False, action_index=action)
                cav.plot_cav(concept)



