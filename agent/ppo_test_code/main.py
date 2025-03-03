"""
	This file is the executable for running PPO. It is based on this medium article: 
	https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""
import os
import sys

# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(project_root)

import math
import sys

import gymnasium as gym
import torch
from arguments import get_args
from eval_policy import eval_policy
from network import FeedForwardNN
from ppo import PPO
from network_policy import FeedForwardNNPolicy

from env import SunburstMazeContinuous
from utils.calculate_fov import calculate_fov_matrix_size

# TODO: Implementere med ulike kart, byttes hver 20 episode

def train(env, hyperparameters, actor_model, critic_model):
	"""
		Trains the model.

		Parameters:
			env - the environment to train on
			hyperparameters - a dict of hyperparameters to use, defined in main
			actor_model - the actor model to load in if we want to continue training
			critic_model - the critic model to load in if we want to continue training

		Return:
			None
	"""	
	print(f"Training", flush=True)

	# Create a model for PPO.
	model = PPO(policy_class=FeedForwardNNPolicy, critic_class=FeedForwardNN, env=env, **hyperparameters)

	# Tries to load in an existing actor/critic model to continue training on
	if actor_model != '' and critic_model != '':
		print(f"Loading in {actor_model} and {critic_model}...", flush=True)
		model.actor.load_state_dict(torch.load(actor_model))
		model.critic.load_state_dict(torch.load(critic_model))
		print(f"Successfully loaded.", flush=True)
	elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
		print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
		sys.exit(0)
	else:
		print(f"Training from scratch.", flush=True)

	# Train the PPO model with a specified total timesteps
	# NOTE: You can change the total timesteps here, I put a big number just because
	# you can kill the process whenever you feel like PPO is converging
	model.learn(total_timesteps=200_000_000)

def test(env, actor_model):
	"""
		Tests the model.

		Parameters:
			env - the environment to test the policy on
			actor_model - the actor model to load in

		Return:
			None
	"""
	print(f"Testing {actor_model}", flush=True)

	

	# If the actor model is not specified, then exit
	if actor_model == '':
		print(f"Didn't specify model file. Exiting.", flush=True)
		sys.exit(0)

	# Extract out dimensions of observation and action spaces
	obs_dim = env.observation_space.n
	act_dim = env.action_space.shape[0]

	# Build our policy the same way we build our actor model in PPO
	policy = FeedForwardNNPolicy(obs_dim, act_dim)

	# Load in the actor model saved by the PPO algorithm
	policy.load_state_dict(torch.load(actor_model))

	# Evaluate our policy with a separate module, eval_policy, to demonstrate
	# that once we are done training the model/policy with ppo.py, we no longer need
	# ppo.py since it only contains the training algorithm. The model/policy itself exists
	# independently as a binary file that can be loaded in with torch.
	eval_policy(policy=policy, env=env, render=True)

def main(args):
	"""
		The main function to run.

		Parameters:
			args - the arguments parsed from command line

		Return:
			None
	"""
	# NOTE: Here's where you can set hyperparameters for PPO. I don't include them as part of
	# ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
	# To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
	hyperparameters = {
				'timesteps_per_batch': 2048, 
				'max_timesteps_per_episode': 500, 
				'gamma': 0.99, 
				'n_updates_per_iteration': 10,
				'lr': 3e-4, 
				'clip': 0.2,
				'render': True,
				'render_every_i': 10
			  }

	# Creates the environment we'll be running. If you want to replace with your own
	# custom environment, note that it must inherit Gym and have both continuous
	# observation and action spaces.
	#env = gym.make('Pendulum-v1', render_mode='human' if args.mode == 'test' else 'rgb_array')

	map_path_train = os.path.join(project_root, "env/map_v0/map_closed_doors_left.csv")

	rewards =  {
            "is_goal": 2,
            "hit_wall": -0.01,
            "has_not_moved": -0.005,
            "new_square": 0.0025,
            "max_steps_reached": -0.025,
            "penalty_per_step": -0.0002,
            "number_of_squares_visible": 0,
            "goal_in_sight": 0.1,
			"is_false_goal": -0.01,
	}
	fov_config = {
        "fov": math.pi / 1.5,
        "ray_length": 10,
        "number_of_rays": 100,
    }

	env = SunburstMazeContinuous(
		maze_file=map_path_train,
		render_mode="rgb_array",
		max_steps_per_episode=hyperparameters["max_timesteps_per_episode"],
		random_start_position=True,
		rewards=rewards,
		fov=fov_config["fov"],
		ray_length=fov_config["ray_length"],
		number_of_rays=fov_config["number_of_rays"],
		grid_length=4
	)

	# Train or test, depending on the mode specified
	'''if args.mode == 'train':
		train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model)
	else:
		test(env=env, actor_model=args.actor_model)'''
	
	actor_model = "../ppo/models/feed-forward/colorful-sunset-826/actor/ppo_actor_675.pth"

	test(env=env, actor_model=actor_model)

if __name__ == '__main__':
	args = get_args() # Parse arguments from command line
	main(args)
