"""
	This file is used only to evaluate our trained policy/actor after
	training in main.py with ppo.py. I wrote this file to demonstrate
	that our trained policy exists independently of our learning algorithm,
	which resides in ppo.py. Thus, we can test our trained policy without 
	relying on ppo.py.
"""
import copy

from collections import deque
from utils.sequence_preprocessing import add_to_sequence


def _log_summary(ep_len, ep_ret, ep_num):
		"""
			Print to stdout what we've logged so far in the most recent episode.

			Parameters:
				None

			Return:
				None
		"""
		# Round decimal places for more aesthetic logging messages
		ep_len = str(round(ep_len, 2))
		ep_ret = str(round(ep_ret, 2))

		# Print logging statements
		print(flush=True)
		print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
		print(f"Episodic Length: {ep_len}", flush=True)
		print(f"Episodic Return: {ep_ret}", flush=True)
		print(f"------------------------------------------------------", flush=True)
		print(flush=True)

def rollout(policy, env, render, device):
	"""
		Returns a generator to roll out each episode given a trained policy and
		environment to test on. 

		Parameters:
			policy - The trained policy to test
			env - The environment to evaluate the policy on
			render - Specifies whether to render or not
		
		Return:
			A generator object rollout, or iterable, which will return the latest
			episodic length and return on each iteration of the generator.

		Note:
			If you're unfamiliar with Python generators, check this out:
				https://wiki.python.org/moin/Generators
			If you're unfamiliar with Python "yield", check this out:
				https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
	"""
	observation_sequence = deque()
	position_sequence = deque()
	print("Rolling out policy")

	# Rollout until user kills process
	ep_num = 0                # episodic number
	while True:
		obs, _ = env.reset()
		done = False

		# number of timesteps so far
		t = 0

		# Logging data
		ep_len = 0            # episodic length
		ep_ret = 0            # episodic return
		

		while not done:
			t += 1
			
			observation_sequence.append(obs)
			position_sequence.append(env.position)
	
			# Render environment if specified, off by default
			if render:
				try:
					env.render()
				except Exception as e:
					print(f"Error rendering environment: {e}")
					pass

			# Query deterministic action from policy and run it
			action = policy(obs).detach().numpy()
			obs, rew, terminated, truncated, _ = env.step(action)
			done = terminated | truncated

			# Sum all episodic rewards as we go along
			ep_ret += rew
			
			
		# Track episodic length
		ep_len = t
		ep_num += 0
		print("Episode:", ep_num)

		# returns episodic length and return in this iteration
		yield ep_len, ep_ret, observation_sequence, position_sequence

def eval_policy(policy, env, device, render=False): # TODO: Add device when calling eval_policy
	"""
		The main function to evaluate our policy with. It will iterate a generator object
		"rollout", which will simulate each episode and return the most recent episode's
		length and return. We can then log it right after. And yes, eval_policy will run
		forever until you kill the process. 

		Parameters:
			policy - The trained policy to test, basically another name for our actor model
			env - The environment to test the policy on
			render - Whether we should render our episodes. False by default.

		Return:
			None

		NOTE: To learn more about generators, look at rollout's function description
	"""
	print("Evaluating policy")
	collected_observations = deque ()
	# Rollout with the policy and environment, and log each episode's data
	for ep_num, (ep_len, ep_ret, observation_sequence, position_sequence) in enumerate(rollout(policy, env, render, device)):
		_log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num)
		collected_observations.append(
                    (
                        copy.deepcopy(observation_sequence),
                        copy.deepcopy(position_sequence),
                    )
                )
		
	return copy.deepcopy(collected_observations)