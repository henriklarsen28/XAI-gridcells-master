import os
import sys
from collections import deque

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# get the path to the project root directory and add it to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(project_root)
# sys.path.append("../")

import gymnasium as gym
import keras as keras
import numpy as np
from neural_network_ff import NeuralNetworkFF

import wandb
from env import SunburstMazeDiscrete

wandb.login()

test_episodes = 100

# Define the CSV file path relative to the project root
map_path_train = os.path.join(project_root, "env/map_v0/map_closed_doors.csv")
map_path_test = os.path.join(project_root, "env/map_v0/map_closed_doors.csv")

def test_agent():

    env = SunburstMazeDiscrete(map_path_test, render_mode="human")
    state_shape = (env.observation_space.n,)
    action_shape = (env.action_space.n,)

    ql = NeuralNetworkFF()
    # model = ql.agent(state_shape, action_shape)

    # Load the old model
    model = keras.models.load_model("model.keras")

    total_reward = 0
    render = True

    for i in range(test_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        print("Episode: ", i)
        while not done:

            if render:
                env.render()

            encoded = state.flatten()
            # Normalized the state
            env_size = env.width, env.height
            encoded = ql.state_to_input(encoded, env_size)
            encoded = encoded.flatten().reshape(1, -1)
            print(model.predict(encoded), encoded)
            action = np.argmax(model.predict(encoded))
            print(action)

            new_state, reward, done, _, info = env.step(action)
            total_reward += reward
            state = new_state

    env.close()


def train_agent():
    # Hyperparameters to log
    config = {
        "loss_function": "mse",
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "adam",
        "total_episodes": 3000,
        "epsilon": 0.8,
        "epsilon_decay": -0.005,
        "epsilon_min": 0.1,
        "discount_factor": 0.9,
        "alpha": 0.01,
        "map_path": map_path_train,
        "target_model_update": 1600, # hard update of the target model
        "max_steps_per_episode":800,
        "random_start_position":True,
        "replay_memory_size":deque(maxlen=2000),
        "rewards" : {
            "is_goal":1,
            "hit_wall":-0.001,
            "has_not_moved":0, # change to 0?
            "new_square":0.01,
            "max_steps_reached":-0.001,
            "distance_to_goal":True
       },

        # TODO
        "observation_space": {
            "position":True,
            "orientation":True,
            "steps_to_goal":True,

        },
        # TODO
        "layers": {
            0: {"activation": "relu",
                "nodes": 32
                },
            1: {"activation": "relu",   
                "nodes": 32
                },
            2: {"activation": "relu", 
                "nodes": 32
                },
            3: {"activation": "relu",
                "nodes": 32
                },
            4: {
                "activation": "linear",
                # "nodes": action_shape[0]
            }
        },
            "last_known_steps":5
        
    }

    render = True
    epsilon = config.get("epsilon")
    env = SunburstMazeDiscrete(
        map_path_train, 
        render_mode="human" if render else "none", 
        max_steps_per_episode=config.get("max_steps_per_episode"), 
        random_start_position=config.get("random_start_position"), 
        rewards=config.get("rewards"), 
        observation_space=config.get("observation_space"))
    state_shape = (env.observation_space.n,)
    action_shape = (env.action_space.n,)
    env_size = (env.width, env.height)

    # print(state_shape, action_shape)

    ql = NeuralNetworkFF()

    model = ql.agent(
        state_shape,
        action_shape,
        config.get("loss_function"),
        config.get("learning_rate"),
    )
    target_model = ql.agent(
        state_shape,
        action_shape,
        config.get("loss_function"),
        config.get("learning_rate"),
    )


    target_model.set_weights(model.get_weights())

    replay_memory = config.get("replay_memory_size")

    network_sync_rate_count = 0
    total_reward = 0
    step_count = 0

    run, config = ql.start_run(project="sunburst-maze", config=config)

    for i in range(config.get("total_episodes")):
        state = env.reset()
        print(
            "=" * 100,
            "\nRunning episode: ",
            i,
            "\nMouse position: ({}, {})".format(state[0], state[1]),
            "\nMouse orientation:",
            state[2],
        )

        done = False
        total_reward = 0
        total_rewards = []

        step_count = 0

        while not done:

            if render:
                env.render()

            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
                # print(action)
            else:

                encoded = state
                encoded = ql.state_to_input(encoded, env_size)
                encoded = encoded.flatten().reshape(1, -1)
                action = np.argmax(model.predict(encoded))
                # print(action)

            new_state, reward, done, _, info = env.step(action)
            # print("Reward: ", reward, "New State: ", new_state, "Done: ", done)
            total_reward += reward
            encoded = ql.state_to_input(state, env_size)
            new_encoded = ql.state_to_input(new_state, env_size)
            replay_memory.append(
                [encoded.flatten(), action, reward, new_encoded.flatten(), done]
            )

            state = new_state

            network_sync_rate_count += 1

            step_count += 1

            if done:
                ql.train(
                    replay_memory=replay_memory,
                    model=model,
                    target_model=target_model,
                    done=done,
                    batch_size=config.get("batch_size"),
                    discount_factor=config.get("discount_factor"),
                    alpha=config.get("alpha"),

                )
                # print(len(total_rewards))
                total_rewards.append(total_reward)

            
                if network_sync_rate_count > config.get("target_model_update"):
                    print("Updating target model")
                    target_model.set_weights(model.get_weights())
                    network_sync_rate_count = 0

                # Decay epsilon
                epsilon = (
                    epsilon + config.get("epsilon_decay")
                    if epsilon > config.get("epsilon_min")
                    else config.get("epsilon_min")
                )
                print(f"Total Reward: {total_reward}, \nEpsilon: {epsilon}")
                # ql.save_losses()

        wandb.log({"Reward per episode": total_reward, "Epsilon decay": epsilon, "Number of steps per episode": step_count})

        if i % 10 == 0 and i != 0:
            print(f"Total Reward: {total_reward}, \nEpsilon: {epsilon:.2f}")
            # wandb.log({"Total reward": total_reward, "Epsilon": epsilon})
            # Save the model
            model.save(f"model_episode_{i}.keras")


    # Save the model
    ql.save_losses()


if __name__ == "__main__":
    train_agent()
    # test_agent()
