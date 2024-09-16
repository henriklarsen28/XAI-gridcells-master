import sys

sys.path.append("../")
from collections import deque
import os

import gymnasium as gym
import numpy as np
from neural_network_ff import NeuralNetworkFF
import tensorflow.keras as keras
from env import SunburstMazeDiscrete

train_episodes = 1000
test_episodes = 100

def test_agent():

    env = SunburstMazeDiscrete("../env/map_v0/map_closed_doors.csv", render_mode="human")
    state_shape = (env.observation_space.n,)
    action_shape = (env.action_space.n,)

    ql = NeuralNetworkFF()
    model = ql.agent(state_shape, action_shape)

    # Load the old model
    model = keras.models.load_model("model.keras")


    replay_memory = deque(maxlen=1_000_000)

    X = []
    y = []

    steps_until_train = 0
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

    epsilon = 1
    epsilon_decay = -0.01
    epsilon_min = 0.05
    render = False

    
    env = SunburstMazeDiscrete("../env/map_v0/map_closed_doors.csv", render_mode="human" if render else None)
    state_shape = (env.observation_space.n,)
    action_shape = (env.action_space.n,)
    env_size = (env.width, env.height)

    print(state_shape, action_shape)

    ql = NeuralNetworkFF()
    model = ql.agent(state_shape, action_shape)
    target_model = ql.agent(state_shape, action_shape)

    # Load the old model
    """if os.path.exists("model.keras"):
        model = keras.models.load_model("model.keras")"""

    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=500_000)

    X = []
    y = []

    steps_until_train = 0
    total_reward = 0

    for i in range(train_episodes):
        state = env.reset()
        print(state)
        done = False
        total_reward = 0
        total_rewards = []
        print("Episode: ", i)
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
                print(model.predict(encoded), encoded)
                action = np.argmax(model.predict(encoded))
                print(action)

            new_state, reward, done, _, info = env.step(action)
            #print("Reward: ", reward, "New State: ", new_state, "Done: ", done)
            total_reward += reward
            encoded = ql.state_to_input(state, env_size)
            new_encoded = ql.state_to_input(new_state, env_size)
            replay_memory.append(
                [encoded.flatten(), action, reward, new_encoded.flatten(), done]
            )

            state = new_state

            steps_until_train += 1

            if done:
                ql.train(
                    replay_memory=replay_memory,
                    model=model,
                    target_model=target_model,
                    done=done,
                )
                print(len(total_rewards))
                total_rewards.append(total_reward)

                if steps_until_train >= 10_000:
                    print("Updating target model")
                    target_model.set_weights(model.get_weights())
                    steps_until_train = 0

                # Decay epsilon
                epsilon = (
                    epsilon + epsilon_decay if epsilon > epsilon_min else epsilon_min
                )
                print(f"Episode: {i}, Total Reward: {total_reward}, Epsilon: {epsilon}")
                ql.save_losses()
        if i % 10 == 0:
            print(f"Episode: {i}, Total Reward: {total_reward}, Epsilon: {epsilon}")
            # Save the model
            model.save(f"model_episode_{i}.keras")
            

    # Save the model
    model.save("model.keras")
    ql.save_losses()
    env.close()


if __name__ == "__main__":
    train_agent()
    #test_agent()
