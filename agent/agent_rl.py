import sys

sys.path.append("../")
from collections import deque

import gymnasium as gym
import numpy as np
from neural_network_ff import NeuralNetworkFF

from env import SunburstMazeDiscrete

train_episodes = 20
test_episodes = 100



def train_agent():

    epsilon = 1
    epsilon_decay = 0.0001
    epsilon_min = 0.1
    render = True


    env = SunburstMazeDiscrete("../env/map_v1/map_closed_doors.csv", render_mode="human")
    state_shape = (env.observation_space.n,)
    action_shape = (env.action_space.n,)

    print(state_shape, action_shape)

    ql = NeuralNetworkFF()
    model = ql.agent(state_shape, action_shape)
    target_model = ql.agent(state_shape, action_shape)

    target_model.set_weights(model.get_weights())

    replay_memory = deque(maxlen=500_000)

    X = []
    y = []

    steps_until_train = 0
    total_reward = 0

    for i in range(train_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        print("Episode: ", i)
        while not done or total_reward > -500:

            if render:
                env.render()

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                # print(action)
            else:
                encoded = state
                action = np.argmax(model.predict(encoded))
                # print(action)

            new_state, reward, done, _, info = env.step(action)
            print("Reward: ", reward)
            total_reward += reward

            replay_memory.append(
                [state.flatten(), action, reward, new_state.flatten(), done]
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

            if steps_until_train >= 5000:

                target_model.set_weights(model.get_weights())
                steps_until_train = 0

                # Decay epsilon
                epsilon = (
                    epsilon + epsilon_decay if epsilon > epsilon_min else epsilon_min
                )


    # Save the model
    model.save("model.keras")
    ql.save_losses()
    env.close()


if __name__ == "__main__":
    train_agent()
