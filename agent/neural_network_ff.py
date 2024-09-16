import random as rd

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import pandas as pd

losses = []
total_rewards = []

class NeuralNetworkFF:

    def agent(self, state_shape, action_shape, learning_rate=0.001):

        model = models.Sequential()
        model.add(layers.Dense(128, activation="relu", input_shape=(7,)))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(action_shape[0], activation="linear"))

        model.compile(
            loss="mse",
            optimizer=optimizers.Adam(learning_rate),
            metrics=["accuracy"],
        )

        return model

    def state_to_input(self,state: list, env_size: tuple):

        env_height = env_size[0]
        env_width = env_size[1]
        # Create a one-hot encoding of the orientation


        state_orientation = np.zeros(4)

        state_orientation[int(state[2])] = 1
        

        state_position_y = state[0] / env_height
        state_position_x = state[1] / env_width
        state_position = [state_position_y, state_position_x]
        #print(np.array([*state_position, *state_orientation]))
        return np.array([*state_position, *state_orientation, state[3]/20])

    def train(
        self,
        replay_memory,
        model,
        target_model,
        done,
        episodes=1000,
        batch_size=128,
        discount_factor=0.97,
        learning_rate=0.6,
    ):

        if len(replay_memory) < 500:
            return

        mini_batch = rd.sample(replay_memory, batch_size)

        states = np.array([batch[0] for batch in mini_batch])

        current_q_values = model.predict(states)
        new_states = np.array([batch[3] for batch in mini_batch])
        future_q_values = target_model.predict(new_states)

        X_train = []
        y_train = []
        total_reward = 0

        for i, [observation, action, reward, new_observation, done] in enumerate(
            mini_batch
        ):
            if not done:
                max_future_q = np.max(future_q_values[i])
                target = reward + discount_factor * max_future_q
            else:
                target = reward

            current_q = current_q_values[i][action]
            current_q_values[i][action] = (
                1 - learning_rate
            ) * current_q + learning_rate * target
            current_q = current_q_values[action]
            print("Current Q: ", current_q, "Target: ", target)
            
            X_train.append(observation)
            y_train.append(current_q)
            total_reward += reward

        history = model.fit(
            np.array(X_train), np.array(y_train), verbose=0, shuffle=True
        )
        loss = history.history["loss"][0]
        losses.append(loss)
        total_rewards.append(total_reward)
        print("Loss: ", loss)

    def save_losses(self):
        # Save the losses and rewards to a CSV file as columns using pandas
        df = pd.DataFrame({"loss": losses, "reward": total_rewards})
        df.to_csv("losses_rewards.csv", index=False)
