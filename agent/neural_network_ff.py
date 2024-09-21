import random
import random as rd

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, optimizers, regularizers, utils

import wandb

utils.disable_interactive_logging()


losses = []
total_rewards = []

class NeuralNetworkFF:
    def agent(self, state_shape, action_shape, loss_function, learning_rate):

        model = models.Sequential()
        model.add(layers.Dense(128, activation="relu", input_shape=(7,)))
        model.add(layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
        model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.01)))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(action_shape[0], activation="linear"))

        model.compile(
            loss=loss_function,
            optimizer=optimizers.Adam(learning_rate),
            metrics=["accuracy"],
        )

        return model

    def state_to_input(self, state: list, env_size: tuple):

        env_height = env_size[0]
        env_width = env_size[1]
        # Create a one-hot encoding of the orientation

        state_orientation = np.zeros(4)

        state_orientation[int(state[2])] = 1

        state_position_y = state[0] / env_height
        state_position_x = state[1] / env_width
        state_position = [state_position_y, state_position_x]
        # print(np.array([*state_position, *state_orientation]))
        return np.array([*state_position, *state_orientation, state[3] / 20])

    def train(
        self,
        replay_memory,
        model,
        target_model,
        done,
        batch_size,
        discount_factor,
        alpha,
    ):
        
        if len(replay_memory) < batch_size:
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

                target = reward + discount_factor * max_future_q  # Bellman equation
            else:
                target = reward

            current_q = current_q_values[i][action]
            current_q_values[i][action] = (1 - alpha) * current_q + alpha * target
            current_q = current_q_values[action]


            X_train.append(observation)
            y_train.append(current_q)
            total_reward += reward

            # print('current q:',current_q)
            # print("current q for action {}: {}".format(action, current_q[action]))

            wandb.log({"Q-value variance for each action": np.var(current_q_values), f"Q-value for selected action": current_q[action]})

        history = model.fit(
            np.array(X_train),
            np.array(y_train),
            verbose=0,
            shuffle=True,
            batch_size=batch_size,
            # callbacks=[WandbCallback()]

        )
        loss = history.history["loss"][0]
        losses.append(loss)
        total_rewards.append(total_reward)

        print("-" * 100)

        print("Loss: ", loss)

        wandb.log({"Loss per episode": loss})

    def save_losses(self):
        # Save the losses and rewards to a CSV file as columns using pandas
        df = pd.DataFrame({"loss": losses, "reward": total_rewards})
        df.to_csv("losses_rewards.csv", index=False)
        
    def start_run(self, project, config):
        run = wandb.init(
                    project=project,
                    config=config
                )

        config = wandb.config
        print("Wandb run initialized.")
        return run, config

    def end_run(self, run):
        run.finish()
        print("Wandb run finished.")

