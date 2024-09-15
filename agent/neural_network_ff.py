import random as rd

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, utils
import pandas as pd


import wandb
import random

utils.disable_interactive_logging()


losses = []
total_rewards = []

# Hyperparameters to log
loss_function = "huber"
lr=0.001
optimizer = optimizers.Adam(lr)


class NeuralNetworkFF:
    def agent(self, state_shape, action_shape, learning_rate=lr):

        model = models.Sequential()
        model.add(layers.Dense(128, activation="relu", input_shape=(6,)))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(32, activation="relu"))
        model.add(layers.Dense(action_shape[0]))

        model.compile(
            loss=loss_function,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        return model

    def state_to_input(self,state: list, env_size: tuple):

        env_height = env_size[0]
        env_width = env_size[1]
        # Create a one-hot encoding of the orientation


        state_orientation = np.zeros(4)
        state_orientation[state[2]] = 1
        

        state_position_y = state[0]
        state_position_x = state[1]
        state_position = [state_position_y, state_position_x]
        #print(np.array([*state_position, *state_orientation]))
        return np.array([*state_position, *state_orientation])

    def train(
        self,
        replay_memory,
        model,
        target_model,
        done,
        config,
        episodes=1000,
        batch_size=256,
        discount_factor=0.90,
        learning_rate=0.7,
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
        episode_loss = 0 

        for i, [observation, action, reward, new_observation, done] in enumerate(
            mini_batch
        ):
            if not done:
                max_future_q = np.max(future_q_values[i])
                target = reward + discount_factor * max_future_q # Bellman equation
            else:
                target = reward
            current_q = current_q_values[action]
            current_q_values[action] = (
                1 - learning_rate
            ) * current_q + learning_rate * target

            current_q = current_q_values[action]
            X_train.append(observation)
            y_train.append(current_q)
            total_reward += reward

            # print('current q:',current_q)
            # print("current q for action {}: {}".format(action, current_q[action]))

            wandb.log({f"Q-value for selected action": current_q[action]})
            wandb.log({"Q-value variance for each action": np.var(current_q_values)})

        history = model.fit(
            np.array(X_train), 
            np.array(y_train), 
            verbose=0, 
            shuffle=True,
            batch_size=config.batch_size,
            # callbacks=[WandbCallback()]

        )
        loss = history.history["loss"][0]
        losses.append(loss)
        total_rewards.append(total_reward)
        print("-"*100)
        print("Loss: ", loss)

        episode_loss += history.history["loss"][0]

        wandb.log({"Loss per episode": episode_loss})

    def save_losses(self):
        # Save the losses and rewards to a CSV file as columns using pandas
        df = pd.DataFrame({"loss": losses, "reward": total_rewards})
        df.to_csv("losses_rewards.csv", index=False)


    def start_run(self):
        run = wandb.init(
                    project="sunburst-maze",
                    config={
                        "learning_rate": lr,
                        # "epochs": episodes,
                        "batch_size":256,
                        "loss_function": loss_function,
                        "architecture": "Dense",
                        "optimizer": optimizer,
                    }
                )

        config = wandb.config
        print("Wandb run initialized.")
        return run, config
    
    def end_run(self, run):
        run.finish()
        print("Wandb run finished.")