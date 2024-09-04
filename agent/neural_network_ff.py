import numpy as np
import random as rd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

losses = []


class NeuralNetworkFF:

    def agent(self, state_shape, action_shape, learning_rate=0.001):

        model = models.Sequential()

        model.add(layers.Dense(128, activation="relu", input_shape=state_shape))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(128, activation="relu"))
        model.add(layers.Dense(action_shape[0], activation="softmax"))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizers.Adam(learning_rate),
            metrics=["accuracy"],
        )

        return model

    def state_to_input(state: tuple):

        state_orientation = np.digitize(state[0], bins=[0, 1, 2, 3])

        state_next_to_wall = np.digitize(state[1], bins=[0, 1])

        return np.array([state_orientation, state_next_to_wall])
    
    def train(self, replay_memory, model, target_model, done, episodes=1000, batch_size=32, discount_factor=0.9, learning_rate=0.1):
        
        if len(replay_memory) < 500:
            return
        
        mini_batch = rd.sample(replay_memory, batch_size)

        states = np.array([batch[0] for batch in mini_batch])

        current_q_values = model.predict(states)
        new_states = np.array([batch[3] for batch in mini_batch])
        future_q_values = target_model.predict(new_states)

        X_train = []
        y_train = []

        for i, [observation, action, reward, new_observation, done] in enumerate(mini_batch):
            if not done:
                max_future_q = np.max(future_q_values[i])
                target = reward + discount_factor * max_future_q
            else:
                target = reward

            current_q = current_q_values[action]
            current_q_values[action] = (1-learning_rate) * current_q + learning_rate * target

            X_train.append(observation)
            y_train.append(current_q)

        history = model.fit(np.array(X_train), np.array(y_train), verbose=0, shuffle=True)
        loss = history.history["loss"][0]
        print("Loss: ", loss)

            