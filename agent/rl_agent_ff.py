import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

losses = []
lossfunction = 

class RLAgentFF:

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
    
    def train(self, episodes=1000, batch_size=32, discount_factor=0.99):
        
        pass
    
            