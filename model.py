import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D

# usually using this for fastest performance
tf.compat.v1.disable_eager_execution()


class CNN_Model:
    def __init__(self, input_shape, action_space, learning_rate, optimizer, critic_loss, actor_loss):

        X_input = Input(input_shape)
        self.action_space = action_space

        # CNN Model
        V = Conv1D(filters=64, kernel_size=6, padding="same",
                   activation="tanh")(X_input)
        V = MaxPooling1D(pool_size=2)(V)
        V = Conv1D(filters=32, kernel_size=3,
                   padding="same", activation="tanh")(V)
        V = MaxPooling1D(pool_size=2)(V)
        V = Flatten()(V)

        value = Dense(1, activation=None)(V)
        self.Critic = Model(inputs=X_input, outputs=value)  # value --> X
        self.Critic.compile(loss=critic_loss, optimizer=optimizer(
            learning_rate=learning_rate))

        #######

        # Actor model
        # CNN Model
        A = Conv1D(filters=64, kernel_size=6, padding="same",
                   activation="tanh")(X_input)
        A = MaxPooling1D(pool_size=2)(A)
        A = Conv1D(filters=32, kernel_size=3,
                   padding="same", activation="tanh")(A)
        A = MaxPooling1D(pool_size=2)(A)
        A = Flatten()(A)

        output = Dense(self.action_space, activation="softmax")(A)

        self.Actor = Model(inputs=X_input, outputs=output)
        self.Actor.compile(loss=actor_loss, optimizer=optimizer(
            learning_rate=learning_rate))

    def actor_predict(self, state):
        return self.Actor.predict(state)

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
