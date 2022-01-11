import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras import backend as K
from icecream import ic


class Shared_Model:
    def __init__(self, input_shape, action_space, learning_rate, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        
        
        ### CNN Model
        V = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
        V = MaxPooling1D(pool_size=2)(V)
        V = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(V)
        V = MaxPooling1D(pool_size=2)(V)
        V = Flatten()(V)

        
        value = Dense(1, activation=None)(V)
        self.Critic = Model(inputs=X_input, outputs = value) # value --> X
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=learning_rate))

        #######

        ## Actor model
        ### CNN Model
        dropout_layer = Dropout(.1)
        A = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
        A = MaxPooling1D(pool_size=2)(A)
        A = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(A)
        A = MaxPooling1D(pool_size=2)(A)
        A = Flatten()(A)

        output = Dense(self.action_space, activation="softmax")(A) # A --> X
        

        self.Actor = Model(inputs = X_input, outputs = output) 
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        

    def ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
