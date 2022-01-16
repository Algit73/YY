from datetime import datetime
from utils import TradingGraph
from model import CNN_Model
from tensorflow.keras.optimizers import Adam, RMSprop
from collections import deque
import random
import numpy as np
import pandas as pd
import copy
import os
from icecream import ic
from tensorflow.keras import backend as K
from Enviorments import *


class PPO_Agent:
    # A custom Bitcoin trading agent
    def __init__(self, env: Env, lookback_window_size=50, learning_rate=0.00005, epochs=1, optimizer=Adam, batch_size=32, state_size=10):
        self.lookback_window_size = lookback_window_size

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = env.Action_space
        self.action_space_size = len(self.action_space)

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"
        ic(self.log_name)

        # State size contains Market+Orders history for the last lookback_window_size steps
        # 10 standard information +9 indicators
        # Original state:
        # self.state_size = (lookback_window_size, 10+9) # 10 standard information +9 indicators

        self.state_size = (lookback_window_size, state_size)

        # Neural Networks part bellow
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create shared Actor-Critic network model
        self.Actor = self.Critic = CNN_Model(input_shape=self.state_size,
                                             action_space=len(
                                                 self.action_space),
                                             learning_rate=self.learning_rate,
                                             optimizer=self.optimizer,
                                             critic_loss="mse",
                                             actor_loss=self.actor_ppo_loss)

        # Variables to keep the folder name and file name
        self.file_name = ""

        self.replay_count = 0

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d,
                  nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        # Compute advantages
        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(
            states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(
            states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.replay_count += 1
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state, is_train):
        # Use the network to predict the next action to take, using the model

        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        # if  np.isnan(prediction).any():
        #   prediction = np.ones(len(prediction))
        # TODO: Adding Exploration/Exploitation
        action_exploited = np.random.choice(self.action_space, p=prediction)
        if (is_train):
            # ic('Train')
            if action_exploited != 2:
                action = np.random.choice(
                    [action_exploited, 0, 2], 1, p=[.4, 0.2, 0.4])
            else:
                action = action_exploited
            # ic(action_exploited)
            # ic(action)
            return action, prediction
        else:
            # ic('Test')
            return action_exploited, prediction

    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        self.file_name = f"{self.log_name}/{score}_{name}"
        self.Actor.Actor.save_weights(
            f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(
            f"{self.log_name}/{score}_{name}_Critic.h5")

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                atgumets = ""
                for arg in args:
                    atgumets += f", {arg}"
                log.write(f"{current_time}{atgumets}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(
            os.path.join(folder, f"{name}_Critic.h5"))

    def actor_ppo_loss(self, y_true, y_pred):
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:,
                                                                      1:1+self.action_space_size], y_true[:, 1+self.action_space_size:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001

        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))

        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING,
                    max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)

        total_loss = actor_loss - entropy

        return total_loss
