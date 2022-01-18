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
from Agents import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_agent(env, agent: PPO_Agent, visualize=False, explore_mode=True,
                train_episodes=50, training_batch_size=500):

    # save n recent (maxlen=n) episodes net worth
    total_average = deque(maxlen=10)
    best_average = 0  # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state, explore_mode)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        a_loss, c_loss = agent.replay(
            states, actions, rewards, predictions, dones, next_states)

        total_average.append(env.net_worth)
        average = np.average(total_average)

        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(
            episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average), args=[
                           episode, average, env.episode_orders, a_loss, c_loss])
            elif episode % 5 == 0:
                agent.save(score="episode_{:.2f}".format(episode), args=[
                           episode, average, env.episode_orders, a_loss, c_loss])

            agent.save()

    agent.end_training_log()


def test_agent(env: CustomEnv, agent: PPO_Agent, visualize=True, test_episodes=10, explore_mode=False,
               folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            action, prediction = agent.act(state, explore_mode)
            state, reward, done = env.step(action)
            is_last_step = (episode == (test_episodes-1))
            env.render(visualize and is_last_step, env.current_step == env.end_step)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance:
                    # calculate episode count where we had negative profit through episode
                    no_profit_episodes += 1
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(
                    episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                break

    print("average {} episodes agent net_worth: {}, orders: {}".format(
        test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))


if __name__ == "__main__":

    # Reading a time-based dataframe with/without indicators
    df = pd.read_csv(
        './Binance_BTCUSDT_Multi_Time_Frame_Interpolated.csv')  # [::-1]

    lookback_window_size = 12
    test_window = 24 * 20    # 30 days

    # Training Section:
    agent = PPO_Agent(CustomEnv, lookback_window_size=lookback_window_size,
                      learning_rate=0.0001, epochs=5, optimizer=Adam, batch_size=24, state_size=10+3)

    TESTING = True

    if not TESTING:
        train_df = df[:-test_window-lookback_window_size]
        train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
        train_agent(train_env, agent, visualize=False, explore_mode=False,
                    train_episodes=3000, training_batch_size=2000)

    else:
        # test_df = df[-test_window:-test_window + 180]
        test_df = df[-test_window:]
        test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size,
                             Show_reward=True, Show_indicators=True)
        test_agent(test_env, agent, visualize=True, test_episodes=3, explore_mode=False,
                   folder="2022_01_17_19_37_Crypto_trader", name="1281.12_Crypto_trader", comment="")
