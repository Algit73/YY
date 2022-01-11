from datetime import datetime
from utils import TradingGraph
from model import Shared_Model
from tensorflow.keras.optimizers import Adam, RMSprop
from collections import deque
import random
import numpy as np
import pandas as pd
import copy
import os
from icecream import ic
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
HOLD, BUY, SELL = 0, 1, 2


class CustomAgent:
    # A custom Bitcoin trading agent
    def __init__(self, lookback_window_size=50, learning_rate=0.00005, epochs=1, optimizer=Adam, batch_size=32, model="", state_size=10):
        self.lookback_window_size = lookback_window_size
        self.model = model

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

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
        self.Actor = self.Critic = Shared_Model(
            input_shape=self.state_size, action_space=self.action_space.shape[0], learning_rate=self.learning_rate, optimizer=self.optimizer)

        # Variables to keep the folder name and file name
        self.folder_name = ""
        self.file_name = ""

        self.replay_count = 0

    def get_folder_name(self):
        ic(self.log_name)
        return self.log_name

    def end_training_log(self):
        with open(self.log_name+"/Parameters.txt", "a+") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training end: {current_date}\n")

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

    def get_file_name(self):
        ic(self.file_name)
        return self.file_name

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(
            os.path.join(folder, f"{name}_Critic.h5"))


class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range  # render range in visualization
        self.Show_reward = Show_reward  # show order reward in rendered visualization
        # show main indicators in rendered visualization
        self.Show_indicators = Show_indicators

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.indicators_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size=0):
        self.visualization = TradingGraph(Render_range=180, Show_reward=self.Show_reward,
                                          Show_indicators=self.Show_indicators)  # init visualization
        # limited orders memory for visualization
        #self.trades = deque(maxlen=self.Render_range)
        self.trades = deque(maxlen=180)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.previous_price = 0
        self.current_price = 0
        self.last_trade_action = 0
        self.action_history = deque(maxlen=2)
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0  # track episode orders count
        self.prev_episode_orders = 0  # track previous episode orders count
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(
                self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append(
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.market_history.append([self.df.loc[current_step, 'open'],
                                        self.df.loc[current_step, 'high'],
                                        self.df.loc[current_step, 'low'],
                                        self.df.loc[current_step, 'close'],
                                        self.df.loc[current_step, 'volume'],
                                        ])

            self.indicators_history.append(
                [
                    self.df.loc[current_step, 'macd_1h'] / 400,
                    self.df.loc[current_step, 'williams_2h'] /
                    self.normalize_value,
                    self.df.loc[current_step, 'williams_4h'] /
                    self.normalize_value,
                    #self.df.loc[current_step, 'williams_8h']/self.normalize_value,
                ])

        state = np.concatenate(
            (self.market_history, self.orders_history), axis=1) / self.normalize_value
        state = np.concatenate((state, self.indicators_history), axis=1)

        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'open'],
                                    self.df.loc[self.current_step, 'high'],
                                    self.df.loc[self.current_step, 'low'],
                                    self.df.loc[self.current_step, 'close'],
                                    self.df.loc[self.current_step, 'volume'],
                                    ])

        self.indicators_history.append(
            [
                self.df.loc[self.current_step, 'macd_1h'] / 400,
                self.df.loc[self.current_step, 'williams_2h'] /
                self.normalize_value,
                self.df.loc[self.current_step, 'williams_4h'] /
                self.normalize_value,
                #self.df.loc[self.current_step, 'williams_8h']/self.normalize_value,
            ])

        obs = np.concatenate(
            (self.market_history, self.orders_history), axis=1) / self.normalize_value
        obs = np.concatenate((obs, self.indicators_history), axis=1)

        return obs

    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to Open
        current_price = self.df.loc[self.current_step, 'open']
        date = self.df.loc[self.current_step, 'date']  # for visualization
        High = self.df.loc[self.current_step, 'high']  # for visualization
        Low = self.df.loc[self.current_step, 'low']  # for visualization

        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance/100:
            self.last_trade_action = action
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'date': date, 'high': High, 'low': Low,
                               'total': self.crypto_bought, 'type': "buy", 'current_price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.last_trade_action = action
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'date': date, 'high': High, 'low': Low,
                               'total': self.crypto_sold, 'type': "sell", 'current_price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Receive calculated reward
        self.current_price = current_price
        reward = self.get_reward()
        self.previous_price = current_price
        self.action_history.append(action)

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

    # Calculate reward
    def get_reward(self):
        self.punish_value += self.net_worth * 0.00005
        #self.punish_value += self.net_worth * self.punishment_cofactor
        #self.punish_value = 0
        if self.episode_orders > 1:  # and self.episode_orders > self.prev_episode_orders:
            # <--Just covers Sell-Buy and Buy-Sell, not others -->
            self.prev_episode_orders = self.episode_orders
            if self.action_history[0] == BUY:
                reward = self.net_worth * \
                    np.log(self.current_price/self.previous_price)
                # self.trades[-2]['total']*self.trades[-1]['current_price']
                #reward -= self.punish_value
                self.punish_value = 0
                if self.episode_orders > self.prev_episode_orders:
                    self.trades[-1]["Reward"] = reward
                return reward
            elif self.action_history[0] == SELL:
                reward = self.net_worth * \
                    np.log(self.previous_price/self.current_price)
                #reward -= self.punish_value
                self.punish_value = 0
                if self.episode_orders > self.prev_episode_orders:
                    self.trades[-1]["Reward"] = reward
                return reward
            else:
                if self.last_trade_action == BUY:
                    reward = self.net_worth * \
                        np.log(self.current_price/self.previous_price)
                else:
                    reward = self.net_worth * \
                        np.log(self.previous_price/self.current_price)
                reward -= self.punish_value
                return reward
        else:
            return 0 - self.punish_value

    # render environment
    def render(self, visualize=False):
        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            # Render the environment to the screen (inside utils.py file)
            img = self.visualization.render(
                self.df.loc[self.current_step], self.net_worth, self.trades)
            return img


def train_agent(env, agent: CustomAgent, visualize=False, explore_mode=True,
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


def test_agent(env, agent, visualize=True, test_episodes=10, explore_mode=False,
               folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize and (episode == (test_episodes-1)))
            action, prediction = agent.act(state, explore_mode)
            state, reward, done = env.step(action)
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
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(
            f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(
            f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')


if __name__ == "__main__":

    # Reading a time-based dataframe with/without indicators
    # df = pd.read_csv('./Binance_BTCUSDT_1h_Base_MACD_PSAR_ATR_BB_ADX_RSI_ICHI_KC_Williams_Cnst_Interpolated.csv')  # [::-1]
    df = pd.read_csv(
        './Binance_BTCUSDT_Multi_Time_Frame_Interpolated.csv')  # [::-1]
    # df = pd.read_csv('./Binance_BTCUSDT_15m.csv')
    #df = df.sort_values('Date').reset_index(drop=True)
    # df = AddIndicators(df)  # insert indicators to df
    # df = df.round(2)   # two digit precision

    lookback_window_size = 12
    test_window = 24 * 20    # 30 days

    # Training Section:
    train_df = df[:-test_window-lookback_window_size]
    agent = CustomAgent(lookback_window_size=lookback_window_size,
                        learning_rate=0.0001, epochs=5, optimizer=Adam, batch_size=24, model="Dense", state_size=10+3)

    train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size)
    train_agent(train_env, agent, visualize=False, explore_mode=False,
                train_episodes=3000, training_batch_size=2000)

    # Testing Section:
    # test_df = df[-test_window:-test_window + 180]
    test_df = df[-test_window:]
    ic(test_df[['open', 'close']])   # Depicting the specified Time-period
    # test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size,
    #                      Show_reward=True, Show_indicators=True)
    # test_agent(test_env, agent, visualize=False, test_episodes=30,explore_mode=False,
    #            folder="2022_01_10_18_14_Crypto_trader", name="1472.41_Crypto_trader", comment="")
