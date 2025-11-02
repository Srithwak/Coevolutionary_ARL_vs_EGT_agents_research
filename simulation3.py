import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Suppress TensorFlow warnings
tf.get_logger().setLevel("ERROR")


class MarketData:
    """
    Loads, processes, and provides real OHLCV market data.
    """

    # (This class is unchanged)
    def __init__(
        self,
        ohlcv_filename,
        volatility_window=20,
        momentum_window=5,
        rsi_window=14,
    ):
        self.filename = ohlcv_filename
        if not os.path.exists(self.filename):
            print(f"Error: Data file not found at '{self.filename}'")
            print(
                "Please provide a valid CSV file with 'High', 'Low', 'Volume' columns."
            )
            raise FileNotFoundError(f"Missing data file: {self.filename}")

        print(f"Loading data from '{self.filename}'...")
        self.data = pd.read_csv(self.filename)

        print("Validating and converting data types...")
        required_cols = ["High", "Low", "Volume"]

        for col in required_cols:
            if col not in self.data.columns:
                print(f"Error: CSV must contain '{col}' column.")
                raise ValueError(f"Missing required column: {col}")
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        original_len = len(self.data)
        self.data.dropna(subset=required_cols, inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        new_len = len(self.data)

        if new_len < original_len:
            print(f"Dropped {original_len - new_len} rows containing non-numeric data.")

        self.data["mid_price"] = (self.data["High"] + self.data["Low"]) / 2.0
        self.max_steps = len(self.data)
        if self.max_steps < volatility_window:
            print(
                f"Warning: Data length ({self.max_steps}) is shorter than feature window ({volatility_window})."
            )

        print("Calculating technical features (volatility, momentum, RSI, etc.)...")
        self.data["returns"] = self.data["mid_price"].pct_change()
        self.data["volatility"] = (
            self.data["returns"].rolling(window=volatility_window).std().fillna(0)
        )
        self.data["price_change"] = self.data["mid_price"].diff()
        self.data["momentum"] = (
            self.data["price_change"].rolling(window=momentum_window).mean().fillna(0)
        )
        rolling_vol_mean = (
            self.data["Volume"].rolling(window=volatility_window).mean().fillna(1)
        )
        self.data["norm_volume"] = (
            self.data["Volume"] / (rolling_vol_mean + 1e-6)
        ).fillna(0)

        delta = self.data["mid_price"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
        rs = gain / (loss + 1e-6)
        self.data["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

        print(
            f"Successfully loaded and processed '{self.filename}' with {self.max_steps} steps."
        )

    def get_market_state(self, step):
        if step >= self.max_steps:
            return None, None, None, None, None
        price = self.data.loc[step, "mid_price"]
        volatility = self.data.loc[step, "volatility"]
        momentum = self.data.loc[step, "momentum"]
        norm_volume = self.data.loc[step, "norm_volume"]
        rsi = self.data.loc[step, "rsi"]
        return price, volatility, momentum, norm_volume, rsi


class DQNAgent:
    """
    The Double Deep Q-Network (DDQN) agent.
    """

    def __init__(self, state_size, action_size, nn_architecture, drl_params):
        self.state_size = state_size
        self.actions = [(0.1, 0.1), (0.2, 0.2), (0.4, 0.4), (0.3, 0.1), (0.1, 0.3)]
        self.action_size = action_size
        self.inventory = 0
        self.cash = 0
        default_params = {
            "gamma": 0.95,
            "epsilon": 1.0,
            "epsilon_decay": 0.995, # change from .995 to .998
            "epsilon_min": 0.01,
            "learning_rate": 0.001,
            "replay_buffer_maxlen": 2000,
            "batch_size": 32,
        }
        self.drl_params = default_params
        self.drl_params.update(drl_params or {})
        self.replay_buffer = deque(maxlen=self.drl_params["replay_buffer_maxlen"])
        self.gamma = self.drl_params["gamma"]
        self.epsilon = self.drl_params["epsilon"]
        self.epsilon_decay = self.drl_params["epsilon_decay"]
        self.epsilon_min = self.drl_params["epsilon_min"]
        self.learning_rate = self.drl_params["learning_rate"]
        self.batch_size = self.drl_params["batch_size"]
        self.nn_architecture = nn_architecture
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(
            Dense(
                self.nn_architecture[0]["units"],
                input_dim=self.state_size,
                activation=self.nn_architecture[0]["activation"],
            )
        )
        for layer in self.nn_architecture[1:]:
            model.add(Dense(layer["units"], activation=layer["activation"]))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # ---
    # --- MODIFIED: Linear Inventory Normalization ---
    # ---
    # def format_state(
    #     self, inventory, volatility, momentum, norm_volume, rsi, step, max_steps
    # ):
    #     """
    #     Formats the agent's and market's state into a normalized vector for the NN.
    #     """
    #     # Linearly clip inventory. Agent sees -1.0 for -100 or less,
    #     # +1.0 for +100 or more, and a linear value (e.g., 0.5 for +50) in between.
    #     norm_inventory = np.clip(inventory, -100, 100) / 100.0
    #     # --- End Modification ---

    #     norm_time = step / max_steps
    #     norm_rsi = (rsi / 50.0) - 1.0
    #     norm_vol = np.tanh(norm_volume - 1.0)
    #     return np.array(
    #         [[norm_inventory, volatility, momentum, norm_vol, norm_rsi, norm_time]]
    #     )

    def format_state(
        self, inventory, volatility, momentum, norm_volume, rsi, step, max_steps
    ):
        # norm_inventory = np.tanh(inventory / 10.0)  # smooth sensitivity to imbalance
        norm_inventory = np.tanh(inventory / 5.0) # ADDED

        norm_time = step / max_steps
        norm_rsi = (rsi / 50.0) - 1.0
        norm_vol = np.tanh(norm_volume - 1.0)
        return np.array(
            [[norm_inventory, volatility, momentum, norm_vol, norm_rsi, norm_time]]
        )

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.append((state, action_idx, reward, next_state, done))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(range(self.action_size))
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        states = np.squeeze(np.array([item[0] for item in minibatch]))
        actions = np.array([item[1] for item in minibatch])
        rewards = np.array([item[2] for item in minibatch])
        next_states = np.squeeze(np.array([item[3] for item in minibatch]))
        dones = np.array([item[4] for item in minibatch])
        target_q_next = self.target_model.predict(next_states, verbose=0)
        best_actions_next = np.argmax(
            self.model.predict(next_states, verbose=0), axis=1
        )
        target_q_values = target_q_next[range(self.batch_size), best_actions_next]
        targets = rewards + self.gamma * target_q_values * (1 - dones)
        current_q = self.model.predict(states, verbose=0)
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]
        self.model.fit(states, current_q, epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class Market:
    # (This class is unchanged from v7)
    def __init__(self, nn_architecture, drl_params, ohlcv_filename):
        self.market_data = MarketData(ohlcv_filename=ohlcv_filename)
        self.step = 0

        self.state_size = 6
        self.action_size = 5
        self.drl_agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            nn_architecture=nn_architecture,
            drl_params=drl_params,
        )
        self.UPDATE_TARGET_EVERY = 20

        self.egt_strategies = ["aggressive", "passive", "random", "momentum"]
        self.egt_proportions = np.array([0.25, 0.25, 0.25, 0.25])
        self.egt_total_payoffs = np.zeros(len(self.egt_strategies))
        self.egt_total_trades = np.zeros(len(self.egt_strategies))
        self.N_EGT_AGENTS_PER_STEP = 15
        self.EVOLVE_EVERY = 50

        # Using the Quadratic Penalty from v7
        self.transaction_cost_per_trade = 0.01
        self.inventory_penalty_factor = 0.001

        self.history = {
            "step": [],
            "mid_price": [],
            "drl_profit": [],
            "drl_inventory": [],
            "chosen_spread_width": [],
            "epsilon": [],
            "reward": [],
            "egt_prop_aggressive": [],
            "egt_prop_passive": [],
            "egt_prop_random": [],
            "egt_prop_momentum": [],
        }
        self.last_drl_value = 0

    def get_egt_action(self, strategy_name, mid_price, momentum):
        action_type = random.choice(["buy", "sell"])
        if strategy_name == "aggressive":
            price_offset = random.uniform(0.02, 0.08)
        elif strategy_name == "passive":
            price_offset = random.uniform(0.05, 0.15)
        elif strategy_name == "momentum":
            if momentum > 0:
                action_type = "buy"
            elif momentum < 0:
                action_type = "sell"
            price_offset = random.uniform(0.04, 0.12)
        else:
            price_offset = random.uniform(0.01, 0.5)
        return (
            (action_type, mid_price + price_offset)
            if action_type == "buy"
            else (action_type, mid_price - price_offset)
        )

    def evolve_population(self):
        fitness = self.egt_total_payoffs / (self.egt_total_trades + 1e-6)
        positive_fitness = fitness - np.min(fitness) + 1
        avg_fitness = np.dot(self.egt_proportions, positive_fitness)
        if avg_fitness == 0:
            return
        self.egt_proportions = self.egt_proportions * (positive_fitness / avg_fitness)
        mutation = 0.001
        self.egt_proportions = self.egt_proportions * (1 - mutation) + (
            mutation / len(self.egt_strategies)
        )
        self.egt_proportions /= np.sum(self.egt_proportions)
        self.egt_total_payoffs.fill(0)
        self.egt_total_trades.fill(0)

    # def run_step(self):
    #     market_data_tuple = self.market_data.get_market_state(self.step)
    #     mid_price, volatility, momentum, norm_volume, rsi = market_data_tuple
    #     if mid_price is None:
    #         return False

    #     state = self.drl_agent.format_state(
    #         self.drl_agent.inventory,
    #         volatility,
    #         momentum,
    #         norm_volume,
    #         rsi,
    #         self.step,
    #         self.market_data.max_steps,
    #     )
    #     action_idx = self.drl_agent.choose_action(state)
    #     bid_spread, ask_spread = self.drl_agent.actions[action_idx]
    #     drl_bid_price = mid_price - bid_spread
    #     drl_ask_price = mid_price + ask_spread

    #     step_payoffs = np.zeros(len(self.egt_strategies))
    #     step_trades = np.zeros(len(self.egt_strategies))
    #     drl_trades_this_step = 0

    #     sampled_agents = np.random.choice(
    #         self.egt_strategies, size=self.N_EGT_AGENTS_PER_STEP, p=self.egt_proportions
    #     )
    #     for strategy_name in sampled_agents:
    #         egt_action, egt_price = self.get_egt_action(
    #             strategy_name, mid_price, momentum
    #         )
    #         strat_idx = self.egt_strategies.index(strategy_name)
    #         if egt_action == "sell" and egt_price <= drl_bid_price:
    #             self.drl_agent.inventory += 1
    #             self.drl_agent.cash -= drl_bid_price
    #             step_payoffs[strat_idx] += drl_bid_price
    #             step_trades[strat_idx] += 1
    #             drl_trades_this_step += 1
    #         elif egt_action == "buy" and egt_price >= drl_ask_price:
    #             self.drl_agent.inventory -= 1
    #             self.drl_agent.cash += drl_ask_price
    #             step_payoffs[strat_idx] -= drl_ask_price
    #             step_trades[strat_idx] += 1
    #             drl_trades_this_step += 1
    #     self.egt_total_payoffs += step_payoffs
    #     self.egt_total_trades += step_trades

    #     inventory_after_trades = self.drl_agent.inventory

    #     # Using the Quadratic Penalty from v7
    #     current_portfolio_value = self.drl_agent.cash + (
    #         inventory_after_trades * mid_price
    #     )
    #     reward = current_portfolio_value - self.last_drl_value
    #     reward -= drl_trades_this_step * self.transaction_cost_per_trade
    #     reward -= (inventory_after_trades**2) * self.inventory_penalty_factor

    #     next_data_tuple = self.market_data.get_market_state(self.step + 1)
    #     next_price, next_vol, next_mom, next_norm_vol, next_rsi = next_data_tuple
    #     done = next_price is None

    #     if done:
    #         next_state = state
    #     else:
    #         next_state = self.drl_agent.format_state(
    #             inventory_after_trades,
    #             next_vol,
    #             next_mom,
    #             next_norm_vol,
    #             next_rsi,
    #             self.step + 1,
    #             self.market_data.max_steps,
    #         )

    #     self.drl_agent.remember(state, action_idx, reward, next_state, done)
    #     self.drl_agent.replay()
    #     self.last_drl_value = current_portfolio_value

    #     if self.step % self.EVOLVE_EVERY == 0:
    #         self.evolve_population()
    #     if self.step % self.UPDATE_TARGET_EVERY == 0:
    #         self.drl_agent.update_target_model()

    #     self.history["step"].append(self.step)
    #     self.history["mid_price"].append(mid_price)
    #     self.history["drl_profit"].append(current_portfolio_value)
    #     self.history["drl_inventory"].append(inventory_after_trades)
    #     self.history["chosen_spread_width"].append(bid_spread + ask_spread)
    #     self.history["epsilon"].append(self.drl_agent.epsilon)
    #     self.history["reward"].append(reward)
    #     for i, strat in enumerate(self.egt_strategies):
    #         self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

    #     self.step += 1
    #     return True

    # def run_step(self):
    #     market_data_tuple = self.market_data.get_market_state(self.step)
    #     mid_price, volatility, momentum, norm_volume, rsi = market_data_tuple
    #     if mid_price is None:
    #         return False

    #     # --- 1. DRL Agent chooses action ---
    #     state = self.drl_agent.format_state(
    #         self.drl_agent.inventory,
    #         volatility,
    #         momentum,
    #         norm_volume,
    #         rsi,
    #         self.step,
    #         self.market_data.max_steps,
    #     )
    #     action_idx = self.drl_agent.choose_action(state)
    #     bid_spread, ask_spread = self.drl_agent.actions[action_idx]

    #     # --- 2. Dynamic spread adjustment (inventory control heuristic) ---
    #     if self.drl_agent.inventory > 0:
    #         # Long inventory → encourage selling
    #         bid_spread *= 1.2  # discourage buying
    #         ask_spread *= 0.8  # encourage selling
    #     elif self.drl_agent.inventory < 0:
    #         # Short inventory → encourage buying
    #         bid_spread *= 0.8
    #         ask_spread *= 1.2

    #     drl_bid_price = mid_price - bid_spread
    #     drl_ask_price = mid_price + ask_spread

    #     # --- 3. Simulate EGT population trading ---
    #     step_payoffs = np.zeros(len(self.egt_strategies))
    #     step_trades = np.zeros(len(self.egt_strategies))
    #     drl_trades_this_step = 0

    #     sampled_agents = np.random.choice(
    #         self.egt_strategies, size=self.N_EGT_AGENTS_PER_STEP, p=self.egt_proportions
    #     )
    #     for strategy_name in sampled_agents:
    #         egt_action, egt_price = self.get_egt_action(strategy_name, mid_price, momentum)
    #         strat_idx = self.egt_strategies.index(strategy_name)

    #         if egt_action == "sell" and egt_price <= drl_bid_price:
    #             self.drl_agent.inventory += 1
    #             self.drl_agent.cash -= drl_bid_price
    #             step_payoffs[strat_idx] += drl_bid_price
    #             step_trades[strat_idx] += 1
    #             drl_trades_this_step += 1
    #         elif egt_action == "buy" and egt_price >= drl_ask_price:
    #             self.drl_agent.inventory -= 1
    #             self.drl_agent.cash += drl_ask_price
    #             step_payoffs[strat_idx] -= drl_ask_price
    #             step_trades[strat_idx] += 1
    #             drl_trades_this_step += 1

    #     self.egt_total_payoffs += step_payoffs
    #     self.egt_total_trades += step_trades

    #     # --- 4. Reward calculation ---
    #     inv = self.drl_agent.inventory
    #     current_value = self.drl_agent.cash + (inv * mid_price)
    #     reward = current_value - self.last_drl_value  # base profit/loss

    #     # Inventory penalties — immediate mean-reversion feedback
    #     reward -= 0.005 * abs(inv)         # linear penalty
    #     reward -= 0.001 * (inv ** 2)       # quadratic penalty
    #     reward -= drl_trades_this_step * self.transaction_cost_per_trade

    #     # Update reward baseline AFTER all penalties
    #     self.last_drl_value = current_value

    #     # --- 5. Next state & training ---
    #     next_data_tuple = self.market_data.get_market_state(self.step + 1)
    #     next_price, next_vol, next_mom, next_norm_vol, next_rsi = next_data_tuple
    #     done = next_price is None

    #     if done:
    #         next_state = state
    #     else:
    #         next_state = self.drl_agent.format_state(
    #             inv,
    #             next_vol,
    #             next_mom,
    #             next_norm_vol,
    #             next_rsi,
    #             self.step + 1,
    #             self.market_data.max_steps,
    #         )

    #     self.drl_agent.remember(state, action_idx, reward, next_state, done)
    #     self.drl_agent.replay()

    #     if self.step % self.EVOLVE_EVERY == 0:
    #         self.evolve_population()
    #     if self.step % self.UPDATE_TARGET_EVERY == 0:
    #         self.drl_agent.update_target_model()

    #     # --- 6. Log data for plots ---
    #     self.history["step"].append(self.step)
    #     self.history["mid_price"].append(mid_price)
    #     self.history["drl_profit"].append(current_value)
    #     self.history["drl_inventory"].append(inv)
    #     self.history["chosen_spread_width"].append(bid_spread + ask_spread)
    #     self.history["epsilon"].append(self.drl_agent.epsilon)
    #     self.history["reward"].append(reward)
    #     for i, strat in enumerate(self.egt_strategies):
    #         self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

    #     self.step += 1
    #     return True



    # CHANGED
    # def run_step(self):
    #     market_data_tuple = self.market_data.get_market_state(self.step)
    #     mid_price, volatility, momentum, norm_volume, rsi = market_data_tuple
    #     if mid_price is None:
    #         return False

    #     # --- 1. DRL Agent chooses action ---
    #     state = self.drl_agent.format_state(
    #         self.drl_agent.inventory,
    #         volatility,
    #         momentum,
    #         norm_volume,
    #         rsi,
    #         self.step,
    #         self.market_data.max_steps,
    #     )
    #     action_idx = self.drl_agent.choose_action(state)
    #     bid_spread, ask_spread = self.drl_agent.actions[action_idx]

    #     # --- 2. Dynamic spread adjustment (inventory control heuristic) ---
    #     inv_bias = np.clip(abs(self.drl_agent.inventory) / 5.0, 1.0, 3.0) # ADDED
    #     # inv_bias = np.clip(abs(self.drl_agent.inventory) / 10.0, 1.0, 2.0)
    #     if self.drl_agent.inventory > 0:
    #         # Long inventory → encourage selling
    #         bid_spread *= inv_bias      # discourage buying
    #         ask_spread /= inv_bias      # encourage selling
    #     elif self.drl_agent.inventory < 0:
    #         # Short inventory → encourage buying
    #         bid_spread /= inv_bias      # encourage buying
    #         ask_spread *= inv_bias      # discourage selling

    #     drl_bid_price = mid_price - bid_spread
    #     drl_ask_price = mid_price + ask_spread

    #     # --- 3. Simulate EGT population trading ---
    #     step_payoffs = np.zeros(len(self.egt_strategies))
    #     step_trades = np.zeros(len(self.egt_strategies))
    #     drl_trades_this_step = 0

    #     sampled_agents = np.random.choice(
    #         self.egt_strategies, size=self.N_EGT_AGENTS_PER_STEP, p=self.egt_proportions
    #     )
    #     for strategy_name in sampled_agents:
    #         egt_action, egt_price = self.get_egt_action(strategy_name, mid_price, momentum)
    #         strat_idx = self.egt_strategies.index(strategy_name)

    #         if egt_action == "sell" and egt_price <= drl_bid_price:
    #             self.drl_agent.inventory += 1
    #             self.drl_agent.cash -= drl_bid_price
    #             step_payoffs[strat_idx] += drl_bid_price
    #             step_trades[strat_idx] += 1
    #             drl_trades_this_step += 1
    #         elif egt_action == "buy" and egt_price >= drl_ask_price:
    #             self.drl_agent.inventory -= 1
    #             self.drl_agent.cash += drl_ask_price
    #             step_payoffs[strat_idx] -= drl_ask_price
    #             step_trades[strat_idx] += 1
    #             drl_trades_this_step += 1

    #     self.egt_total_payoffs += step_payoffs
    #     self.egt_total_trades += step_trades

    #     # # --- 4. Reward calculation ---
    #     # inv = self.drl_agent.inventory
    #     # current_value = self.drl_agent.cash + (inv * mid_price)
    #     # reward = current_value - self.last_drl_value  # base profit/loss

    #     # # Inventory penalties — immediate mean-reversion feedback
    #     # reward -= 0.005 * abs(inv)          # linear penalty
    #     # reward -= 0.001 * (inv ** 2)        # quadratic penalty
    #     # reward -= drl_trades_this_step * self.transaction_cost_per_trade

    #     # # --- NEW: Mean-reversion incentive ---
    #     # if abs(inv) > 0:
    #     #     prev_inv = self.history["drl_inventory"][-1] if self.history["drl_inventory"] else 0
    #     #     if abs(inv) < abs(prev_inv):  # reduced exposure
    #     #         reward += 0.003 * (abs(prev_inv) - abs(inv))

    #     # # Update reward baseline AFTER all penalties and bonuses
    #     # self.last_drl_value = current_value


    #     # # --- 4. Reward calculation ---
    #     # inv = self.drl_agent.inventory
    #     # current_value = self.drl_agent.cash + (inv * mid_price)
    #     # reward = current_value - self.last_drl_value  # base profit/loss since last step

    #     # # --- Inventory penalties: discourage imbalance ---
    #     # reward -= 0.005 * abs(inv)           # linear penalty for holding inventory
    #     # reward -= 0.001 * (inv ** 2)         # stronger penalty for large exposure
    #     # reward -= drl_trades_this_step * self.transaction_cost_per_trade  # transaction costs

    #     # # --- Mean-reversion incentive: reward reducing exposure magnitude ---
    #     # if abs(inv) > 0:
    #     #     prev_inv = self.history["drl_inventory"][-1] if self.history["drl_inventory"] else 0
    #     #     if abs(inv) < abs(prev_inv):
    #     #         reward += 0.003 * (abs(prev_inv) - abs(inv))  # reward for moving toward 0

    #     # # --- Zero-balance incentive: reward staying centered near 0 inventory ---
    #     # reward += 0.002 * (1 - min(abs(inv) / 20.0, 1.0))  # peaks at 0, fades by ±20

    #     # # --- Update reward baseline AFTER all penalties and bonuses ---
    #     # self.last_drl_value = current_value

    #     # --- 4. Reward calculation ---
    #     inv = self.drl_agent.inventory
    #     current_value = self.drl_agent.cash + (inv * mid_price)
    #     reward = current_value - self.last_drl_value  # base profit/loss since last step

    #     # --- Inventory penalties: discourage imbalance ---
    #     reward -= 0.005 * abs(inv)            # linear penalty for any exposure
    #     reward -= 0.001 * (inv ** 2)          # stronger penalty for large inventory
    #     reward -= drl_trades_this_step * self.transaction_cost_per_trade  # transaction costs

    #     # --- Mean-reversion incentive: reward reductions toward 0 ---
    #     if abs(inv) > 0:
    #         prev_inv = self.history["drl_inventory"][-1] if self.history["drl_inventory"] else 0
    #         if abs(inv) < abs(prev_inv):
    #             reward += 0.003 * (abs(prev_inv) - abs(inv))  # reward for moving closer to neutral

    #     # --- Zero-balance incentive: small bonus near zero inventory ---
    #     reward += 0.002 * (1 - min(abs(inv) / 20.0, 1.0))  # peaks at 0, fades by ±20

    #     # --- NEW: Symmetry correction (counteracts persistent bias) ---
    #     reward -= 0.002 * inv  # pushes negative inventory upward, positive downward

    #     # --- Update reward baseline AFTER all penalties and bonuses ---
    #     self.last_drl_value = current_value







    #     # --- 5. Next state & training ---
    #     next_data_tuple = self.market_data.get_market_state(self.step + 1)
    #     next_price, next_vol, next_mom, next_norm_vol, next_rsi = next_data_tuple
    #     done = next_price is None

    #     if done:
    #         next_state = state
    #     else:
    #         next_state = self.drl_agent.format_state(
    #             inv,
    #             next_vol,
    #             next_mom,
    #             next_norm_vol,
    #             next_rsi,
    #             self.step + 1,
    #             self.market_data.max_steps,
    #         )

    #     self.drl_agent.remember(state, action_idx, reward, next_state, done)
    #     self.drl_agent.replay()

    #     # --- 6. Population evolution & target updates ---
    #     if self.step % self.EVOLVE_EVERY == 0:
    #         self.evolve_population()
    #     if self.step % self.UPDATE_TARGET_EVERY == 0:
    #         self.drl_agent.update_target_model()

    #     # --- 7. Logging for analysis ---
    #     self.history["step"].append(self.step)
    #     self.history["mid_price"].append(mid_price)
    #     self.history["drl_profit"].append(current_value)
    #     self.history["drl_inventory"].append(inv)
    #     self.history["chosen_spread_width"].append(bid_spread + ask_spread)
    #     self.history["epsilon"].append(self.drl_agent.epsilon)
    #     self.history["reward"].append(reward)
    #     for i, strat in enumerate(self.egt_strategies):
    #         self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

    #     self.step += 1
    #     return True

    def run_step(self):
        market_data_tuple = self.market_data.get_market_state(self.step)
        mid_price, volatility, momentum, norm_volume, rsi = market_data_tuple
        if mid_price is None:
            return False

        # --- 1. DRL Agent chooses action ---
        state = self.drl_agent.format_state(
            self.drl_agent.inventory,
            volatility,
            momentum,
            norm_volume,
            rsi,
            self.step,
            self.market_data.max_steps,
        )
        action_idx = self.drl_agent.choose_action(state)
        bid_spread, ask_spread = self.drl_agent.actions[action_idx]

        # --- 2. Dynamic spread adjustment (tighter inventory control) ---
        inv_bias = np.clip(abs(self.drl_agent.inventory) / 5.0, 1.0, 3.0)
        if self.drl_agent.inventory > 0:
            bid_spread *= inv_bias      # discourage buying
            ask_spread /= inv_bias      # encourage selling
        elif self.drl_agent.inventory < 0:
            bid_spread /= inv_bias      # encourage buying
            ask_spread *= inv_bias      # discourage selling

        drl_bid_price = mid_price - bid_spread
        drl_ask_price = mid_price + ask_spread

        # --- 3. Simulate EGT population trading ---
        step_payoffs = np.zeros(len(self.egt_strategies))
        step_trades = np.zeros(len(self.egt_strategies))
        drl_trades_this_step = 0
        trade_profit_total = 0.0  # accumulate spread capture

        sampled_agents = np.random.choice(
            self.egt_strategies, size=self.N_EGT_AGENTS_PER_STEP, p=self.egt_proportions
        )
        for strategy_name in sampled_agents:
            egt_action, egt_price = self.get_egt_action(strategy_name, mid_price, momentum)
            strat_idx = self.egt_strategies.index(strategy_name)

            if egt_action == "sell" and egt_price <= drl_bid_price:
                self.drl_agent.inventory += 1
                self.drl_agent.cash -= drl_bid_price
                step_payoffs[strat_idx] += drl_bid_price
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1
                trade_profit_total += (mid_price - drl_bid_price)  # profit for buying under mid
            elif egt_action == "buy" and egt_price >= drl_ask_price:
                self.drl_agent.inventory -= 1
                self.drl_agent.cash += drl_ask_price
                step_payoffs[strat_idx] -= drl_ask_price
                step_trades[strat_idx] += 1
                drl_trades_this_step += 1
                trade_profit_total += (drl_ask_price - mid_price)  # profit for selling above mid

        self.egt_total_payoffs += step_payoffs
        self.egt_total_trades += step_trades

        # --- 4. Reward calculation ---
        inv = self.drl_agent.inventory
        current_value = self.drl_agent.cash + (inv * mid_price)
        reward = current_value - self.last_drl_value  # base PnL change

        # Inventory penalties (weakened)
        reward -= 0.002 * abs(inv)
        reward -= 0.0005 * (inv ** 2)
        reward -= drl_trades_this_step * self.transaction_cost_per_trade

        # Mean-reversion incentive (reward for reducing exposure)
        if abs(inv) > 0:
            prev_inv = self.history["drl_inventory"][-1] if self.history["drl_inventory"] else 0
            if abs(inv) < abs(prev_inv):
                reward += 0.003 * (abs(prev_inv) - abs(inv))

        # Zero-balance incentive
        reward += 0.002 * (1 - min(abs(inv) / 10.0, 1.0))

        # Spread-capture incentive (direct trading profit)
        reward += 0.002 * trade_profit_total

        # Symmetry correction (avoid directional drift)
        reward -= 0.0015 * inv

        self.last_drl_value = current_value

        # --- 5. Next state & training ---
        next_data_tuple = self.market_data.get_market_state(self.step + 1)
        next_price, next_vol, next_mom, next_norm_vol, next_rsi = next_data_tuple
        done = next_price is None
        next_state = state if done else self.drl_agent.format_state(
            inv, next_vol, next_mom, next_norm_vol, next_rsi,
            self.step + 1, self.market_data.max_steps,
        )

        self.drl_agent.remember(state, action_idx, reward, next_state, done)
        self.drl_agent.replay()

        # --- 6. Population evolution & target updates ---
        if self.step % self.EVOLVE_EVERY == 0:
            self.evolve_population()
        if self.step % self.UPDATE_TARGET_EVERY == 0:
            self.drl_agent.update_target_model()

        # --- 7. Logging for analysis ---
        self.history["step"].append(self.step)
        self.history["mid_price"].append(mid_price)
        self.history["drl_profit"].append(current_value)
        self.history["drl_inventory"].append(inv)
        self.history["chosen_spread_width"].append(bid_spread + ask_spread)
        self.history["epsilon"].append(self.drl_agent.epsilon)
        self.history["reward"].append(reward)
        for i, strat in enumerate(self.egt_strategies):
            self.history[f"egt_prop_{strat}"].append(self.egt_proportions[i])

        self.step += 1
        return True




    def run_simulation(self):
        # (This method is unchanged)
        print(f"Starting simulation: 1 DRL agent vs Evolving EGT population.")
        print(f"Total data steps: {self.market_data.max_steps}")
        while self.run_step():
            if self.step % 100 == 0:
                print(
                    f"Step {self.step}/{self.market_data.max_steps}... Epsilon: {self.drl_agent.epsilon:.2f}"
                )
        print("Simulation complete.")
        self.print_summary()
        self.plot_results()

    def print_summary(self):
        # (This method is unchanged)
        if not self.history["drl_profit"]:
            print("Simulation ended before any history was recorded.")
            return
        final_drl_profit = self.history["drl_profit"][-1]
        print("\n" + "=" * 30)
        print("=== FINAL RESULTS ===")
        print(f"DRL Agent Final Profit: ${final_drl_profit:,.2f}")
        print("\nFinal EGT Population Distribution:")
        for i, strat in enumerate(self.egt_strategies):
            print(f"  {strat.capitalize()}: {self.egt_proportions[i]:.3f}")
        print("=" * 30 + "\n")

    # def plot_results(self):
    #     # (This method is unchanged)
    #     STD_FIGSIZE = (8, 6)
    #     STD_DPI = 300
    #     df_history = pd.DataFrame(self.history)
    #     if df_history.empty:
    #         print("History is empty, skipping plotting.")
    #         return
    #     print("PASS 1: Saving individual plot files...")
    #     fig1, ax1 = plt.subplots(figsize=STD_FIGSIZE)
    #     ax1.plot(
    #         df_history["step"],
    #         df_history["drl_profit"],
    #         label="DRL Profit",
    #         color="blue",
    #     )
    #     ax1.set_title("DRL Agent Cumulative Profit Over Time")
    #     ax1.set_xlabel("Time Step")
    #     ax1.set_ylabel("Total Profit ($)")
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("plot_1_drl_profit.png", dpi=STD_DPI)
    #     plt.close(fig1)
    #     fig2, ax2 = plt.subplots(figsize=STD_FIGSIZE)
    #     ax2.plot(
    #         df_history["step"],
    #         df_history["drl_inventory"],
    #         label="DRL Inventory",
    #         color="green",
    #     )
    #     ax2.axhline(
    #         y=0, color="black", linestyle="--", linewidth=1, label="Neutral (0)"
    #     )
    #     ax2.set_title("DRL Agent Inventory Over Time")
    #     ax2.set_xlabel("Time Step")
    #     ax2.set_ylabel("Inventory (Units)")
    #     ax2.legend()
    #     ax2.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("plot_2_drl_inventory.png", dpi=STD_DPI)
    #     plt.close(fig2)
    #     fig3, ax3 = plt.subplots(figsize=STD_FIGSIZE)
    #     egt_labels = [s.capitalize() for s in self.egt_strategies]
    #     egt_data = [df_history[f"egt_prop_{s}"] for s in self.egt_strategies]
    #     ax3.stackplot(df_history["step"], egt_data, labels=egt_labels)
    #     ax3.set_title("EGT Population Evolution (Replicator Dynamics)")
    #     ax3.set_xlabel("Time Step")
    #     ax3.set_ylabel("Proportion of Population")
    #     ax3.set_ylim(0, 1)
    #     ax3.legend(loc="upper left")
    #     plt.tight_layout()
    #     plt.savefig("plot_3_egt_dynamics.png", dpi=STD_DPI)
    #     plt.close(fig3)
    #     fig4, ax4 = plt.subplots(figsize=STD_FIGSIZE)
    #     ax4.plot(
    #         df_history["step"],
    #         df_history["chosen_spread_width"],
    #         label="Chosen Spread Width",
    #         color="red",
    #         alpha=0.6,
    #     )
    #     ax4.set_title("DRL Agent's Chosen Spread Width")
    #     ax4.set_xlabel("Time Step")
    #     ax4.set_ylabel("Spread Width ($)")
    #     ax4.grid(True, alpha=0.3)
    #     ax4_twin = ax4.twinx()
    #     ax4_twin.plot(
    #         df_history["step"],
    #         df_history["epsilon"],
    #         label="Epsilon",
    #         color="grey",
    #         linestyle=":",
    #     )
    #     ax4_twin.set_ylabel("Epsilon")
    #     lines, labels = ax4.get_legend_handles_labels()
    #     lines2, labels2 = ax4_twin.get_legend_handles_labels()
    #     ax4.legend(lines + lines2, labels + labels2, loc="upper right")
    #     plt.tight_layout()
    #     plt.savefig("plot_4_drl_spread.png", dpi=STD_DPI)
    #     plt.close(fig4)
    #     fig5, ax5 = plt.subplots(figsize=STD_FIGSIZE)
    #     df_history["rolling_avg_reward"] = (
    #         df_history["reward"].rolling(window=100, min_periods=1).mean()
    #     )
    #     ax5.plot(
    #         df_history["step"],
    #         df_history["rolling_avg_reward"],
    #         label="Rolling Avg. Reward (100 steps)",
    #         color="purple",
    #     )
    #     ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
    #     ax5.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
    #     ax5.set_xlabel("Time Step")
    #     ax5.set_ylabel("Average Reward")
    #     ax5.legend()
    #     ax5.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("plot_5_rolling_reward.png", dpi=STD_DPI)
    #     plt.close(fig5)
    #     fig6, ax6 = plt.subplots(figsize=STD_FIGSIZE)
    #     halfway_idx = len(df_history["step"]) // 2
    #     exploiting_spreads = df_history["chosen_spread_width"].iloc[halfway_idx:]
    #     if not exploiting_spreads.empty:
    #         spread_counts = pd.Series(exploiting_spreads).value_counts().sort_index()
    #         spread_dist = spread_counts / len(exploiting_spreads)
    #         spread_dist.plot(kind="bar", ax=ax6, color="teal")
    #         ax6.set_title(f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)")
    #     else:
    #         ax6.set_title("DRL Learned Policy (No data)")
    #     ax6.set_xlabel("Chosen Spread Width ($)")
    #     ax6.set_ylabel("Proportion of Actions")
    #     ax6.set_ylim(0, 1)
    #     ax6.tick_params(axis="x", rotation=0)
    #     ax6.grid(axis="y", linestyle="--", alpha=0.5)
    #     plt.tight_layout()
    #     plt.savefig("plot_6_policy_dist.png", dpi=STD_DPI)
    #     plt.close(fig6)
    #     print(f"Successfully saved 6 plots (e.g., 'plot_1_drl_profit.png').")
    #     print("PASS 2: Generating combined plot window...")
    #     fig_display, axes = plt.subplots(3, 2, figsize=(16, 15))
    #     fig_display.suptitle("DRL Agent vs EGT Population Simulation", fontsize=16)
    #     ax_0_0 = axes[0, 0]
    #     ax_0_0.plot(
    #         df_history["step"],
    #         df_history["drl_profit"],
    #         label="DRL Profit",
    #         color="blue",
    #     )
    #     ax_0_0.set_title("DRL Agent Cumulative Profit Over Time")
    #     ax_0_0.set_xlabel("Time Step")
    #     ax_0_0.set_ylabel("Total Profit ($)")
    #     ax_0_0.legend()
    #     ax_0_0.grid(True, alpha=0.3)
    #     ax_0_1 = axes[0, 1]
    #     ax_0_1.plot(
    #         df_history["step"],
    #         df_history["drl_inventory"],
    #         label="DRL Inventory",
    #         color="green",
    #     )
    #     ax_0_1.axhline(
    #         y=0, color="black", linestyle="--", linewidth=1, label="Neutral (0)"
    #     )
    #     ax_0_1.set_title("DRL Agent Inventory Over Time")
    #     ax_0_1.set_xlabel("Time Step")
    #     ax_0_1.set_ylabel("Inventory (Units)")
    #     ax_0_1.legend()
    #     ax_0_1.grid(True, alpha=0.3)
    #     ax_1_0 = axes[1, 0]
    #     ax_1_0.stackplot(df_history["step"], egt_data, labels=egt_labels)
    #     ax_1_0.set_title("EGT Population Evolution (Replicator Dynamics)")
    #     ax_1_0.set_xlabel("Time Step")
    #     ax_1_0.set_ylabel("Proportion of Population")
    #     ax_1_0.set_ylim(0, 1)
    #     ax_1_0.legend(loc="upper left")
    #     ax_1_1 = axes[1, 1]
    #     ax_1_1.plot(
    #         df_history["step"],
    #         df_history["chosen_spread_width"],
    #         label="Chosen Spread Width",
    #         color="red",
    #         alpha=0.6,
    #     )
    #     ax_1_1.set_title("DRL Agent's Chosen Spread Width")
    #     ax_1_1.set_xlabel("Time Step")
    #     ax_1_1.set_ylabel("Spread Width ($)")
    #     ax_1_1.grid(True, alpha=0.3)
    #     ax_1_1_twin = ax_1_1.twinx()
    #     ax_1_1_twin.plot(
    #         df_history["step"],
    #         df_history["epsilon"],
    #         label="Epsilon",
    #         color="grey",
    #         linestyle=":",
    #     )
    #     ax_1_1_twin.set_ylabel("Epsilon")
    #     lines, labels = ax_1_1.get_legend_handles_labels()
    #     lines2, labels2 = ax_1_1_twin.get_legend_handles_labels()
    #     ax_1_1.legend(lines + lines2, labels + labels2, loc="upper right")
    #     ax_2_0 = axes[2, 0]
    #     ax_2_0.plot(
    #         df_history["step"],
    #         df_history["rolling_avg_reward"],
    #         label="Rolling Avg. Reward (100 steps)",
    #         color="purple",
    #     )
    #     ax_2_0.axhline(y=0, color="black", linestyle="--", linewidth=1)
    #     ax_2_0.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
    #     ax_2_0.set_xlabel("Time Step")
    #     ax_2_0.set_ylabel("Average Reward")
    #     ax_2_0.legend()
    #     ax_2_0.grid(True, alpha=0.3)
    #     ax_2_1 = axes[2, 1]
    #     if not exploiting_spreads.empty:
    #         spread_counts = pd.Series(exploiting_spreads).value_counts().sort_index()
    #         spread_dist = spread_counts / len(exploiting_spreads)
    #         spread_dist.plot(kind="bar", ax=ax_2_1, color="teal")
    #         ax_2_1.set_title(
    #             f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)"
    #         )
    #     else:
    #         ax_2_1.set_title("DRL Learned Policy (No data)")
    #     ax_2_1.set_xlabel("Chosen Spread Width ($)")
    #     ax_2_1.set_ylabel("Proportion of Actions")
    #     ax_2_1.set_ylim(0, 1)
    #     ax_2_1.tick_params(axis="x", rotation=0)
    #     ax_2_1.grid(axis="y", linestyle="--", alpha=0.5)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.savefig("plot_0_combined_summary.png", dpi=STD_DPI)
    #     print("Displaying combined plot window...")




    #     # --- NEW: DRL–EGT Dependency Graph ---
    #     fig_dep, ax_dep = plt.subplots(figsize=STD_FIGSIZE)
    #     df_history["egt_avg"] = df_history[
    #         [f"egt_prop_{s}" for s in self.egt_strategies]
    #     ].mean(axis=1)

    #     # Use correlation between DRL profit changes and EGT shifts as dependency metric
    #     df_history["drl_profit_change"] = df_history["drl_profit"].diff().fillna(0)
    #     df_history["egt_change"] = df_history["egt_avg"].diff().fillna(0)
    #     rolling_corr = (
    #         df_history["drl_profit_change"]
    #         .rolling(window=100, min_periods=10)
    #         .corr(df_history["egt_change"])
    #     )

    #     ax_dep.plot(
    #         df_history["step"],
    #         rolling_corr,
    #         color="darkorange",
    #         label="Rolling Correlation (DRL Profit vs EGT Change)",
    #     )
    #     ax_dep.axhline(y=0, color="black", linestyle="--", linewidth=1)
    #     ax_dep.set_title("Dependency of DRL Agent on EGT Population")
    #     ax_dep.set_xlabel("Time Step")
    #     ax_dep.set_ylabel("Rolling Correlation")
    #     ax_dep.legend()
    #     ax_dep.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig("plot_7_drl_egt_dependency.png", dpi=STD_DPI)
    #     plt.close(fig_dep)




    #     plt.show()


    def plot_results(self):
        STD_FIGSIZE = (8, 6)
        STD_DPI = 300
        df_history = pd.DataFrame(self.history)
        if df_history.empty:
            print("History is empty, skipping plotting.")
            return

        print("PASS 1: Saving individual plot files...")

        # --- Plot 1: DRL Profit ---
        fig1, ax1 = plt.subplots(figsize=STD_FIGSIZE)
        ax1.plot(df_history["step"], df_history["drl_profit"], label="DRL Profit", color="blue")
        ax1.set_title("DRL Agent Cumulative Profit Over Time")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Total Profit ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_1_drl_profit.png", dpi=STD_DPI)
        plt.close(fig1)

        # --- Plot 2: DRL Inventory ---
        fig2, ax2 = plt.subplots(figsize=STD_FIGSIZE)
        ax2.plot(df_history["step"], df_history["drl_inventory"], label="DRL Inventory", color="green")
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1, label="Neutral (0)")
        ax2.set_title("DRL Agent Inventory Over Time")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Inventory (Units)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_2_drl_inventory.png", dpi=STD_DPI)
        plt.close(fig2)

        # --- Plot 3: EGT Population Dynamics ---
        fig3, ax3 = plt.subplots(figsize=STD_FIGSIZE)
        egt_labels = [s.capitalize() for s in self.egt_strategies]
        egt_data = [df_history[f"egt_prop_{s}"] for s in self.egt_strategies]
        ax3.stackplot(df_history["step"], egt_data, labels=egt_labels)
        ax3.set_title("EGT Population Evolution (Replicator Dynamics)")
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("Proportion of Population")
        ax3.set_ylim(0, 1)
        ax3.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("plot_3_egt_dynamics.png", dpi=STD_DPI)
        plt.close(fig3)

        # --- Plot 4: Spread Width + Epsilon ---
        fig4, ax4 = plt.subplots(figsize=STD_FIGSIZE)
        ax4.plot(df_history["step"], df_history["chosen_spread_width"], label="Chosen Spread Width", color="red", alpha=0.6)
        ax4.set_title("DRL Agent's Chosen Spread Width")
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Spread Width ($)")
        ax4.grid(True, alpha=0.3)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df_history["step"], df_history["epsilon"], label="Epsilon", color="grey", linestyle=":")
        ax4_twin.set_ylabel("Epsilon")
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc="upper right")
        plt.tight_layout()
        plt.savefig("plot_4_drl_spread.png", dpi=STD_DPI)
        plt.close(fig4)

        # --- Plot 5: Rolling Average Reward ---
        fig5, ax5 = plt.subplots(figsize=STD_FIGSIZE)
        df_history["rolling_avg_reward"] = df_history["reward"].rolling(window=100, min_periods=1).mean()
        ax5.plot(df_history["step"], df_history["rolling_avg_reward"], label="Rolling Avg. Reward (100 steps)", color="purple")
        ax5.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax5.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
        ax5.set_xlabel("Time Step")
        ax5.set_ylabel("Average Reward")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("plot_5_rolling_reward.png", dpi=STD_DPI)
        plt.close(fig5)

        # --- Plot 6: Policy Distribution (CLEANED UP) ---
        fig6, ax6 = plt.subplots(figsize=STD_FIGSIZE)
        halfway_idx = len(df_history["step"]) // 2
        exploiting_spreads = df_history["chosen_spread_width"].iloc[halfway_idx:]

        if not exploiting_spreads.empty:
            # ✅ Round spread widths to clean up x-axis clutter
            spread_widths_rounded = exploiting_spreads.round(2)
            spread_counts = spread_widths_rounded.value_counts().sort_index()
            spread_dist = spread_counts / len(spread_widths_rounded)

            spread_dist.plot(kind="bar", ax=ax6, color="teal")
            ax6.set_title(f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)")
            ax6.set_xlabel("Chosen Spread Width ($)")
            ax6.set_ylabel("Proportion of Actions")
            ax6.set_ylim(0, 1)

            # ✅ Make x-axis clean and readable
            ax6.set_xticklabels(spread_dist.index, rotation=0)
            ax6.grid(axis="y", linestyle="--", alpha=0.5)
        else:
            ax6.set_title("DRL Learned Policy (No data)")

        plt.tight_layout()
        plt.savefig("plot_6_policy_dist.png", dpi=STD_DPI)
        plt.close(fig6)

        print("Successfully saved 6 plots (e.g., 'plot_1_drl_profit.png').")
        print("PASS 2: Generating combined plot window...")

        # --- Combined Summary Display ---
        fig_display, axes = plt.subplots(3, 2, figsize=(16, 15))
        fig_display.suptitle("DRL Agent vs EGT Population Simulation", fontsize=16)

        # Row 1
        ax_0_0 = axes[0, 0]
        ax_0_0.plot(df_history["step"], df_history["drl_profit"], label="DRL Profit", color="blue")
        ax_0_0.set_title("DRL Agent Cumulative Profit Over Time")
        ax_0_0.set_xlabel("Time Step")
        ax_0_0.set_ylabel("Total Profit ($)")
        ax_0_0.legend()
        ax_0_0.grid(True, alpha=0.3)

        ax_0_1 = axes[0, 1]
        ax_0_1.plot(df_history["step"], df_history["drl_inventory"], label="DRL Inventory", color="green")
        ax_0_1.axhline(y=0, color="black", linestyle="--", linewidth=1, label="Neutral (0)")
        ax_0_1.set_title("DRL Agent Inventory Over Time")
        ax_0_1.set_xlabel("Time Step")
        ax_0_1.set_ylabel("Inventory (Units)")
        ax_0_1.legend()
        ax_0_1.grid(True, alpha=0.3)

        # Row 2
        ax_1_0 = axes[1, 0]
        ax_1_0.stackplot(df_history["step"], egt_data, labels=egt_labels)
        ax_1_0.set_title("EGT Population Evolution (Replicator Dynamics)")
        ax_1_0.set_xlabel("Time Step")
        ax_1_0.set_ylabel("Proportion of Population")
        ax_1_0.set_ylim(0, 1)
        ax_1_0.legend(loc="upper left")

        ax_1_1 = axes[1, 1]
        ax_1_1.plot(df_history["step"], df_history["chosen_spread_width"], label="Chosen Spread Width", color="red", alpha=0.6)
        ax_1_1_twin = ax_1_1.twinx()
        ax_1_1_twin.plot(df_history["step"], df_history["epsilon"], label="Epsilon", color="grey", linestyle=":")
        ax_1_1.set_title("DRL Agent's Chosen Spread Width")
        ax_1_1.set_xlabel("Time Step")
        ax_1_1.set_ylabel("Spread Width ($)")
        ax_1_1_twin.set_ylabel("Epsilon")
        lines, labels = ax_1_1.get_legend_handles_labels()
        lines2, labels2 = ax_1_1_twin.get_legend_handles_labels()
        ax_1_1.legend(lines + lines2, labels + labels2, loc="upper right")
        ax_1_1.grid(True, alpha=0.3)

        # Row 3
        ax_2_0 = axes[2, 0]
        ax_2_0.plot(df_history["step"], df_history["rolling_avg_reward"], label="Rolling Avg. Reward (100 steps)", color="purple")
        ax_2_0.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax_2_0.set_title("DRL Agent's Progressive Learning (Rolling Reward)")
        ax_2_0.set_xlabel("Time Step")
        ax_2_0.set_ylabel("Average Reward")
        ax_2_0.legend()
        ax_2_0.grid(True, alpha=0.3)

        ax_2_1 = axes[2, 1]
        if not exploiting_spreads.empty:
            spread_dist.plot(kind="bar", ax=ax_2_1, color="teal")
            ax_2_1.set_title(f"DRL Learned Policy (Last {len(exploiting_spreads)} Steps)")
            ax_2_1.set_xlabel("Chosen Spread Width ($)")
            ax_2_1.set_ylabel("Proportion of Actions")
            ax_2_1.set_ylim(0, 1)
            ax_2_1.set_xticklabels(spread_dist.index, rotation=0)
            ax_2_1.grid(axis="y", linestyle="--", alpha=0.5)
        else:
            ax_2_1.set_title("DRL Learned Policy (No data)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("plot_0_combined_summary.png", dpi=STD_DPI)
        print("Displaying combined plot window...")
        plt.show()





if __name__ == "__main__":
    SEED = 69
    # SEED = 42
    print(f"Setting global random seed to: {SEED}")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    DATA_FILE = "stock_data.csv"

    drl_params = {
        "gamma": 0.95,
        "epsilon_decay": 0.995,
        "learning_rate": 0.0005,
        "replay_buffer_maxlen": 20000,
        "batch_size": 32,
        "epsilon_min": 0.01,
    }

    nn_architecture = [
        {"units": 64, "activation": "relu"},
        {"units": 64, "activation": "relu"},
    ]

    print(f"Running simulation with data file: {DATA_FILE}")
    try:
        market_sim = Market(
            nn_architecture=nn_architecture,
            drl_params=drl_params,
            ohlcv_filename=DATA_FILE,
        )
        market_sim.run_simulation()
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print(f"Could not find data file: '{DATA_FILE}'.")
        print(
            "Please download a real OHLCV data file and update the DATA_FILE variable."
        )
        print("It must contain 'High', 'Low', 'Close', 'Open', and 'Volume' columns.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()

# fixed x axis clutter on spread policy chart
