import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MarketData:
    """
    Loads and provides market data from the CSV file.
    This class simulates the 'real world' data feed.
    """
    def __init__(self, filename="market_data.csv", volatility_window=20, short_ma=5, long_ma=30):
        if not os.path.exists(filename):
            print(f"Error: '{filename}' not found.")
            print("Please run 'generate_data.py' first to create the market data file.")
            exit()
            
        self.data = pd.read_csv(filename)
        self.max_steps = len(self.data)
        
        # Pre-calculate volatility
        self.data['returns'] = self.data['mid_price'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(window=volatility_window).std().fillna(0)
        
        # Pre-calculate trend (new state feature)
        self.data['short_ma'] = self.data['mid_price'].rolling(window=short_ma).mean()
        self.data['long_ma'] = self.data['mid_price'].rolling(window=long_ma).mean()
        
        # Fill NaNs at the beginning
        self.data.fillna(method='bfill', inplace=True) 
        
        # Create a trend signal (normalized)
        self.data['trend'] = (self.data['short_ma'] - self.data['long_ma']) / self.data['mid_price']

    def get_price(self, step):
        """Returns the mid_price at a given step."""
        if step >= self.max_steps:
            return None
        return self.data.loc[step, 'mid_price']
        
    def get_market_state(self, step):
        """Returns the market state (volatility, trend) at a given step."""
        if step >= self.max_steps:
            return (None, None)
        vol = self.data.loc[step, 'volatility']
        trend = self.data.loc[step, 'trend']
        return (vol, trend)

class EGTAgent:
    """
    Fixed-strategy "dumb money" agent.
    (No changes from original)
    """
    def __init__(self, agent_id, strategy):
        self.id = agent_id
        self.strategy = strategy
        self.profit = 0

    def get_action(self, mid_price):
        action_type = random.choice(['buy', 'sell'])
        
        if self.strategy == 'aggressive':
            price_offset = random.uniform(0.01, 0.05)
        elif self.strategy == 'passive':
            price_offset = random.uniform(0.3, 0.6)
        else: # 'random'
            price_offset = random.uniform(0.01, 0.5)
            
        if action_type == 'buy':
            return 'buy', mid_price + price_offset
        else: # 'sell'
            return 'sell', mid_price - price_offset

class QLearningAgent:
    """
    The original Adaptive Reinforcement Learning agent, now renamed.
    This serves as our baseline model.
    """
    def __init__(self, agent_id):
        self.id = agent_id
        self.inventory = 0
        self.cash = 0
        
        # Actions are the *symmetric spreads*
        self.actions = [0.1, 0.2, 0.4] # Narrow, Medium, Wide
        
        # Q-Learning parameters
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha = 0.1 
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def get_state(self, volatility, trend_signal):
        """
        Discretizes the continuous state for the Q-table.
        NOTE: This baseline agent *only* uses volatility,
        to show a clear comparison with the DQN.
        """
        # Bin 1: Inventory
        inv_bin = max(-5, min(5, self.inventory))
        
        # Bin 2: Volatility
        if volatility < 0.005:
            vol_bin = 'low'
        elif volatility < 0.015:
            vol_bin = 'medium'
        else:
            vol_bin = 'high'
            
        return (inv_bin, vol_bin)

    def choose_action(self, state):
        """Chooses an action (a spread) using epsilon-greedy."""
        if random.random() < self.epsilon:
            action_idx = random.choice(range(len(self.actions)))
        else:
            action_idx = np.argmax(self.q_table[state])
            
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return action_idx

    def learn(self, state, action_idx, reward, next_state):
        """Update Q-table using the Bellman equation."""
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q

class DQNAgent:
    """
    IMPROVEMENT 1: New Deep Q-Network (DQN) Agent.
    This agent uses a neural network to handle a continuous state space.
    """
    def __init__(self, agent_id):
        self.id = agent_id
        self.inventory = 0
        self.cash = 0
        
        # IMPROVEMENT 2: Enhanced State (Inventory, Volatility, Trend)
        self.state_size = 3 
        
        # IMPROVEMENT 3: Asymmetric Action Space
        # (Narrow, Wide, Skew-Sell, Skew-Buy)
        self.actions = [(0.1, 0.1), (0.3, 0.3), (0.3, 0.1), (0.1, 0.3)]
        self.action_size = len(self.actions)
        
        # DQN parameters
        self.replay_buffer = deque(maxlen=2000)
        self.gamma = 0.95    # Discount factor
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # Build the Q-Network and the Target Network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Builds a simple neural network for Q-value approximation."""
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) # Linear output for Q-values
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def get_state(self, volatility, trend_signal):
        """Returns the continuous state vector."""
        # Normalize inventory for the NN
        norm_inventory = self.inventory / 10.0 
        state = np.array([[norm_inventory, volatility, trend_signal]])
        return state

    def choose_action(self, state):
        """Chooses an action using epsilon-greedy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # Explore
        
        # Exploit
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state, action_idx, reward, next_state, done):
        """Stores an experience in the replay buffer."""
        self.replay_buffer.append((state, action_idx, reward, next_state, done))
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def replay(self, batch_size):
        """Trains the model by replaying experiences from the buffer."""
        if len(self.replay_buffer) < batch_size:
            return # Not enough samples to train
            
        minibatch = random.sample(self.replay_buffer, batch_size)
        
        for state, action_idx, reward, next_state, done in minibatch:
            # Get the Q-value for the next state from the *target* model
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state, verbose=0)[0])
            else:
                target = reward
            
            # Get the current Q-values from the *main* model
            q_values = self.model.predict(state, verbose=0)
            
            # Update the Q-value for the action that was taken
            q_values[0][action_idx] = target
            
            # Train the main model
            self.model.fit(state, q_values, epochs=1, verbose=0)

class Market:
    """
    Simulated market environment.
    Now supports different agent types.
    """
    def __init__(self, agent_type='dqn'):
        self.market_data = MarketData()
        self.step = 0
        self.agent_type = agent_type
        
        # Create ARL Market Maker
        if self.agent_type == 'q_learning':
            self.arl_agent = QLearningAgent(agent_id=0)
        elif self.agent_type == 'dqn':
            self.arl_agent = DQNAgent(agent_id=0)
        else:
            raise ValueError("Unknown agent type")
            
        self.actions = self.arl_agent.actions
        
        # Create EGT population
        self.egt_agents = []
        strategies = {'aggressive': 5, 'passive': 5, 'random': 5}
        agent_id = 1
        for strategy, count in strategies.items():
            for _ in range(count):
                self.egt_agents.append(EGTAgent(agent_id, strategy))
                agent_id += 1
                
        # History tracking
        self.history = {
            'step': [], 'mid_price': [], 'arl_profit': [], 'arl_inventory': [],
            'avg_egt_profit': [], 'chosen_action_idx': [], 'epsilon': []
        }
        self.last_arl_value = 0

    def run_step(self):
        """Execute one time step of the market."""
        mid_price = self.market_data.get_price(self.step)
        volatility, trend_signal = self.market_data.get_market_state(self.step)
        
        if mid_price is None:
            return False # Simulation over
            
        # 1. ARL AGENT'S TURN (Set Quotes)
        state = self.arl_agent.get_state(volatility, trend_signal)
        action_idx = self.arl_agent.choose_action(state)
        
        # Interpret the chosen action
        if self.agent_type == 'q_learning':
            chosen_spread = self.actions[action_idx]
            arl_bid_price = mid_price - chosen_spread
            arl_ask_price = mid_price + chosen_spread
        else: # dqn
            bid_spread, ask_spread = self.actions[action_idx]
            arl_bid_price = mid_price - bid_spread
            arl_ask_price = mid_price + ask_spread
        
        # 2. EGT AGENTS' TURN (Trade against ARL)
        random.shuffle(self.egt_agents)
        trades_made = 0
        
        for egt_agent in self.egt_agents:
            egt_action, egt_price = egt_agent.get_action(mid_price)
            
            if egt_action == 'sell' and egt_price <= arl_bid_price:
                self.arl_agent.inventory += 1
                self.arl_agent.cash -= arl_bid_price
                egt_agent.profit += arl_bid_price
                trades_made += 1
                
            elif egt_action == 'buy' and egt_price >= arl_ask_price:
                self.arl_agent.inventory -= 1
                self.arl_agent.cash += arl_ask_price
                egt_agent.profit -= arl_ask_price
                trades_made += 1
        
        # 3. ARL AGENT LEARNS (Calculate Reward)
        current_portfolio_value = self.arl_agent.cash + (self.arl_agent.inventory * mid_price)
        reward = current_portfolio_value - self.last_arl_value
        
        # Inventory penalty
        inventory_penalty = (self.arl_agent.inventory ** 2) * 0.01 
        reward -= inventory_penalty
        
        # IMPROVEMENT 4: "Green AI" Penalty
        energy_penalty = trades_made * 0.005 # Small cost per-trade
        reward -= energy_penalty
        
        # Get next state for learning
        next_mid_price = self.market_data.get_price(self.step + 1)
        next_volatility, next_trend_signal = self.market_data.get_market_state(self.step + 1)
        
        done = (next_mid_price is None)
        
        if not done:
            next_state = self.arl_agent.get_state(next_volatility, next_trend_signal)
            
            # Update the appropriate agent
            if self.agent_type == 'q_learning':
                self.arl_agent.learn(state, action_idx, reward, next_state)
            else: # dqn
                self.arl_agent.remember(state, action_idx, reward, next_state, done)
        
        self.last_arl_value = current_portfolio_value
        
        # 4. RECORD HISTORY
        self.history['step'].append(self.step)
        self.history['mid_price'].append(mid_price)
        self.history['arl_profit'].append(current_portfolio_value)
        self.history['arl_inventory'].append(self.arl_agent.inventory)
        self.history['avg_egt_profit'].append(np.mean([a.profit for a in self.egt_agents]))
        self.history['chosen_action_idx'].append(action_idx)
        self.history['epsilon'].append(self.arl_agent.epsilon)
        
        self.step += 1
        return True

    def run_simulation(self):
        """Run the full simulation."""
        print(f"Starting simulation ({self.agent_type}): 1 ARL agent vs {len(self.egt_agents)} EGT agents.")
        
        batch_size = 32 # For DQN replay
        update_target_freq = 50 # For DQN
        
        while self.run_step():
            if self.step % 100 == 0:
                print(f"Step {self.step}/{self.market_data.max_steps}...")
            
            # DQN-specific learning steps
            if self.agent_type == 'dqn':
                if len(self.arl_agent.replay_buffer) > batch_size:
                    self.arl_agent.replay(batch_size)
                
                if self.step % update_target_freq == 0:
                    self.arl_agent.update_target_model()
        
        print(f"Simulation complete ({self.agent_type}).")
        self.print_summary()
        return self.history # Return history for plotting

    def print_summary(self):
        """Print final summary statistics."""
        final_arl_profit = self.history['arl_profit'][-1]
        print("\n" + "="*30)
        print(f"=== FINAL RESULTS ({self.agent_type}) ===")
        print(f"ARL Agent Final Profit: ${final_arl_profit:,.2f}")
        print("="*30 + "\n")

def plot_comparative_results(baseline_history, dqn_history):
    """
    IMPROVEMENT 5: Visualize simulation results with 4 plots,
    comparing the baseline Q-Learner to the new DQN agent.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Market Maker Simulation Results: Q-Learning vs. DQN', fontsize=16)
    
    # Plot 1: ARL vs EGT Profits
    ax1 = axes[0, 0]
    ax1.plot(baseline_history['step'], baseline_history['arl_profit'], 
             label='Baseline Q-Learning Profit', color='orange', linestyle='--', linewidth=2)
    ax1.plot(dqn_history['step'], dqn_history['arl_profit'], 
             label='New DQN Profit', color='blue', linewidth=2)
    ax1.set_title('Cumulative Profit Over Time (Baseline vs. New)')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Total Profit ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: DQN Agent Inventory
    ax2 = axes[0, 1]
    ax2.plot(dqn_history['step'], dqn_history['arl_inventory'], 
             label='DQN Inventory', color='green')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Neutral (0)')
    ax2.set_title('DQN Agent Inventory Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Inventory (Units)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Market Price
    ax3 = axes[1, 0]
    ax3.plot(dqn_history['step'], dqn_history['mid_price'], 
             label='Market Mid-Price', color='purple')
    ax3.set_title('Market Price (from CSV)')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DQN's Chosen Action vs Epsilon
    ax4 = axes[1, 1]
    # Use 'o' marker for discrete action indices
    ax4.plot(dqn_history['step'], dqn_history['chosen_action_idx'], 
             label='DQN Action Index', color='red', alpha=0.4, linestyle='None', marker='o', markersize=1)
    ax4.set_title('DQN Agent\'s Chosen Action (0-3)')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Action Index')
    ax4.set_yticks(range(4))
    ax4.set_yticklabels(['Narrow', 'Wide', 'Skew-Sell', 'Skew-Buy'])
    ax4.grid(True, alpha=0.3)
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(dqn_history['step'], dqn_history['epsilon'], 
                  label='Epsilon (Exploration)', color='grey', linestyle=':', alpha=0.8)
    ax4_twin.set_ylabel('Epsilon')
    
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('arl_comparative_results.png', dpi=300)
    print("Plots saved as 'arl_comparative_results.png'")
    plt.show()

if __name__ == "__main__":
    # 1. Run the baseline simulation
    print("--- Running baseline Q-Learning simulation... ---")
    market_baseline = Market(agent_type='q_learning')
    history_baseline = market_baseline.run_simulation()
    
    # 2. Run the new DQN simulation
    print("--- Running new DQN simulation... ---")
    market_dqn = Market(agent_type='dqn')
    history_dqn = market_dqn.run_simulation()
    
    # 3. Plot the comparative results
    print("--- Plotting comparative results... ---")
    plot_comparative_results(history_baseline, history_dqn)