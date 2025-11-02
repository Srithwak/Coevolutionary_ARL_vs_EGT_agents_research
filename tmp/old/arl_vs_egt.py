import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import os

class MarketData:
    """
    Loads and provides market data from the CSV file.
    This class simulates the 'real world' data feed.
    """
    def __init__(self, filename="market_data.csv", volatility_window=20):
        if not os.path.exists(filename):
            print(f"Error: '{filename}' not found.")
            print("Please run 'generate_data.py' first to create the market data file.")
            exit()
            
        self.data = pd.read_csv(filename)
        self.max_steps = len(self.data)
        self.volatility_window = volatility_window
        
        # Pre-calculate volatility (rolling standard deviation of returns)
        self.data['returns'] = self.data['mid_price'].pct_change()
        self.data['volatility'] = self.data['returns'].rolling(window=volatility_window).std().fillna(0)
        
    def get_price(self, step):
        """Returns the mid_price at a given step."""
        if step >= self.max_steps:
            return None
        return self.data.loc[step, 'mid_price']
        
    def get_volatility(self, step):
        """Returns the market volatility at a given step."""
        if step >= self.max_steps:
            return None
        return self.data.loc[step, 'volatility']

class EGTAgent:
    """
    Fixed-strategy "dumb money" agent.
    It randomly decides to buy or sell at a price relative to the mid_price.
    """
    def __init__(self, agent_id, strategy):
        self.id = agent_id
        self.strategy = strategy
        self.profit = 0

    def get_action(self, mid_price):
        """
        Decides what to do (buy/sell) and at what price.
        :return: (action_type, price)
        """
        action_type = random.choice(['buy', 'sell'])
        
        # Define the price based on strategy
        if self.strategy == 'aggressive':
            # Trades very close to the mid-price (will cross narrow spreads)
            price_offset = random.uniform(0.01, 0.05)
        elif self.strategy == 'passive':
            # Trades far from the mid-price (needs a wide spread)
            price_offset = random.uniform(0.3, 0.6)
        else: # 'random'
            price_offset = random.uniform(0.01, 0.5)
            
        if action_type == 'buy':
            return 'buy', mid_price + price_offset
        else: # 'sell'
            return 'sell', mid_price - price_offset

class ARLAgent:
    """
    Adaptive Reinforcement Learning Market Maker.
    Learns the optimal bid-ask spread to quote.
    """
    def __init__(self, agent_id):
        self.id = agent_id
        
        # Agent's state
        self.inventory = 0
        self.cash = 0
        
        # Actions are the *spreads* it can quote
        # (e.g., 0.1 = mid_price +/- 0.1)
        self.actions = [0.1, 0.2, 0.4] # Narrow, Medium, Wide spreads
        
        # Q-Learning parameters
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def get_state(self, volatility):
        """
        Discretizes the continuous state (inventory, volatility)
        into bins for the Q-table.
        """
        # Bin 1: Inventory (limit to -5 to +5)
        inv_bin = max(-5, min(5, self.inventory))
        
        # Bin 2: Volatility (is it low, medium, or high?)
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
            # Explore
            action_idx = random.choice(range(len(self.actions)))
        else:
            # Exploit
            action_idx = np.argmax(self.q_table[state])
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return action_idx

    def learn(self, state, action_idx, reward, next_state):
        """Update Q-table using the Bellman equation."""
        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action_idx] = new_q

class Market:
    """
    Simulated market environment.
    Orchestrates the interaction between the ARL agent and EGT agents
    using the "real" market data.
    """
    def __init__(self):
        self.market_data = MarketData()
        self.step = 0
        
        # Create ARL Market Maker
        self.arl_agent = ARLAgent(agent_id=0)
        
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
            'step': [],
            'mid_price': [],
            'arl_profit': [],
            'arl_inventory': [],
            'avg_egt_profit': [],
            'chosen_spread': [],
            'epsilon': []
        }
        
        # Store last portfolio value for reward calculation
        self.last_arl_value = 0

    def run_step(self):
        """Execute one time step of the market."""
        mid_price = self.market_data.get_price(self.step)
        volatility = self.market_data.get_volatility(self.step)
        
        if mid_price is None:
            return False # Simulation over
            
        # 1. ARL AGENT'S TURN (Set Quotes)
        
        # Get current state
        state = self.arl_agent.get_state(volatility)
        
        # Choose an action (which spread to use)
        action_idx = self.arl_agent.choose_action(state)
        chosen_spread = self.arl_agent.actions[action_idx]
        
        # Set the ARL agent's quotes
        arl_bid_price = mid_price - chosen_spread
        arl_ask_price = mid_price + chosen_spread
        
        # 2. EGT AGENTS' TURN (Trade against ARL)
        
        # Shuffle EGT agents to randomize trade order
        random.shuffle(self.egt_agents)
        
        for egt_agent in self.egt_agents:
            egt_action, egt_price = egt_agent.get_action(mid_price)
            
            if egt_action == 'sell' and egt_price <= arl_bid_price:
                # EGT agent SELLS, ARL agent BUYS
                self.arl_agent.inventory += 1
                self.arl_agent.cash -= arl_bid_price
                egt_agent.profit += arl_bid_price
                
            elif egt_action == 'buy' and egt_price >= arl_ask_price:
                # EGT agent BUYS, ARL agent SELLS
                self.arl_agent.inventory -= 1
                self.arl_agent.cash += arl_ask_price
                egt_agent.profit -= arl_ask_price # EGT cash goes down
        
        # 3. ARL AGENT LEARNS (Calculate Reward)
        
        # Reward is the change in total portfolio value (cash + inventory value)
        # This is "Mark-to-Market" profit
        current_portfolio_value = self.arl_agent.cash + (self.arl_agent.inventory * mid_price)
        
        # The reward is the *change* in value from the last step
        reward = current_portfolio_value - self.last_arl_value
        
        # Add a small penalty for holding inventory (risk)
        # This encourages the agent to be a *market maker* (flat)
        # not a *speculator* (long/short)
        inventory_penalty = (self.arl_agent.inventory ** 2) * 0.01 
        reward -= inventory_penalty
        
        # Get next state for learning
        next_mid_price = self.market_data.get_price(self.step + 1)
        next_volatility = self.market_data.get_volatility(self.step + 1)
        
        if next_mid_price is not None:
            next_state = self.arl_agent.get_state(next_volatility)
            
            # Update the Q-table
            self.arl_agent.learn(state, action_idx, reward, next_state)
        
        # Update value for next step's reward calculation
        self.last_arl_value = current_portfolio_value
        
        # 4. RECORD HISTORY
        self.history['step'].append(self.step)
        self.history['mid_price'].append(mid_price)
        self.history['arl_profit'].append(current_portfolio_value)
        self.history['arl_inventory'].append(self.arl_agent.inventory)
        self.history['avg_egt_profit'].append(np.mean([a.profit for a in self.egt_agents]))
        self.history['chosen_spread'].append(chosen_spread)
        self.history['epsilon'].append(self.arl_agent.epsilon)
        
        self.step += 1
        return True

    def run_simulation(self):
        """Run the full simulation."""
        print(f"Starting simulation: 1 ARL agent vs {len(self.egt_agents)} EGT agents.")
        print(f"Data steps: {self.market_data.max_steps}")
        
        while self.run_step():
            if self.step % 100 == 0:
                print(f"Step {self.step}/{self.market_data.max_steps}...")
        
        print("Simulation complete.")
        self.print_summary()
        self.plot_results()

    def print_summary(self):
        """Print final summary statistics."""
        final_arl_profit = self.history['arl_profit'][-1]
        final_avg_egt_profit = self.history['avg_egt_profit'][-1]
        
        print("\n" + "="*30)
        print("=== FINAL RESULTS ===")
        print(f"ARL Agent Final Profit:     ${final_arl_profit:,.2f}")
        print(f"Avg. EGT Agent Final Profit: ${final_avg_egt_profit:,.2f}")
        
        best_egt = max(self.egt_agents, key=lambda a: a.profit)
        worst_egt = min(self.egt_agents, key=lambda a: a.profit)
        
        print(f"\nBest EGT ({best_egt.strategy}):   ${best_egt.profit:,.2f}")
        print(f"Worst EGT ({worst_egt.strategy}): ${worst_egt.profit:,.2f}")
        print("="*30 + "\n")

    def plot_results(self):
        """Visualize simulation results with 4 plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Market Maker Simulation Results', fontsize=16)
        
        # Plot 1: ARL vs EGT Profits
        ax1 = axes[0, 0]
        ax1.plot(self.history['step'], self.history['arl_profit'], 
                 label='ARL Agent Profit', color='blue', linewidth=2)
        ax1.plot(self.history['step'], self.history['avg_egt_profit'], 
                 label='Avg. EGT Agent Profit', color='orange', linestyle='--')
        ax1.set_title('Cumulative Profit Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Total Profit ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ARL Agent Inventory
        ax2 = axes[0, 1]
        ax2.plot(self.history['step'], self.history['arl_inventory'], 
                 label='ARL Inventory', color='green')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Neutral (0)')
        ax2.set_title('ARL Agent Inventory Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Inventory (Units)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Market Price
        ax3 = axes[1, 0]
        ax3.plot(self.history['step'], self.history['mid_price'], 
                 label='Market Mid-Price', color='purple')
        ax3.set_title('Market Price (from CSV)')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Price ($)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: ARL's Chosen Spread vs Epsilon
        ax4 = axes[1, 1]
        ax4.plot(self.history['step'], self.history['chosen_spread'], 
                 label='Chosen Spread', color='red', alpha=0.6)
        ax4.set_title('ARL Agent\'s Chosen Spread')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Spread ($)')
        ax4.grid(True, alpha=0.3)
        
        # Add epsilon on a secondary y-axis
        ax4_twin = ax4.twinx()
        ax4_twin.plot(self.history['step'], self.history['epsilon'], 
                      label='Epsilon (Exploration)', color='grey', linestyle=':', alpha=0.8)
        ax4_twin.set_ylabel('Epsilon')
        
        # Combine legends for plot 4
        lines, labels = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('arl_market_maker_results.png', dpi=300)
        print("Plots saved as 'arl_market_maker_results.png'")
        plt.show()

if __name__ == "__main__":
    # 1. First, make sure you have run 'generate_data.py'
    # 2. Then, run this script.
    market = Market()
    market.run_simulation()
