import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from matplotlib.colors import ListedColormap

# --- Configuration ---
SEED = 101
np.random.seed(SEED)
random.seed(SEED)

NUM_EPISODES = 60
TRADING_DAYS = 200
NUM_ERL_AGENTS = 25
INITIAL_CAPITAL = 50000.0
TRANSACTION_COST = 0.001
BASE_LIQUIDITY = 1000

# Market Regimes
REGIME_BEAR = 0
REGIME_NEUTRAL = 1
REGIME_BULL = 2

# --- Market & Agent Logic (Optimized for Data Collection) ---

class ComplexMarket:
    def __init__(self, start_price=100.0):
        self.start_price = start_price
        self.trans_matrix = np.array([
            [0.9, 0.1, 0.0],
            [0.05, 0.9, 0.05],
            [0.0, 0.1, 0.9]
        ])
        self.reset()

    def reset(self):
        self.prices = [self.start_price]
        self.current_step = 0
        self.regime = REGIME_NEUTRAL
        self.regime_history = [self.regime]
        self.news_signal = 0 
        return self._get_state()

    def _get_state(self):
        if len(self.prices) < 2: ret = 0.0
        else: ret = np.log(self.prices[-1] / self.prices[-2])
        
        if len(self.prices) < 11: vol = 0.01
        else: vol = np.std(self.prices[-10:])
        
        return np.array([ret * 10, vol * 100, self.news_signal])

    def update_regime(self):
        self.regime = np.random.choice([0, 1, 2], p=self.trans_matrix[self.regime])
        self.regime_history.append(self.regime)

    def get_slippage(self, order_size):
        return (abs(order_size) / BASE_LIQUIDITY) ** 2 * 0.01

    def step(self, net_order_flow):
        self.update_regime()
        prev_price = self.prices[-1]
        
        if self.regime == REGIME_BEAR:
            drift, base_vol, news_bias = -0.0015, 0.025, -0.5
        elif self.regime == REGIME_BULL:
            drift, base_vol, news_bias = 0.0015, 0.01, 0.5
        else:
            drift, base_vol, news_bias = 0.0, 0.015, 0.0

        self.news_signal = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
        shock = np.random.normal(drift, base_vol)
        price_impact = (net_order_flow / NUM_ERL_AGENTS) * 0.01
        news_impact = (self.news_signal + news_bias) * 0.005

        new_price = prev_price * np.exp(shock + price_impact + news_impact)
        self.prices.append(new_price)
        self.current_step += 1
        return self._get_state(), new_price, self.current_step >= TRADING_DAYS

class Agent:
    def __init__(self, name):
        self.capital = INITIAL_CAPITAL
        self.shares = 0
        self.portfolio_history = []

    def get_portfolio_value(self, current_price):
        return self.capital + (self.shares * current_price)

class ComplexERLAgent(Agent):
    def __init__(self, id, type_override=None):
        super().__init__(f"ERL_{id}")
        self.strategy_type = type_override if type_override is not None else random.choice([0, 1])
        self.weights = np.random.randn(3, 3) 
        
    def act(self, state):
        mod_state = state.copy()
        if self.strategy_type == 1: mod_state[0] = -mod_state[0]
        logits = np.dot(mod_state, self.weights)
        return np.argmax(logits)

    def execute(self, action, price, slippage):
        lot = 10 
        eff_buy = price * (1 + slippage + TRANSACTION_COST)
        eff_sell = price * (1 - slippage - TRANSACTION_COST)
        if action == 1 and self.capital >= eff_buy * lot:
            self.capital -= eff_buy * lot; self.shares += lot; return lot
        elif action == 2 and self.shares >= lot:
            self.capital += eff_sell * lot; self.shares -= lot; return -lot
        return 0

    def mutate(self):
        self.weights += np.random.randn(*self.weights.shape) * 0.5
        if np.random.random() < 0.05: self.strategy_type = 1 - self.strategy_type

class AdvancedARLAgent(Agent):
    def __init__(self):
        super().__init__("ARL_Pro")
        self.q_table = {} 
        self.learning_rate = 0.1; self.discount = 0.95; self.epsilon = 1.0
        self.prev_state = None; self.prev_act = 0; self.prev_val = INITIAL_CAPITAL

    def _key(self, m_state, herd):
        return (int(round(m_state[0])), int(round(m_state[1])), int(m_state[2]), 1 if herd>30 else (-1 if herd<-30 else 0))

    def act(self, m_state, herd):
        key = self._key(m_state, herd)
        if np.random.random() < self.epsilon: return np.random.randint(0, 3)
        return np.argmax(self.q_table.get(key, np.zeros(3)))

    def execute(self, action, price, slippage):
        lot = 50
        eff_buy = price * (1 + slippage + TRANSACTION_COST)
        eff_sell = price * (1 - slippage - TRANSACTION_COST)
        if action == 1 and self.shares < 500:
            self.capital -= eff_buy * lot; self.shares += lot; return lot
        elif action == 2 and self.shares > -500:
            self.capital += eff_sell * lot; self.shares -= lot; return -lot
        return 0

    def learn(self, price, m_state, herd):
        val = self.get_portfolio_value(price)
        reward = val - self.prev_val
        curr_key = self._key(m_state, herd)
        
        old_q = self.q_table.get(self.prev_state, np.zeros(3))
        next_max = np.max(self.q_table.get(curr_key, np.zeros(3)))
        
        # Update Q
        old_q[self.prev_act] += self.learning_rate * (reward + self.discount * next_max - old_q[self.prev_act])
        self.q_table[self.prev_state] = old_q
        
        if self.epsilon > 0.05: self.epsilon *= 0.96
        self.prev_val = val; self.prev_state = curr_key

    def set_choice(self, act): self.prev_act = act

# --- Simulation Runner ---

def run_simulation_for_viz():
    market = ComplexMarket()
    erl_pop = [ComplexERLAgent(i, i%2) for i in range(NUM_ERL_AGENTS)]
    arl = AdvancedARLAgent()
    
    # Data Storage
    arl_curve, erl_curve = [], []
    herd_mix = [] # [Count Mom, Count Rev]
    regime_perf = {0: [], 1: [], 2: []} # Returns per regime
    
    print("Simulating Market...")
    for ep in range(NUM_EPISODES):
        state = market.reset()
        for a in erl_pop + [arl]: a.capital = INITIAL_CAPITAL; a.shares = 0; a.prev_val = INITIAL_CAPITAL

        ep_regimes = []
        ep_arl_start = arl.get_portfolio_value(market.prices[-1])
        
        for t in range(TRADING_DAYS):
            price = market.prices[-1]
            
            # Actions
            erl_acts = [a.act(state) for a in erl_pop]
            exp_flow = sum([10 if a==1 else (-10 if a==2 else 0) for a in erl_acts])
            arl_act = arl.act(state, exp_flow)
            arl.set_choice(arl_act)
            
            # Execute
            slip = market.get_slippage(exp_flow + (50 if arl_act==1 else -50 if arl_act==2 else 0))
            real_flow = sum([e.execute(a, price, slip) for e, a in zip(erl_pop, erl_acts)])
            real_flow += arl.execute(arl_act, price, slip)
            
            # Step
            next_state, new_price, done = market.step(real_flow)
            arl.learn(new_price, next_state, exp_flow)
            state = next_state
            
            # Track Regime Performance (Daily returns)
            daily_ret = (new_price - price) / price
            # Approximate: did ARL make money today? 
            # Simplified: We track EPISODE returns per dominant regime instead for clearer boxplots
            ep_regimes.append(market.regime)

        # End Episode
        final_price = market.prices[-1]
        erl_pop.sort(key=lambda x: x.get_portfolio_value(final_price), reverse=True)
        
        avg_erl = np.mean([a.get_portfolio_value(final_price) for a in erl_pop])
        val_arl = arl.get_portfolio_value(final_price)
        
        arl_curve.append(val_arl)
        erl_curve.append(avg_erl)
        
        # Herd Composition
        moms = sum(1 for a in erl_pop if a.strategy_type == 0)
        herd_mix.append([moms, NUM_ERL_AGENTS - moms])
        
        # Regime classification for this episode (Dominant regime)
        dom_regime = max(set(ep_regimes), key=ep_regimes.count)
        regime_perf[dom_regime].append(val_arl - avg_erl) # Alpha per regime

        # Evolution
        survivors = erl_pop[:int(NUM_ERL_AGENTS*0.25)]
        new_pop = survivors[:]
        while len(new_pop) < NUM_ERL_AGENTS:
            child = copy.deepcopy(random.choice(survivors))
            child.mutate()
            child.shares = 0; child.capital = INITIAL_CAPITAL
            new_pop.append(child)
        erl_pop = new_pop

    return arl_curve, erl_curve, herd_mix, regime_perf, arl.q_table

# --- Advanced Visualization Functions ---

def plot_research_dashboard(arl_data, erl_data, herd_data, regime_data, q_table):
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"ARL vs Herd: Research Diagnostics (Seed {SEED})", fontsize=16)

    # --- 1. Underwater Plot (Drawdown & Risk) ---
    ax1 = plt.subplot(2, 2, 3)
    
    def get_drawdown(curve):
        curve = np.array(curve)
        peaks = np.maximum.accumulate(curve)
        drawdown = (curve - peaks) / peaks
        return drawdown * 100

    dd_arl = get_drawdown(arl_data)
    dd_erl = get_drawdown(erl_data)

    ax1.fill_between(range(len(dd_arl)), dd_arl, 0, color='red', alpha=0.3, label='ARL Drawdown')
    ax1.plot(dd_arl, color='red', linewidth=1)
    ax1.plot(dd_erl, color='gray', linestyle='--', linewidth=1, label='Herd Drawdown')
    ax1.set_title("Risk Analysis: Maximum Drawdown (%)")
    ax1.set_ylabel("Drawdown %")
    ax1.set_xlabel("Episode")
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.2)

    # --- 2. Policy Brain Scan (Heatmap) ---
    ax2 = plt.subplot(2, 2, 2)
    
    # Grid: Momentum (x) vs Volatility (y)
    # We reconstruct the policy from Q-table
    mom_range = range(-5, 6)
    vol_range = range(0, 6)
    policy_grid = np.zeros((len(vol_range), len(mom_range)))
    
    for i, vol in enumerate(vol_range):
        for j, mom in enumerate(mom_range):
            # Assume neutral news (0) and neutral herd (0) for base map
            key = (mom, vol, 0, 0) 
            if key in q_table:
                best_action = np.argmax(q_table[key]) # 0, 1, 2
            else:
                best_action = 0 # Default Hold
            
            # Map actions to values: Sell=-1, Hold=0, Buy=1 for coloring
            val = -1 if best_action == 2 else (1 if best_action == 1 else 0)
            policy_grid[i, j] = val

    # Custom Colormap: Red (Sell), Gray (Hold), Green (Buy)
    cmap = ListedColormap(['#ffcccc', '#f0f0f0', '#ccffcc']) # Light Red, Gray, Light Green
    
    cax = ax2.imshow(policy_grid, cmap=cmap, origin='lower', aspect='auto')
    
    # Add text annotations
    for i in range(len(vol_range)):
        for j in range(len(mom_range)):
            val = policy_grid[i, j]
            txt = "S" if val == -1 else ("B" if val == 1 else "-")
            col = "red" if val == -1 else ("green" if val == 1 else "gray")
            ax2.text(j, i, txt, ha='center', va='center', color=col, fontweight='bold')

    ax2.set_title("ARL Brain Scan: Policy Map (Neutral News/Herd)")
    ax2.set_xticks(range(len(mom_range)))
    ax2.set_xticklabels(mom_range)
    ax2.set_yticks(range(len(vol_range)))
    ax2.set_yticklabels(vol_range)
    ax2.set_xlabel("Market Momentum (Return)")
    ax2.set_ylabel("Market Volatility")
    
    # Legend for Heatmap
    patches = [mpatches.Patch(color='#ccffcc', label='Buy Region'),
               mpatches.Patch(color='#ffcccc', label='Short Region')]
    ax2.legend(handles=patches, loc='upper right', fontsize='small')

    # --- 3. Regime Robustness (Box Plot) ---
    ax3 = plt.subplot(2, 2, 4)
    
    # Data prep
    data_to_plot = [regime_data[0], regime_data[1], regime_data[2]]
    labels = ['Bear Market', 'Neutral', 'Bull Market']
    
    # Remove empty lists to avoid errors
    valid_data = []
    valid_labels = []
    for d, l in zip(data_to_plot, labels):
        if len(d) > 0:
            valid_data.append(d)
            valid_labels.append(l)
    
    if valid_data:
        bp = ax3.boxplot(valid_data, patch_artist=True, labels=valid_labels)
        
        # Color code boxes
        colors = ['#ff9999', '#ffff99', '#99ff99'] # Red, Yellow, Green
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        ax3.axhline(0, color='black', linestyle='--', linewidth=1)
        ax3.set_title("Performance (Alpha) by Market Regime")
        ax3.set_ylabel("Excess Profit ($)")
    else:
        ax3.text(0.5, 0.5, "Insufficient Regime Data", ha='center')

    # --- 4. Herd Evolution (Stacked Area) ---
    ax4 = plt.subplot(2, 2, 1)
    
    moms = [x[0] for x in herd_data]
    revs = [x[1] for x in herd_data]
    episodes = range(len(moms))
    
    ax4.stackplot(episodes, moms, revs, labels=['Momentum Traders', 'Mean Reverters'], 
                  colors=['#3498db', '#e67e22'], alpha=0.7)
    
    # Overlay Wealth Curve
    ax4_twin = ax4.twinx()
    ax4_twin.plot(arl_data, color='black', linewidth=2, label='ARL Wealth')
    ax4_twin.set_ylabel("ARL Wealth ($)")
    
    ax4.set_title("Herd Evolution & ARL Wealth")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Agent Count")
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='lower right')
    ax4.set_xlim(0, len(episodes)-1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    arl_c, erl_c, herd_mix, regime_p, q_tab = run_simulation_for_viz()
    plot_research_dashboard(arl_c, erl_c, herd_mix, regime_p, q_tab)