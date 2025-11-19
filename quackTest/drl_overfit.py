import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import copy

# --- Configuration ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

NUM_EPISODES = 60
TRADING_DAYS = 252
NUM_AGENTS = 30
INITIAL_CAPITAL = 100000.0
RISK_FREE_RATE = 0.02 / 252
INPUT_NOISE_LEVEL = 0.05 

# --- Robust Neural Network ---
class RobustNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X, training=False):
        # Safe input handling
        X = np.nan_to_num(X)
        
        if training:
            noise = np.random.normal(0, INPUT_NOISE_LEVEL, X.shape)
            X = X + noise
            
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1) 
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2 

    def mutate(self, rate=0.05, strength=0.05):
        if np.random.random() < rate:
            self.W1 += np.random.randn(*self.W1.shape) * strength
            self.b1 += np.random.randn(*self.b1.shape) * strength
            self.W2 += np.random.randn(*self.W2.shape) * strength
            self.b2 += np.random.randn(*self.b2.shape) * strength

# --- Market Environment ---
class MarketEnv:
    def __init__(self, start_price=100.0):
        self.start_price = start_price
        self.reset()

    def reset(self):
        self.prices = [self.start_price]
        self.current_step = 0
        self.volatility = 0.015
        return self._get_state()

    def _get_state(self):
        if len(self.prices) < 30: window = self.prices
        else: window = self.prices[-30:]
        
        # Safety Check: Ensure no zeros in denominator
        last = max(window[-1], 1e-8)
        prev = max(window[-2] if len(window) >=2 else window[-1], 1e-8)
        prev5 = max(window[-6] if len(window) >=6 else window[-1], 1e-8)
        
        ret_1 = np.log(last/prev)
        ret_5 = np.log(last/prev5)
        
        mean_win = np.mean(window) + 1e-8
        vol = np.std(window) / mean_win
        
        return np.array([[ret_1 * 10, ret_5 * 10, vol * 100]])

    def step(self, net_order_flow):
        prev_price = self.prices[-1]
        
        self.volatility = 0.95 * self.volatility + 0.05 * (0.01 + abs(net_order_flow)*0.001)
        self.volatility = min(max(self.volatility, 0.001), 0.2) # Cap vol
        
        shock = np.random.normal(0, self.volatility)
        impact = net_order_flow * 0.005
        
        new_price = prev_price * np.exp(shock + impact)
        new_price = max(new_price, 0.01) # Floor price at penny
        
        self.prices.append(new_price)
        self.current_step += 1
        
        return self._get_state(), new_price, self.volatility, self.current_step >= TRADING_DAYS

# --- Risk-Aware Agent ---
class RiskAgent:
    def __init__(self, id):
        self.id = id
        self.brain = RobustNetwork(3, 8, 2) 
        self.capital = INITIAL_CAPITAL
        self.shares = 0
        self.equity_curve = [INITIAL_CAPITAL]

    def get_equity(self, price):
        return self.capital + (self.shares * price)

    def act(self, state, market_vol):
        outputs = self.brain.forward(state, training=True)
        action_logit = outputs[0, 0]
        conf_logit = outputs[0, 1]
        
        action = 1 if action_logit > 0 else -1
        confidence = 1 / (1 + np.exp(-np.clip(conf_logit, -10, 10)))
        
        safe_vol = max(market_vol, 0.001)
        target_exposure = (confidence * 0.02) / safe_vol 
        target_exposure = np.clip(target_exposure, 0, 1.0) 
        
        return action, target_exposure

    def execute(self, action, exposure, price):
        if price <= 0.01 or np.isnan(price): return 0
        
        current_equity = self.get_equity(price)
        target_value = current_equity * exposure
        
        current_exposure_value = self.shares * price
        diff = 0
        
        if action == 1: diff = target_value - current_exposure_value
        else: diff = (-target_value) - current_exposure_value
            
        try:
            shares_to_trade = int(diff / price)
        except:
            shares_to_trade = 0
        
        if shares_to_trade > 0: 
            cost = shares_to_trade * price
            if self.capital >= cost:
                self.capital -= cost
                self.shares += shares_to_trade
                return shares_to_trade
        elif shares_to_trade < 0: 
            revenue = abs(shares_to_trade) * price
            self.capital += revenue
            self.shares += shares_to_trade 
            return shares_to_trade
        return 0

    def calc_fitness(self):
        curve = np.array(self.equity_curve)
        # Check if curve is flat or invalid
        if len(curve) < 2 or np.all(curve == curve[0]): return -1.0
        
        returns = np.diff(curve) / (curve[:-1] + 1e-8)
        returns = np.nan_to_num(returns)
        
        mean_ret = np.mean(returns) * 252
        
        neg_rets = returns[returns < 0]
        if len(neg_rets) == 0: downside_std = 0.001
        else: downside_std = np.std(neg_rets) * np.sqrt(252)
        
        sortino = mean_ret / (downside_std + 1e-6)
        return sortino

# --- Simulation ---
def run_robust_sim():
    market = MarketEnv()
    population = [RiskAgent(i) for i in range(NUM_AGENTS)]
    
    history_best_equity = []
    history_avg_sortino = []
    history_prices = []
    
    print("Running Robust Sim (Sortino Optimization)...")
    
    for episode in range(NUM_EPISODES):
        state = market.reset()
        
        for ag in population:
            ag.capital = INITIAL_CAPITAL
            ag.shares = 0
            ag.equity_curve = [INITIAL_CAPITAL]
            
        for t in range(TRADING_DAYS):
            price = market.prices[-1]
            vol = market.volatility
            
            net_flow = 0
            for ag in population:
                act, exposure = ag.act(state, vol)
                flow = ag.execute(act, exposure, price)
                net_flow += flow / 100 
                ag.equity_curve.append(ag.get_equity(price))
                
            state, _, _, _ = market.step(net_flow)
            
        population.sort(key=lambda x: x.calc_fitness(), reverse=True)
        
        best_agent = population[0]
        history_best_equity.append(best_agent.equity_curve[-1])
        history_avg_sortino.append(np.mean([a.calc_fitness() for a in population]))
        history_prices.append(market.prices)
        
        survivors = population[:6] 
        new_pop = copy.deepcopy(survivors)
        
        while len(new_pop) < NUM_AGENTS:
            parent = random.choice(survivors)
            child = copy.deepcopy(parent)
            child.brain.mutate(rate=0.1, strength=0.1)
            child.id = len(new_pop)
            new_pop.append(child)
            
        population = new_pop
        
        if episode % 10 == 0:
            print(f"Ep {episode}: Best Sortino: {best_agent.calc_fitness():.2f} | Wealth: ${best_agent.equity_curve[-1]:.0f}")
            
    return history_best_equity, history_avg_sortino, history_prices, best_agent

# --- Visualization ---
def plot_robustness(equity, sortinos, prices, best_agent):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 2)
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(equity, color='green', label='Best Agent Wealth')
    ax1.set_title("Consistent Wealth Growth (No Bankruptcy)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(sortinos, color='blue', label='Population Avg Sortino')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_title("Population Risk-Adjusted Intelligence")
    ax2.set_ylabel("Sortino Ratio")
    
    ax3 = plt.subplot(gs[1, :])
    curve = best_agent.equity_curve
    
    # Normalize market price for comparison
    final_price_arr = np.array(prices[-1])
    if final_price_arr[0] > 0:
        price_norm = final_price_arr / final_price_arr[0] * INITIAL_CAPITAL
    else:
        price_norm = final_price_arr # Fallback
        
    ax3.plot(curve, color='green', lw=2, label='Robust Agent')
    ax3.plot(price_norm, color='gray', alpha=0.5, label='Market Benchmark')
    ax3.fill_between(range(len(curve)), curve, INITIAL_CAPITAL, where=(np.array(curve)<INITIAL_CAPITAL), color='red', alpha=0.3)
    
    ax3.set_title("Final Episode: Agent Stability vs Market Noise")
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    eq, sort, pr, agent = run_robust_sim()
    plot_robustness(eq, sort, pr, agent)