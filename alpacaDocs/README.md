# Alpaca Options Trading Library

This python code is for using the ALPACA API for options trading. It includes placing PUTS and CALLS, account management, and market daat retrieval.

## Features

- **Simple Options Trading**: Buy calls, puts, and execute multi-leg strategies with single function calls
- **Account Management**: View account info, positions, and order history
- **Market Data**: Real-time quotes, option chains, and historical data
- **Position Management**: Easy position tracking and closing
- **Paper Trading**: Safe testing environment before live trading
- **Streaming Data**: Real-time trade updates and market data (async)

## Installation

### Prerequisites

```bash
pip install alpaca-py python-dotenv pandas
```

### Required Packages
- `alpaca-py` - Alpaca trading API
- `python-dotenv` - Environment variable management
- `pandas` - Data manipulation
- `zoneinfo` - Timezone handling (Python 3.9+)

## Quick Start

### 1. Setup API Keys

Create a `.env` file in your project directory:

```env
ALPACA_PAPER_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_API_KEY=your_alpaca_secret_key_here
```

**Get your API keys from [Alpaca Markets](https://alpaca.markets/)**

### 2. Basic Usage

```python
from alpaca_options_trading import create_trader

# Create trader instance (paper trading by default)
trader = create_trader(paper=True)

# Check account info
account = trader.get_account_info()
print(f"Buying Power: ${account.buying_power}")

# Get current stock price
spy_price = trader.get_underlying_price("SPY")
print(f"SPY Price: ${spy_price}")

# Buy a put option (highest open interest)
order = trader.buy_put_option("SPY", qty=1)
print(f"Order placed: {order.id}")

# Check positions
positions = trader.get_all_positions()
for pos in positions:
    print(f"{pos.symbol}: {pos.qty} shares, P&L: ${pos.unrealized_pl}")
```

## Core Functions

### Account Management

```python
# Get account information
account = trader.get_account_info()

# Get account configuration
config = trader.get_account_config()

# Get all positions
positions = trader.get_all_positions()

# Get specific position
position = trader.get_position("SPY241129P00580000")
```

### Options Trading

```python
# Buy put option with highest open interest
put_order = trader.buy_put_option("SPY", qty=1)

# Buy call option with highest open interest
call_order = trader.buy_call_option("AAPL", qty=2)

# Place a long straddle strategy
straddle_order = trader.place_long_straddle("SPY", qty=1)

# Place custom option order
order = trader.place_option_order(
    symbol="SPY241129P00580000",
    qty=1,
    side=OrderSide.BUY,
    client_order_id="my_custom_id"
)
```

### Position Management

```python
# Close entire position
trader.close_position("SPY241129P00580000")

# Close partial position
trader.close_position("SPY241129P00580000", qty="1")

# Get position details
position = trader.get_position("SPY241129P00580000")
print(f"Cost Basis: ${position.cost_basis}")
print(f"Unrealized P&L: ${position.unrealized_pl}")
```

### Market Data

```python
# Get option contracts
spy_puts = trader.get_option_contracts(
    underlying_symbol="SPY",
    contract_type=ContractType.PUT,
    days_ahead_min=1,
    days_ahead_max=30,
    limit=50
)

# Get option chain
chain = trader.get_option_chain("SPY")

# Get latest quote
quote = trader.get_option_latest_quote("SPY241129P00580000")

# Get latest trade
trade = trader.get_option_latest_trade("SPY241129P00580000")
```

### Order Management

```python
# Get all orders
orders = trader.get_orders(limit=10)

# Get orders for specific symbols
spy_orders = trader.get_orders(symbols=["SPY241129P00580000"])

# Get order by client ID
order = trader.get_order_by_client_id("my_custom_id")
```

## Advanced Features

### Multi-Leg Strategies

```python
# Long Straddle (buy call + put at same strike)
straddle = trader.place_long_straddle(
    underlying_symbol="SPY",
    qty=1,
    strike_range_pct=0.01,  # 1% range around current price
    days_ahead_min=7,       # 7-8 days to expiration
    days_ahead_max=8
)
```

### Contract Filtering

```python
# Get specific contracts
contracts = trader.get_option_contracts(
    underlying_symbol="AAPL",
    contract_type=ContractType.CALL,
    days_ahead_min=30,
    days_ahead_max=45,
    strike_min=150.0,
    strike_max=200.0,
    limit=100
)

# Find contract with highest open interest
best_contract = trader.find_highest_open_interest_contract(contracts)

# Find contract closest to target strike
target_contract = trader.find_nearest_strike_contract(contracts, target_price=175.0)
```

### Streaming Data (Advanced)

```python
import asyncio

async def trade_handler(data):
    print(f"Trade update: {data}")

async def market_data_handler(data):
    print(f"Market data: {data}")

async def main():
    # Stream trade updates
    await trader.stream_trade_updates(trade_handler)
    
    # Stream market data for specific symbols
    symbols = ["SPY241129P00580000"]
    await trader.stream_option_data(symbols, market_data_handler)

# Run async function
asyncio.run(main())
```

## Configuration Options

### Initialize Trader

```python
# Paper trading (default)
trader = create_trader(paper=True)

# Live trading
trader = create_trader(paper=False)

# With custom API keys
trader = AlpacaOptionsTrader(
    api_key="your_key",
    secret_key="your_secret",
    paper=True
)
```

### Environment Variables

Create a `.env` file:

```env
# Required
ALPACA_API_KEY=PKxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxx

# Optional - will use defaults if not specified
ALPACA_PAPER_TRADE=true
```

## Error Handling

```python
try:
    order = trader.buy_put_option("SPY", qty=1)
    print(f"Order successful: {order.id}")
except ValueError as e:
    print(f"Validation error: {e}")
except APIError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Important Notes

### Trading Hours & Expiration Rules
- **Option orders on expiration day**: Must be submitted before 3:15 PM ET (3:30 PM for broad-based ETFs like SPY/QQQ)
- **Auto-liquidation**: Expiring positions are auto-closed starting at 3:30 PM ET (3:45 PM for ETFs)
- **Paper trading**: Always test strategies in paper mode first

### Risk Management
- Start with paper trading to test strategies
- Always monitor positions regularly
- Be aware of options expiration dates
- Understand the risks of options trading

## Example Strategies

### 1. Simple Put Purchase

```python
# Buy protective put
trader = create_trader(paper=True)
put_order = trader.buy_put_option("SPY", qty=1)
print(f"Protective put purchased: {put_order.id}")
```

### 2. Long Straddle

```python
# Profit from high volatility
straddle = trader.place_long_straddle("AAPL", qty=1)
print(f"Straddle placed: {straddle.id}")
```

### 3. Systematic Trading

```python
# Get high open interest contracts
contracts = trader.get_option_contracts("SPY", ContractType.PUT)
best_contract = trader.find_highest_open_interest_contract(contracts)

if best_contract:
    order = trader.place_option_order(best_contract.symbol, qty=1)
    print(f"Systematic order placed: {order.id}")
```

## References

- **Alpaca API Documentation**: [https://docs.alpaca.markets/](https://docs.alpaca.markets/)
- **Alpaca Support**: [https://support.alpaca.markets/](https://support.alpaca.markets/)
- **Python alpaca-py**: [https://github.com/alpacahq/alpaca-py](https://github.com/alpacahq/alpaca-py)
- **Github Code**: [https://github.com/alpacahq/alpaca-py/tree/master/examples/options](https://github.com/alpacahq/alpaca-py/tree/master/examples/options)
- **Alpaca Dashboard**: [https://app.alpaca.markets/dashboard/overview](https://app.alpaca.markets/dashboard/overview)
---
