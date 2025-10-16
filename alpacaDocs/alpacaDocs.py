import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
from dotenv import load_dotenv
import os
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.stock import StockHistoricalDataClient, StockLatestTradeRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.option import OptionDataStream

from alpaca.data.requests import (
    OptionBarsRequest, OptionTradesRequest, OptionLatestQuoteRequest,
    OptionLatestTradeRequest, OptionSnapshotRequest, OptionChainRequest
)
from alpaca.trading.requests import (
    GetOptionContractsRequest, GetAssetsRequest, MarketOrderRequest,
    GetOrdersRequest, ClosePositionRequest, OptionLegRequest
)
from alpaca.trading.enums import (
    AssetStatus, ExerciseStyle, OrderSide, OrderClass, OrderType,
    TimeInForce, QueryOrderStatus, ContractType
)
from alpaca.common.exceptions import APIError


class AlpacaOptionsTrader:
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        load_dotenv()
        
        self.api_key = api_key or os.environ.get('ALPACA_PAPER_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_PAPER_API_SECRET')
        self.paper = paper
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API keys not found. Please provide them or set in .env file")
        
        # Initialize clients
        self.trade_client = TradingClient(
            api_key=self.api_key, 
            secret_key=self.secret_key, 
            paper=self.paper
        )
        self.stock_data_client = StockHistoricalDataClient(
            api_key=self.api_key, 
            secret_key=self.secret_key
        )
        self.option_data_client = OptionHistoricalDataClient(
            api_key=self.api_key, 
            secret_key=self.secret_key
        )
    
    # Account Management Functions
    def get_account_info(self) -> Dict[str, Any]:
        return self.trade_client.get_account()
    
    def get_account_config(self) -> Dict[str, Any]:
        return self.trade_client.get_account_configurations()
    
    def get_options_enabled_assets(self) -> List[Any]:
        """Get list of assets that are options enabled"""
        req = GetAssetsRequest(attributes="options_enabled")
        return self.trade_client.get_all_assets(req)
    
    # Options Contract Functions
    def get_option_contracts(self, underlying_symbol: str, 
                           contract_type: ContractType = None,
                           days_ahead_min: int = 1,
                           days_ahead_max: int = 60,
                           strike_min: float = None,
                           strike_max: float = None,
                           limit: int = 100) -> List[Any]:
        now = datetime.now(tz=ZoneInfo("America/New_York"))
        day_min = now + timedelta(days=days_ahead_min)
        day_max = now + timedelta(days=days_ahead_max)
        
        req = GetOptionContractsRequest(
            underlying_symbol=[underlying_symbol],
            status=AssetStatus.ACTIVE,
            expiration_date_gte=day_min.date(),
            expiration_date_lte=day_max.strftime(format="%Y-%m-%d"),
            type=contract_type,
            style=ExerciseStyle.AMERICAN,
            strike_price_gte=strike_min,
            strike_price_lte=strike_max,
            limit=limit
        )
        
        res = self.trade_client.get_option_contracts(req)
        return res.option_contracts
    
    def get_contract_by_symbol(self, symbol: str):
        """Get option contract by symbol"""
        return self.trade_client.get_option_contract(symbol)
    
    def find_highest_open_interest_contract(self, contracts: List[Any]):
        """Find the contract with highest open interest from a list"""
        max_open_interest = 0
        best_contract = None
        
        for contract in contracts:
            if contract.open_interest is not None:
                try:
                    oi = int(contract.open_interest)
                    if oi > max_open_interest:
                        max_open_interest = oi
                        best_contract = contract
                except ValueError:
                    continue
        
        return best_contract
    
    def get_underlying_price(self, symbol: str) -> float:
        """Get the latest price of the underlying stock"""
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        response = self.stock_data_client.get_stock_latest_trade(req)
        return response[symbol].price
    
    def find_nearest_strike_contract(self, contracts: List[Any], target_price: float):
        min_diff = float('inf')
        best_contract = None
        
        for contract in contracts:
            diff = abs(float(contract.strike_price) - target_price)
            if diff < min_diff:
                min_diff = diff
                best_contract = contract
        
        return best_contract
    
    # Trading Functions
    def place_option_order(self, symbol: str, qty: int = 1, 
                          side: OrderSide = OrderSide.BUY,
                          order_type: OrderType = OrderType.MARKET,
                          client_order_id: str = None) -> Any:
        if client_order_id is None:
            client_order_id = f"order_{uuid.uuid4().hex[:8]}"
        
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id
        )
        
        return self.trade_client.submit_order(req)
    
    def buy_put_option(self, underlying_symbol: str, qty: int = 1) -> Any:
        # Get put contracts
        contracts = self.get_option_contracts(
            underlying_symbol, 
            contract_type=ContractType.PUT
        )
        
        if not contracts:
            raise ValueError(f"No put contracts found for {underlying_symbol}")
        
        # Find highest open interest contract
        best_contract = self.find_highest_open_interest_contract(contracts)
        
        if not best_contract:
            raise ValueError("No suitable contract found")
        
        return self.place_option_order(best_contract.symbol, qty, OrderSide.BUY)
    
    def buy_call_option(self, underlying_symbol: str, qty: int = 1) -> Any:
        # Get call contracts
        contracts = self.get_option_contracts(
            underlying_symbol, 
            contract_type=ContractType.CALL
        )
        
        if not contracts:
            raise ValueError(f"No call contracts found for {underlying_symbol}")
        
        # Find highest open interest contract
        best_contract = self.find_highest_open_interest_contract(contracts)
        
        if not best_contract:
            raise ValueError("No suitable contract found")
        
        return self.place_option_order(best_contract.symbol, qty, OrderSide.BUY)
    
    def place_long_straddle(self, underlying_symbol: str, qty: int = 1,
                           strike_range_pct: float = 0.01,
                           days_ahead_min: int = 7,
                           days_ahead_max: int = 8) -> Any:
        # Get underlying price
        underlying_price = self.get_underlying_price(underlying_symbol)
        
        # Define strike range
        min_strike = underlying_price * (1 - strike_range_pct)
        max_strike = underlying_price * (1 + strike_range_pct)
        
        order_legs = []
        
        # Get both call and put contracts
        for contract_type in [ContractType.CALL, ContractType.PUT]:
            contracts = self.get_option_contracts(
                underlying_symbol,
                contract_type=contract_type,
                days_ahead_min=days_ahead_min,
                days_ahead_max=days_ahead_max,
                strike_min=min_strike,
                strike_max=max_strike
            )
            
            if contracts:
                best_contract = self.find_nearest_strike_contract(contracts, underlying_price)
                if best_contract:
                    order_legs.append(OptionLegRequest(
                        symbol=best_contract.symbol,
                        side=OrderSide.BUY,
                        ratio_qty=qty
                    ))
        
        if len(order_legs) != 2:
            raise ValueError("Could not find suitable contracts for both call and put")
        
        req = MarketOrderRequest(
            qty=qty,
            order_class=OrderClass.MLEG,
            time_in_force=TimeInForce.DAY,
            legs=order_legs
        )
        
        return self.trade_client.submit_order(req)
    
    # Position Management Functions
    def get_all_positions(self) -> List[Any]:
        return self.trade_client.get_all_positions()
    
    def get_position(self, symbol: str) -> Any:
        return self.trade_client.get_open_position(symbol_or_asset_id=symbol)
    
    def close_position(self, symbol: str, qty: str = None) -> Any:
        close_request = ClosePositionRequest(qty=qty) if qty else None
        return self.trade_client.close_position(
            symbol_or_asset_id=symbol,
            close_options=close_request
        )
    
    # Order Management Functions
    def get_orders(self, symbols: List[str] = None, limit: int = 50) -> List[Any]:
        req = GetOrdersRequest(
            status=QueryOrderStatus.ALL,
            symbols=symbols,
            limit=limit
        )
        return self.trade_client.get_orders(req)
    
    def get_order_by_client_id(self, client_id: str) -> Any:
        return self.trade_client.get_order_by_client_id(client_id=client_id)
    
    # Market Data Functions
    def get_option_latest_quote(self, symbol: str) -> Any:
        req = OptionLatestQuoteRequest(symbol_or_symbols=[symbol])
        return self.option_data_client.get_option_latest_quote(req)
    
    def get_option_latest_trade(self, symbol: str) -> Any:
        req = OptionLatestTradeRequest(symbol_or_symbols=[symbol])
        return self.option_data_client.get_option_latest_trade(req)
    
    def get_option_chain(self, underlying_symbol: str) -> Any:
        req = OptionChainRequest(underlying_symbol=underlying_symbol)
        return self.option_data_client.get_option_chain(req)
    
    # Streaming Functions (Advanced)
    async def stream_trade_updates(self, handler_func):
        trade_stream = TradingStream(self.api_key, self.secret_key, paper=self.paper)
        trade_stream.subscribe_trade_updates(handler_func)
        await trade_stream._run_forever()
    
    async def stream_option_data(self, symbols: List[str], handler_func):
        option_stream = OptionDataStream(self.api_key, self.secret_key)
        option_stream.subscribe_quotes(handler_func, *symbols)
        option_stream.subscribe_trades(handler_func, *symbols)
        await option_stream._run_forever()


# Convenience Functions
def create_trader(paper: bool = True) -> AlpacaOptionsTrader:
    """Create a new AlpacaOptionsTrader instance"""
    return AlpacaOptionsTrader(paper=paper)


def main():
    trader = create_trader(paper=True)
    
    try:
        # Check account info
        print("=== Account Info ===")
        account = trader.get_account_info()
        print(f"Account Status: {account.status}")
        print(f"Buying Power: ${account.buying_power}")
        
        # Get options-enabled assets
        print("\n=== Options Enabled Assets (first 5) ===")
        assets = trader.get_options_enabled_assets()
        for asset in assets[:5]:
            print(f"{asset.symbol}: {asset.name}")
        
        # Example: Get SPY put options
        print("\n=== SPY Put Options ===")
        spy_puts = trader.get_option_contracts("SPY", ContractType.PUT, limit=5)
        for contract in spy_puts:
            print(f"Symbol: {contract.symbol}, Strike: {contract.strike_price}, "
                  f"Expiry: {contract.expiration_date}, OI: {contract.open_interest}")
        
        # Example: Get current SPY price
        print(f"\n=== Current SPY Price ===")
        spy_price = trader.get_underlying_price("SPY")
        print(f"SPY Price: ${spy_price}")
        
        # Example: Buy a put option 
        # print("\n=== Buying SPY Put Option ===")
        # order = trader.buy_put_option("SPY", qty=1)
        # print(f"Order placed: {order.id}")
        
        # Example: Place a long straddle 
        # print("\n=== Placing Long Straddle ===")
        # straddle_order = trader.place_long_straddle("SPY", qty=1)
        # print(f"Straddle order placed: {straddle_order.id}")
        
        # Get positions 
        print("\n=== Current Positions ===")
        positions = trader.get_all_positions()
        if positions:
            for pos in positions:
                print(f"Symbol: {pos.symbol}, Qty: {pos.qty}, "
                      f"Cost Basis: ${pos.cost_basis}, P&L: ${pos.unrealized_pl}")
        else:
            print("No open positions")
        
        # Get recent orders
        print("\n=== Recent Orders ===")
        orders = trader.get_orders(limit=5)
        for order in orders:
            print(f"Order ID: {order.id}, Symbol: {order.symbol}, "
                  f"Side: {order.side}, Status: {order.status}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

