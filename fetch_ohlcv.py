#!/usr/bin/env python
"""
Downloads OHLCV data from Yahoo Finance and saves it to a CSV
in the format required by the simulation.
"""

import yfinance as yf
import argparse
import os
from datetime import datetime

# --- Yahoo Finance Data Limitations ---
# 1. Intraday data (e.g., '1m', '5m') is typically limited to the last 7 days.
# 2. Intervals of '1h' are often available for the last ~2 years.
# 3. Daily data ('1d') is available for many years ('max').
# Be mindful of these when choosing your parameters.


def fetch_data(ticker, period, interval, start, end, output_filename):
    """
    Fetches data from yfinance and saves it to a CSV.
    """
    print(f"Fetching data for ticker: {ticker}")
    print(f"Interval: {interval}")

    # Use start/end dates if provided, otherwise use period
    if start and end:
        print(f"Date Range: {start} to {end}")
        data = yf.download(tickers=ticker, start=start, end=end, interval=interval)
    else:
        print(f"Period: {period}")
        data = yf.download(tickers=ticker, period=period, interval=interval)

    if data.empty:
        print(f"Error: No data found for {ticker} with the specified parameters.")
        print("Please check the ticker symbol and the requested time span/interval.")
        print("Note: 1-minute (1m) data is only available for the last 7 days.")
        return

    # --- Ensure Correct Format ---
    # 1. Reset index to get 'Date' or 'Datetime' as a column
    data.reset_index(inplace=True)

    # 2. Rename the date column to 'Date' for consistency
    if "Datetime" in data.columns:
        data.rename(columns={"Datetime": "Date"}, inplace=True)

    # 3. Check for the required columns our simulation needs
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required_cols):
        print(f"Error: Downloaded data is missing required columns.")
        print(f"Required: {required_cols}")
        print(f"Found: {data.columns.tolist()}")
        return

    # 4. Save to CSV
    try:
        data.to_csv(output_filename, index=False)
        print(f"\nSuccessfully downloaded {len(data)} rows.")
        print(f"Data saved to '{output_filename}'")
    except Exception as e:
        print(f"\nError saving data to file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch OHLCV Data from Yahoo Finance",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--ticker",
        required=True,
        help="Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GC=F' for gold)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="stock_data.csv",
        help="Output CSV file name (default: stock_data.csv)",
    )

    # Time span arguments (mutually exclusive)
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "-p",
        "--period",
        default="1y",
        help="Time period (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', 'max')\n(default: 1y)",
    )
    time_group.add_argument("-s", "--start", help="Start date (YYYY-MM-DD)")

    parser.add_argument("-e", "--end", help="End date (YYYY-MM-DD) (requires --start)")

    parser.add_argument(
        "-i",
        "--interval",
        default="1d",
        help="Data interval (e.g., '1m', '5m', '15m', '1h', '1d', '1wk', '1mo')\n(default: 1d)",
    )

    args = parser.parse_args()

    # Validate start/end date logic
    if args.end and not args.start:
        parser.error("--end requires --start.")

    # If start is used, nullify the period default
    if args.start:
        args.period = None

    fetch_data(
        args.ticker, args.period, args.interval, args.start, args.end, args.output
    )
