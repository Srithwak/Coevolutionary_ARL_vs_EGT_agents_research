import pandas as pd
import numpy as np
import os

def generate_market_data(filename="market_data.csv", steps=1000, start_price=100.0, mu=0.0001, sigma=0.01):
    """
    Generates a realistic market price CSV using Geometric Brownian Motion (GBM).
    
    :param filename: The name of the CSV file to create.
    :param steps: The number of time steps (rows) to generate.
    :param start_price: The starting price of the asset.
    :param mu: The drift (average daily return).
    :param sigma: The volatility (standard deviation of returns).
    """
    
    # if os.path.exists(filename):
    #     print(f"'{filename}' already exists. Skipping generation.")
    #     return

    print(f"Generating '{filename}' with {steps} steps...")
    
    prices = [start_price]
    
    for _ in range(steps - 1):
        # GBM formula: S_t = S_{t-1} * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)
        # We simplify with dt=1 (one step)
        dt = 1
        Z = np.random.standard_normal()
        new_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(new_price)
    
    df = pd.DataFrame(prices, columns=["mid_price"])
    df.to_csv(filename, index=False)
    print(f"Successfully created '{filename}'.")

if __name__ == "__main__":
    # This script will create the 'market_data.csv' file needed by main.py
    # You only need to run this script once.
    generate_market_data(steps=1000, start_price=100.0, mu=0.0001, sigma=0.01)
