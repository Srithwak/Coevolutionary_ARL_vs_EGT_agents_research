import pandas as pd
import numpy as np
import torch

def get_rsi(close, lookback=14):
    """Calculate Relative Strength Index."""
    ret = close.diff()
    up = ret.clip(lower=0)
    down = -ret.clip(upper=0)
    ema_up = up.ewm(com=lookback - 1, adjust=False).mean()
    ema_down = down.ewm(com=lookback - 1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def get_macd(close, slow=26, fast=12, signal=9):
    """Calculate Moving Average Convergence Divergence."""
    exp1 = close.ewm(span=fast, adjust=False).mean()
    exp2 = close.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def prepare_data_from_csv(symbols: list, csv_path: str, seq_len=30):
    """
    Loads data from CSV, calculates indicators, creates sequences,
    and builds the graph structure.
    """
    try:
        # Alpaca's CSV format has the symbol as a column, which is perfect
        bars = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
        print("Please run the download_data.py script first.")
        return None, None, None, None
        
    all_features = []
    all_labels = []
    processed_symbols = []
    
    for symbol in symbols:
        # Select data for the current symbol from the multi-index DataFrame
        if symbol not in bars['symbol'].unique():
            print(f"Warning: Symbol {symbol} not found in CSV. Skipping.")
            continue
            
        df = bars[bars['symbol'] == symbol].copy()

        if df.empty or len(df) < seq_len + 15: # Need enough data for indicators
            print(f"Warning: Skipping {symbol} due to insufficient data.")
            continue
        
        # Calculate features
        df['rsi'] = get_rsi(df['close'])
        df['macd'], df['macd_signal'] = get_macd(df['close'])
        df.dropna(inplace=True)
        
        features_df = df[['close', 'high', 'low', 'open', 'volume', 'rsi', 'macd', 'macd_signal']]
        
        # Normalize features
        features_df = (features_df - features_df.mean()) / features_df.std()
        
        # Use only the most recent data to create one sample for prediction
        if len(features_df) < seq_len + 1:
            continue
            
        x_sample = features_df.iloc[-(seq_len+1):-1].values
        
        # Create label for the most recent sequence
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
        if price_change < -0.005:
            y_label = 0 # Down
        elif price_change > 0.005:
            y_label = 2 # Up
        else:
            y_label = 1 # Neutral
                
        all_features.append(x_sample)
        all_labels.append(y_label)
        processed_symbols.append(symbol)

    # Create graph structure based on processed symbols
    returns_df = bars.reset_index().pivot(index='timestamp', columns='symbol', values='close').pct_change()
    corr_matrix = returns_df.corr()
    
    edge_index = []
    threshold = 0.6
    symbol_map = {sym: i for i, sym in enumerate(processed_symbols)}

    for i in range(len(processed_symbols)):
        for j in range(i + 1, len(processed_symbols)):
            sym1 = processed_symbols[i]
            sym2 = processed_symbols[j]
            if abs(corr_matrix.loc[sym1, sym2]) > threshold:
                edge_index.append([symbol_map[sym1], symbol_map[sym2]])
                edge_index.append([symbol_map[sym2], symbol_map[sym1]])
    
    if not edge_index:
        print("Warning: No strong correlations found. Graph will be disconnected.")

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x_ts = np.array(all_features)
    labels = np.array(all_labels)

    return torch.tensor(x_ts, dtype=torch.float), torch.tensor(labels, dtype=torch.long), edge_index, processed_symbols
