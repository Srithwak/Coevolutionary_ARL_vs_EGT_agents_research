import os
import pandas as pd
import time
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# --- Configuration ---
# This list of stock symbols MUST match the list used in your main.py file.
# The model's graph structure depends on this list being consistent.
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ']
# This is the name of the file where all the downloaded data will be saved.
OUTPUT_FILENAME = 'stock_data.csv'

# --- Load API Key ---
# This loads your secret API key from a .env file in the same folder.
# This is a secure way to manage credentials without hardcoding them in the script.
load_dotenv()
API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')

# Check to ensure the API key was successfully loaded before proceeding.
if not API_KEY:
    raise ValueError("Alpha Vantage API key not found. Please create a .env file and add your key.")

print(f"--- Starting Download for {len(SYMBOLS)} Stocks using Alpha Vantage ---")

# --- Robust Download Logic ---
# We'll store the data for each successfully downloaded stock in this list.
all_data_frames = []
# Initialize the Alpha Vantage client with your API key.
ts = TimeSeries(key=API_KEY, output_format='pandas')

# Loop through each symbol to download its data individually.
# This is more reliable than a single bulk request.
for symbol in SYMBOLS:
    try:
        print(f"\nDownloading full historical data for {symbol}...")
        # Fetch the daily data. 'outputsize=full' gets the entire history (up to 20+ years).
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        
        if data.empty:
            print(f"  -> No data found for {symbol}, skipping.")
            continue
        
        # Add a 'symbol' column so we can identify this stock's data later.
        data['symbol'] = symbol
        all_data_frames.append(data)
        print(f"  -> Successfully downloaded {len(data)} data points for {symbol}.")
        
        # The free Alpha Vantage API key has a limit of 5 calls per minute.
        # Pausing for 15 seconds ensures we stay well within this limit to avoid being blocked.
        print("  -> Pausing for 15 seconds to respect API rate limit...")
        time.sleep(15) 

    except Exception as e:
        print(f"  -> ERROR: Could not download data for {symbol}. Reason: {e}")
# --------------------------------

# --- Data Compilation ---
# This section only runs if at least one stock was successfully downloaded.
if not all_data_frames:
    print("\nDownload failed for all symbols. Please check your API key and internet connection.")
else:
    print("\n--- Compiling all downloaded data into a single file ---")
    # Combine all the individual stock DataFrames into one large one.
    full_df = pd.concat(all_data_frames)
    
    # Clean and format the combined dataframe for consistency.
    full_df.index.name = 'timestamp'
    full_df.reset_index(inplace=True)
    full_df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    }, inplace=True)
    
    # Ensure we only have the columns our model expects.
    full_df = full_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]
    
    # Sort the data first by symbol, then by date. This is crucial for creating correct time-series sequences.
    full_df.sort_values(by=['symbol', 'timestamp'], inplace=True)

    # Save the final, clean dataset to the specified CSV file.
    full_df.to_csv(OUTPUT_FILENAME, index=False)
    
    # --- Final Summary ---
    print(f"\n✅ Data for {len(all_data_frames)} symbols saved successfully to {OUTPUT_FILENAME}")
    print("--- Summary of the final dataset ---")
    print(f"Total rows: {len(full_df)}")
    print(f"Date range: {full_df['timestamp'].min().strftime('%Y-%m-%d')} to {full_df['timestamp'].max().strftime('%Y-%m-%d')}")
    print(f"Symbols included: {', '.join(full_df['symbol'].unique())}")

