import os
import pandas as pd
import mplfinance as mpf
import sys

# --- Prerequisites ---
# pip install pandas matplotlib mplfinance
# ---------------------

def plot_market_data(folder_name='market_data'):
    """
    Finds .csv files in folder_name, asks user to choose one, 
    and plots a candlestick chart with volume.
    """
    
    # --- 1. Define folder path relative to this script ---
    try:
        # Get directory containing this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for environments where __file__ is not defined
        script_dir = os.getcwd() 
        
    folder_path = os.path.join(script_dir, folder_name)
    
    # --- 2. Check if directory exists ---
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' not found.", file=sys.stderr)
        return

    # --- 3. Find all CSV files ---
    try:
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]
    except Exception as e:
        print(f"Error reading directory {folder_path}: {e}", file=sys.stderr)
        return

    if not csv_files:
        print(f"No .csv files found in '{folder_path}'.")
        return

    # --- 4. Ask the user which file to plot ---
    print("Found CSV files:")
    for i, file_name in enumerate(csv_files):
        print(f"  {i + 1}: {file_name}")
    
    selected_file_name = None
    while selected_file_name is None:
        try:
            choice_str = input(f"Enter file number to plot (1-{len(csv_files)}): ")
            choice_int = int(choice_str)
            
            if 1 <= choice_int <= len(csv_files):
                selected_file_name = csv_files[choice_int - 1]
            else:
                print(f"Invalid choice (1-{len(csv_files)}).", file=sys.stderr)
        except ValueError:
            print("Invalid input. Please enter a number.", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            print("\nPlotting cancelled.")
            return

    # --- 5. Process the selected file ---
    print(f"Processing {selected_file_name}...")
    try:
        file_path = os.path.join(folder_path, selected_file_name)
        
        # Read CSV, skipping the second row (index 1)
        data = pd.read_csv(file_path, parse_dates=['Date'], header=0, skiprows=[1])
        
        data.set_index('Date', inplace=True)
        
        # Ensure required columns are numeric
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: '{col}'")
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data.dropna(inplace=True) # Drop rows with bad data
        data.sort_index(inplace=True) # Ensure chronological order

        if data.empty:
            print(f"Skipping {selected_file_name}: No valid data.")
            return

        print(f"Plotting {selected_file_name}...")

        # --- 6. Generate the plot ---
        mpf.plot(data, 
                 type='candle', 
                 style='charles',
                 title=f'Price and Volume for {selected_file_name}',
                 ylabel='Price ($)',
                 ylabel_lower='Volume',
                 volume=True,
                 mav=(20, 50),
                 figratio=(16,9))

    except Exception as e:
        print(f"Error processing file {selected_file_name}: {e}", file=sys.stderr)

if __name__ == "__main__":
    plot_market_data()
    print("Script finished.")

