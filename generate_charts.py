import os
import pandas as pd
import mplfinance as mpf

def generate_charts(folder='market_data'):
    folder_path = os.path.join(os.getcwd(), folder)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    for file in csv_files:
        path = os.path.join(folder_path, file)
        data = pd.read_csv(path, parse_dates=['Date'], header=0, skiprows=[1])
        data.set_index('Date', inplace=True)

        # Convert columns
        for col in ['Open','High','Low','Close','Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(inplace=True)

        # Save chart
        out_path = os.path.join(folder_path, file.replace('.csv','.png'))
        mpf.plot(data, type='candle', volume=True, mav=(20,50),
                 style='charles', savefig=out_path)

if __name__ == "__main__":
    generate_charts()
