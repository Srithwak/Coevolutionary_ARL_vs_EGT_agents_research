# --- Diagnostic Step 1: Confirm the script is running ---
print("--- Python script has started ---")

import torch
import torch.optim as optim
import torch.nn.functional as F

# --- Diagnostic Step 2: Confirm modules are being imported ---
try:
    from model import StockPredictor
    from data_handler import prepare_data_from_csv
    print("--- Successfully imported model and data_handler modules ---")
except ImportError as e:
    print(f"---!!! IMPORT ERROR: Could not import a required file. Error: {e} !!!---")
    # Exit here if imports fail, so the error message is clearly visible.
    import sys
    sys.exit()


def main():
    # --- Configuration ---
    # NOTE: This list should contain the stocks you downloaded.
    SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ']
    
    # NOTE: This should match the filename from your download script.
    CSV_PATH = 'stock_data.csv' 
    
    SEQ_LEN = 30 # Use last 30 days of data for each sequence

    # --- Model Hyperparameters ---
    INPUT_DIM = 8  # close, high, low, open, volume, rsi, macd, macd_signal
    MODEL_DIM = 64
    NUM_HEADS_TRANSFORMER = 4
    NUM_LAYERS_TRANSFORMER = 2
    NUM_HEADS_GAT = 4
    NUM_CLASSES = 3 # Down, Neutral, Up
    LEARNING_RATE = 0.001
    EPOCHS = 50

    # 1. Prepare Data from your CSV
    print(f"Loading and preparing data from {CSV_PATH}...")
    x_ts, labels, edge_index, processed_symbols = prepare_data_from_csv(SYMBOLS, CSV_PATH, SEQ_LEN)
    
    if x_ts is None or x_ts.shape[0] == 0:
        print("Data preparation failed or resulted in no data. Exiting.")
        return
        
    print(f"Data prepared for {len(processed_symbols)} symbols. Tensor shape: {x_ts.shape}")

    # --- FIX: Add a check to ensure there are enough stocks for the graph model ---
    if len(processed_symbols) <= 1:
        print("\n---!!! MODEL ERROR: The Graph Attention Network requires at least 2 stocks to build relationships. !!!---")
        print("Please check your 'stock_data.csv' to ensure it contains data for multiple symbols from the SYMBOLS list.")
        print("Exiting.")
        return # Exit the main function gracefully
        
    # 2. Initialize Model and Optimizer
    model = StockPredictor(
        input_dim=INPUT_DIM, model_dim=MODEL_DIM,
        num_heads_transformer=NUM_HEADS_TRANSFORMER, num_layers_transformer=NUM_LAYERS_TRANSFORMER,
        num_heads_gat=NUM_HEADS_GAT, num_classes=NUM_CLASSES
    )
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print("\nStarting model training...")
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(x_ts, edge_index)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}')

    # 4. Get and Display Predictions
    print("\n--- Generating Final Predictions ---")
    model.eval()
    with torch.no_grad():
        predictions_logits = model(x_ts, edge_index)
        predictions = torch.argmax(predictions_logits, dim=1)

    print("\n--- Model Predictions ---")    
    label_map = {0: "DOWN 📉", 1: "NEUTRAL 횡", 2: "UP 📈"}
    for i, symbol in enumerate(processed_symbols):
        pred_label = label_map[predictions[i].item()]
        print(f"Prediction for {symbol}: {pred_label}")

if __name__ == "__main__":
    # --- Diagnostic Step 3: Confirm main function is being called ---
    print("--- Calling main() function ---")
    main()

