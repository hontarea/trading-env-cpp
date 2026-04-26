import numpy as np
import pandas as pd

def generate_synthetic_data(num_bars=10000, base_price=100.0):
    # Create a time index
    t = np.arange(num_bars)
    
    # Create the predictable signal using a sine function
    period = 100
    signal = 10 * np.sin(2 * np.pi * t / period)
    
    noise = np.random.normal(0, 1.0, num_bars)
    
    close_price = base_price + signal + noise
    
    # Generate OHL by simulating intra-bar movement
    open_price = close_price + np.random.normal(0, 0.5, num_bars)
    high_price = np.maximum(open_price, close_price) + np.abs(np.random.normal(0, 0.5, num_bars))
    low_price = np.minimum(open_price, close_price) - np.abs(np.random.normal(0, 0.5, num_bars))
    
    # Create the forecast with cosine function which perfectly predicts the next step's signal direction
    forecast = np.cos(2 * np.pi * t / period)
    
    timestamps =  (t * 60000) # 1 minute intervals
    
    df = pd.DataFrame({
        "timestamp": timestamps.astype(np.int64),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": np.random.randint(100, 10000, num_bars).astype(float),
        "forecast": forecast
    })
    
    # Reorder columns to match C++ expectations
    # {"timestamp", "open", "high", "low", "close", "volume", "forecast"}
    df = df[["timestamp", "open", "high", "low", "close", "volume", "forecast"]]
    
    df.to_csv("data/synthetic_data.csv", index=False)
    print(f"Saved {num_bars} rows to synthetic_data.csv")

if __name__ == "__main__":
    generate_synthetic_data()
