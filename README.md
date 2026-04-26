# C++ Trading Environment for Reinforcement Learning 

## Overview
This project provides a high-performance, Gymnasium-compatible trading environment written in C++20 with Python bindings via `pybind11`. It is designed to train Reinforcement Learning (RL) agents (such as PPO) on market data + additional features (such as forecasts).

## Project Structure
```text
trading_env/
├── include/               # Header files
│   ├── Bar.h              # Market bar and feature mapping
│   ├── Exceptions.h       # Custom exceptions
│   ├── ExecutionModel.h   # Slippage and transaction cost logic
│   ├── MarketData.h       # Data container and feature indexing
│   ├── MarketDataLoader.h # CSV data ingestion
│   ├── Portfolio.h        # Account state and position tracking
│   ├── RewardFunctions.h  # Reward calculation strategies
│   └── TradingEnv.h       # Main C++ environment logic
├── src/                   # Implementation files 
│   ├── ExecutionModel.cpp
│   ├── MarketData.cpp
│   ├── MarketDataLoader.cpp
│   ├── Portfolio.cpp
│   ├── RewardFunctions.cpp
│   └── TradingEnv.cpp
├── data/                  # Market data and scripts
│   ├── generate_data.py   # Synthetic data generation script
│   └── synthetic_data.csv # Sample market dataset
├── bindings.cpp           # Pybind11 bridge definitions
├── gym_wrapper.py         # Gymnasium API wrapper 
├── train_agent.py         # RL training pipeline
├── config.ini             # Doxygen configuration
├── CMakeLists.txt         # Build system configuration
└── environment.yml        # Conda environment specification
```

## Architecture
The project is split into two primary layers:

### 1. The C++ Core 
An engine that handles:
*   **Data Management (`MarketData`, `MarketDataLoader`):** Loads and stores sequential OHLCV + features market bars.
*   **State & Accounting (`Portfolio`):** Tracks cash, equity, and position sizing without look-ahead bias.
*   **Execution (`ExecutionModel`):** Simulates real-world market friction, including bid-ask spread and slippage.
*   **Signals (`RewardFunctions`):** Calculates the reward function for the RL agent.

### 2. The Python Bridge (`trading_env_py`)
The `bindings.cpp` file exposes the C++ components to Python. The `gym_wrapper.py` script wraps the C++ engine in a standard `gymnasium.Env` interface, handling feature normalization and action mapping.

## Data Generation Process
The environment uses a synthetic dataset generator (`generate_data.py`) for verifying the RL pipeline.
*   **Sine Wave (Price):** Simulates predictable market cycles.
*   **Cosine Wave (Forecast):** Acts as the mathematical derivative of the sine wave. It provides the agent with a leading indicator of future direction.
*   **Gaussian Noise:** Added to the signal to prevent the agent from over-fitting to a perfect curve and to simulate real-world market fluctuations. 

## Customizability
The project is designed to be easily adapted for specific research needs:
*   **Environment Configuration:** You can specify constants like `initial_cash`, `start_index`, and `end_index` through the `EnvConfig` struct in Python.
*   **Market Friction:** The `ExecutionModel` parameters (spread, impact, commission) can be adjusted when initializing the environment to simulate different asset classes.
*   **Dynamic Features:** The C++ `Bar` structure is completely dynamic. You can add arbitrary columns (like RSI, MACD, or sentiment scores) to your CSV data. The C++ engine automatically parses and maps these features at runtime, and they can be accessed in Python without any C++ recompilation.
*   **Centralized Constants:** Critical feature names (e.g., `"open"`, `"close"`, `"forecast"`) are defined as C++ `constexpr` constants in `MarketData.h` and automatically bound to Python (e.g., `trading_env_py.FEATURE_CLOSE`). This provides a safe, single source of truth across both languages.
*   **Extensibility:** New reward functions or technical indicators can be added to the core C++ library and exposed via `pybind11` without changing the underlying engine. 

## Quick Start

### 1. Environment Setup
We use [Conda](https://docs.conda.io/en/latest/), an open-source package and environment management system, to isolate dependencies.
```bash
conda env create -f environment.yml
conda activate trading_env
```

### 2. Build
Generate the C++ core and Python bindings:
```bash
cmake -B build -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
cmake --build build
```

### 2*. Test Env
Before running the agent there is an oportunity to run the test script implemented in C++,
do not forget to generate data first:
```
python data/generate_data.py
./build/trading_test data/synthetic_data.csv
```

### 3. Train
Generate synthetic data and start training the agent:
```bash
python data/generate_data.py
python train_agent.py
```
