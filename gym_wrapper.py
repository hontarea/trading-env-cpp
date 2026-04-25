import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add the build directory to the path so we can import the C++ module
sys.path.append(os.path.join(os.getcwd(), 'build'))

try:
    import trading_env_py as env_py
except ImportError as e:
    print(f"Error: Could not import trading_env_py: {e}")
    print(f"Looked in: {os.path.join(os.getcwd(), 'build')}")
    print(f"Files in build: {os.listdir('build') if os.path.exists('build') else 'build folder missing'}")
    sys.exit(1)

class GymTradingEnv(gym.Env):
    """
    A Gymnasium wrapper for the C++ Trading Environment.
    """
    def __init__(self, data_path, initial_cash=10000.0):
        super(GymTradingEnv, self).__init__()

        # Load data and setup C++ components
        self.data = env_py.MarketDataLoader.load_csv(data_path)
        
        # Hyperparameters for execution and reward
        self.exec_model = env_py.ExecutionModel(0.00001, 0.00001, 0.0001)
        self.reward_fn = env_py.log_return_reward()
        
        self.config = env_py.EnvConfig()
        self.config.initial_cash = initial_cash

        # Fetch feature indices
        self.open_idx = self.data.get_feature_index("open")
        self.close_idx = self.data.get_feature_index("close")
        self.forecast_idx = self.data.get_feature_index("forecast")
        
        # Initialize the underlying C++ Environment
        self._env = env_py.TradingEnv(
            self.data, 
            self.exec_model, 
            self.reward_fn, 
            self.config
        )

        # Define Action Space: 0 (Short), 1 (Flat), 2 (Long)
        self.action_space = spaces.Discrete(3)

        # Define Observation Space:
        # [last_log_return*100, short_window_mean_return*100, pos, return_inception, cash_ratio, time_frac, forecast]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(7,), 
            dtype=np.float32
        )

    def _get_obs_array(self, obs_struct):
        return_since_inception = (obs_struct.equity / self.config.initial_cash) - 1.0
        cash_ratio = obs_struct.cash / obs_struct.equity if obs_struct.equity > 0 else 1.0
        time_fraction = obs_struct.steps_remaining / len(self.data)

        return np.array([
            obs_struct.last_log_return * 100.0,
            obs_struct.short_window_mean_return * 100.0,
            float(obs_struct.current_position),
            return_since_inception,
            cash_ratio,
            time_fraction,
            obs_struct.forecast
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Handle seeding if needed (C++ env currently deterministic)
        super().reset(seed=seed)
        
        obs_struct = self._env.reset()
        observation = self._get_obs_array(obs_struct)
        
        info = {}
        return observation, info

    def step(self, action):
        # Map Gym action [0, 1, 2] to C++ action [-1, 0, 1]
        cpp_action = action - 1
        
        result = self._env.step(cpp_action)
        
        observation = self._get_obs_array(result.observation)
        reward = result.reward
        terminated = result.terminated
        truncated = result.truncated 
        
        info = {
            "equity": self._env.portfolio().equity,
            "position": self._env.portfolio().position,
            "step": self._env.current_step()
        }
        
        return observation, reward, terminated, truncated, info

