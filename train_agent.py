from cmath import exp
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from gym_wrapper import GymTradingEnv
import os

def train():
    # Create the environment
    data_path = "data/synthetic_data.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run generate_data.py first.")
        return

    env = GymTradingEnv(data_path)

    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10
    )

    # Train the agent
    print("Starting training...")
    model.learn(total_timesteps=10_000)  

    print("\nTesting the trained model...")
    obs, info = env.reset()
    total_reward = 0
    history = []
    done = False
    # 0: Short, 1: Flat, 2: Long
    action_counts = {0: 0, 1: 0, 2: 0} 
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # Action is a numpy scalar, convert to int for dict key
        action_item = int(action)
        action_counts[action_item] += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        history.append(exp(total_reward).real - 1.0) 
        done = terminated or truncated
        
    total_reward = exp(total_reward).real - 1.0  
    print(f"Test Final Equity: {info['equity']:.2f}")
    print(f"Test Total Reward: {total_reward:.6f}")
    
    total_actions = sum(action_counts.values())
    print(f"Action Distribution: Short: {action_counts[0]/total_actions:.1%}, "
          f"Flat: {action_counts[1]/total_actions:.1%}, "
          f"Long: {action_counts[2]/total_actions:.1%}")
    
    plot_equity_curve(history) 

    
def plot_equity_curve(history):
    plt.plot(history)
    plt.title("Equity Curve")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Return")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    train()
