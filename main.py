# This file runs the training process.
# Previously known as `train_agent.py`.
# The classes `Adversary` and `XAIObfuscationEnv` are in `src/environment.py`.
# The function `load_and_train_target` is in `src/utils.py`.

import sys
from datetime import datetime
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment import XAIObfuscationEnv
from src.utils import load_and_train_target

SEED = 42
MONITOR_DIR = "training_logs"

if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting execution...")
    
    try:
        print("1. Loading dataset and training target model...")
        X_train, X_test, target_model, explainer = load_and_train_target()
        print(" -> Target model trained successfully.")
        
        ablation_pairs = [
            (0.0, 1.0),   # Extreme Utility
            (1.0, 0.5),   # High Utility
            (1.0, 0.2),   # Moderate-High Utility
            (1.0, 0.1),   # Moderate Utility
            (1.0, 0.05),  # Current Baseline
            (1.0, 0.01),  # High Security
            (1.0, 0.0)    # Extreme Security
        ]

        for lam, mu in ablation_pairs:
            print(f"\n--- Training agent with lambda={lam}, mu={mu} ---")
            
            print("2. Initializing Custom RL Environment...")
            env_kwargs = {
                "X_data": X_train,
                "target_model": target_model,
                "explainer": explainer,
                "lambda_param": lam,
                "mu_param": mu,
            }
            # Use a unique log directory for each pair if we want to keep monitor files
            log_dir = f"{MONITOR_DIR}/l{lam}_m{mu}"
            import os
            os.makedirs(log_dir, exist_ok=True)
            
            env = make_vec_env(XAIObfuscationEnv, n_envs=1, env_kwargs=env_kwargs, seed=SEED, monitor_dir=log_dir)
            print(" -> Environment initialized.")

            print("3. Setting up PPO Agent (Defender)...")
            agent = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, seed=SEED)

            print("4. Starting Training Loop...")
            agent.learn(total_timesteps=50000)
            
            model_name = f"ppo_xai_defender_l{lam}_m{mu}"
            agent.save(model_name)
            print(f" -> Agent saved as {model_name}.")

            # Optional: Plot learning curve for this specific run
            monitor_df = pd.read_csv(f"{log_dir}/0.monitor.csv", skiprows=1)
            rolling_reward = monitor_df["r"].rolling(window=10, min_periods=1).mean()
            plt.figure(figsize=(10, 6))
            plt.plot(rolling_reward)
            plt.title(f"PPO Training: lambda={lam}, mu={mu}")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.savefig(f"{log_dir}/learning_curve.png")
            plt.close()

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] All training runs complete.")

    except Exception as e:
        print(f"\nERROR ENCOUNTERED: {e}")
        sys.exit(1)