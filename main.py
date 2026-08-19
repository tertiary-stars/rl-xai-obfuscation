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
        
        print("2. Initializing Custom RL Environment...")
        env_kwargs = {
            "X_data": X_train,
            "target_model": target_model,
            "explainer": explainer,
            "lambda_param": 1.0, # Weight for extraction loss (security)
            "mu_param": 0.05,    # Weight for utility loss (distortion)
        }
        # monitor_dir logs episode rewards to CSV for the learning-curve plot below.
        env = make_vec_env(XAIObfuscationEnv, n_envs=1, env_kwargs=env_kwargs, seed=SEED, monitor_dir=MONITOR_DIR)
        print(" -> Environment initialized.")

        print("3. Setting up PPO Agent (Defender)...")
        # Set seed for PPO agent for reproducible initialization
        agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, seed=SEED)

        print("4. Starting Training Loop (this may take a few minutes)...")
        # 5000 steps is under one PPO rollout - not enough to learn. lambda_param=50 makes obfuscation worth its utility cost.
        agent.learn(total_timesteps=50000)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Complete. Saving agent...")
        agent.save("ppo_xai_defender_test")
        print(" -> Agent saved successfully.")

        print("5. Plotting learning curve...")
        # monitor.csv's first line is a JSON comment (run metadata), so skip it.
        monitor_df = pd.read_csv(f"{MONITOR_DIR}/0.monitor.csv", skiprows=1)
        rolling_reward = monitor_df["r"].rolling(window=10, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rolling_reward)
        ax.set_title("PPO Training Learning Curve", fontsize=16)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward (10-episode rolling mean)")
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("training_learning_curve.png")
        print(" -> Plot saved to training_learning_curve.png. Exiting.")

    except Exception as e:
        print(f"\nERROR ENCOUNTERED: {e}")
        sys.exit(1)