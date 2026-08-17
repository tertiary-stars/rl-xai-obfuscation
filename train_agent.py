import sys
from datetime import datetime
import numpy as np
import random
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment import XAIObfuscationEnv
from src.utils import load_and_train_target

SEED = 42

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
            "lambda_param": 5.0,
            "mu_param": 1.0,
        }
        env = make_vec_env(XAIObfuscationEnv, n_envs=1, env_kwargs=env_kwargs, seed=SEED)
        print(" -> Environment initialized.")
        
        print("3. Setting up PPO Agent (Defender)...")
        # Set seed for PPO agent for reproducible initialization
        agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, seed=SEED)
        
        print("4. Starting Training Loop (this may take a moment)...")
        agent.learn(total_timesteps=5000) # Increased timesteps for better training
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Complete. Saving agent...")
        agent.save("ppo_xai_defender_test")
        print(" -> Agent saved successfully. Exiting.")
        
    except Exception as e:
        print(f"\nERROR ENCOUNTERED: {e}")
        sys.exit(1)