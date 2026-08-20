import sys
import os
from datetime import datetime
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.environment import XAIObfuscationEnv
from src.utils import load_and_train_credit_dnn

SEED = 42
MONITOR_DIR = "training_logs_credit"

if __name__ == "__main__":
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting execution for Credit Card pipeline...")
    
    try:
        X_train, X_test, target_model, explainer = load_and_train_credit_dnn()
        
        ablation_pairs = [(0.0, 1.0), (1.0, 0.5), (1.0, 0.2), (1.0, 0.1), (1.0, 0.05), (1.0, 0.01), (1.0, 0.0)]

        for lam, mu in ablation_pairs:
            print(f"\n--- Training credit agent with lambda={lam}, mu={mu} ---")
            
            env_kwargs = {
                "X_data": X_train, "target_model": target_model,
                "explainer": explainer, "lambda_param": lam, "mu_param": mu,
            }
            log_dir = f"{MONITOR_DIR}/l{lam}_m{mu}"
            os.makedirs(log_dir, exist_ok=True)
            
            env = make_vec_env(XAIObfuscationEnv, n_envs=1, env_kwargs=env_kwargs, seed=SEED, monitor_dir=log_dir)
            agent = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, seed=SEED)
            agent.learn(total_timesteps=50000)
            
            model_name = f"ppo_xai_defender_credit_l{lam}_m{mu}"
            agent.save(model_name)
            
    except Exception as e:
        print(f"\nERROR ENCOUNTERED: {e}")
        sys.exit(1)