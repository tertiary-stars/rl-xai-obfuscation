import sys
from datetime import datetime

from stable_baselines3 import PPO

from src.environment import XAIObfuscationEnv
from src.utils import load_and_train_target

if __name__ == "__main__":
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting execution...")
    
    try:
        print("1. Loading dataset and training target model...")
        X_train, X_test, target_model, explainer = load_and_train_target()
        print(" -> Target model trained successfully.")
        
        print("2. Initializing Custom RL Environment...")
        env = XAIObfuscationEnv(
            X_data=X_train, 
            target_model=target_model, 
            explainer=explainer, 
            lambda_param=5.0,
            mu_param=1.0
        )
        print(" -> Environment initialized.")
        
        print("3. Setting up PPO Agent (Defender)...")
        agent = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
        
        print("4. Starting Training Loop (this may take a moment)...")
        agent.learn(total_timesteps=1000)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Training Complete. Saving agent...")
        agent.save("ppo_xai_defender_test")
        print(" -> Agent saved successfully. Exiting.")
        
    except Exception as e:
        print(f"\nERROR ENCOUNTERED: {e}")
        sys.exit(1)