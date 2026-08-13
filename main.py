import sys
from datetime import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import shap
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from stable_baselines3 import PPO

# ==========================================
# 1. Dataset & Target Model Setup
# ==========================================
def load_and_train_target():
    """Loads Adult Income dataset and trains an XGBoost black-box model."""
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    
    X = adult.data.select_dtypes(include=[np.number]).dropna() 
    y = (adult.target.loc[X.index] == '>50K').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    target_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss')
    target_model.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(target_model)
    
    return X_train.values, X_test.values, target_model, explainer

# ==========================================
# 2. The Adversary Loop (Surrogate Model)
# ==========================================
class Adversary:
    def __init__(self, input_dim):
        self.surrogate = MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            max_iter=1, 
            # Removed warm_start=True to prevent conflicts with partial_fit
            random_state=42 
        )
        self.is_initialized = False
        self.classes = np.array([0, 1])
        
    def update(self, X_batch, y_batch):
        # Always pass classes to partial_fit when processing single samples
        self.surrogate.partial_fit(X_batch, y_batch, classes=self.classes)
        self.is_initialized = True
            
    def compute_loss(self, X_batch, y_true):
        if not self.is_initialized:
            return 1.0  
            
        surrogate_probs = self.surrogate.predict_proba(X_batch)
        return log_loss(y_true, surrogate_probs, labels=self.classes)

# ==========================================
# 3. Custom Gymnasium Environment
# ==========================================
class XAIObfuscationEnv(gym.Env):
    def __init__(self, X_data, target_model, explainer, lambda_param=1.0, mu_param=1.0):
        super(XAIObfuscationEnv, self).__init__()
        
        self.X_data = X_data
        self.target_model = target_model
        self.explainer = explainer
        
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        
        self.n_features = X_data.shape[1]
        self.current_step = 0
        self.max_steps = min(1000, len(X_data)) 
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features + 1,), 
            dtype=np.float32
        )
        
        self.adversary = Adversary(input_dim=self.n_features)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.adversary = Adversary(input_dim=self.n_features)
        return self._get_obs(), {}

    def _get_obs(self):
        current_query = self.X_data[self.current_step]
        history_ratio = np.array([self.current_step / self.max_steps])
        return np.concatenate([current_query, history_ratio], axis=0).astype(np.float32)
        
    def step(self, action):
        a_t = action[0]
        
        x_query = self.X_data[self.current_step].reshape(1, -1)
        
        shap_values = self.explainer.shap_values(x_query)
        e_true = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        noise_std = np.std(self.X_data, axis=0) * 0.1 
        noise = np.random.normal(loc=0.0, scale=noise_std, size=e_true.shape)
        e_output = e_true + (a_t * noise)
        
        y_target_pred = self.target_model.predict(x_query)
        y_target_prob = self.target_model.predict_proba(x_query)
        
        self.adversary.update(x_query, y_target_pred)
        
        l_extract = self.adversary.compute_loss(x_query, y_target_pred)
        utility_loss = np.linalg.norm(e_true - e_output)
        
        reward = (self.lambda_param * l_extract) - (self.mu_param * utility_loss)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'l_extract': l_extract,
            'utility_loss': utility_loss,
            'action_a_t': a_t
        }
        
        return self._get_obs(), float(reward), done, truncated, info

# ==========================================
# 4. Execution Block
# ==========================================
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