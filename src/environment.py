import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .adversary import Adversary

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
        self.adversary.update(x_query, y_target_pred)
        
        l_extract = self.adversary.compute_loss(x_query, y_target_pred)
        utility_loss = np.linalg.norm(e_true - e_output)
        reward = (self.lambda_param * l_extract) - (self.mu_param * utility_loss)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        info = {'l_extract': l_extract, 'utility_loss': utility_loss, 'action_a_t': a_t}
        
        return self._get_obs(), float(reward), done, truncated, info