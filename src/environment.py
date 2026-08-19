import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from src.adversary import Adversary

def generate_noise(shap_values, data_std, noise_level=1.0, seed_generator=np.random):
    """Generates Gaussian noise scaled by data_std."""
    noise = seed_generator.normal(loc=0.0, scale=data_std, size=shap_values.shape)
    return noise * noise_level

class XAIObfuscationEnv(gym.Env):
    def __init__(self, X_data, target_model, explainer, lambda_param=1.0, mu_param=0.05, history_len=32):
        super(XAIObfuscationEnv, self).__init__()

        self.X_data = X_data
        self.y_data = target_model.predict(X_data)
        self.target_model = target_model
        self.explainer = explainer

        self.lambda_param = lambda_param
        self.mu_param = mu_param

        self.data_std = np.std(self.X_data, axis=0)
        self.n_features = X_data.shape[1]
        self.current_step = 0
        self.history_len = history_len
        self.adversary_history = deque(maxlen=self.history_len)

        self.max_steps = min(1000, len(X_data))

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_features + 1,),
            dtype=np.float32
        )

        # Typical SHAP magnitude - scale for "full obfuscation" and for normalizing the adversary's input.
        sample = X_data[:min(200, len(X_data))]
        sample_shap = self.explainer.shap_values(sample)
        sample_shap = sample_shap[0] if isinstance(sample_shap, list) else sample_shap
        self.e_std = np.std(sample_shap, axis=0)
        self.e_std[self.e_std == 0] = 1e-6

        adversary_std = np.concatenate([self.data_std, self.e_std])
        adversary_std[adversary_std == 0] = 1.0
        self.adversary_mean = np.concatenate([np.mean(self.X_data, axis=0), np.mean(sample_shap, axis=0)])
        self.adversary_std = adversary_std

        # The adversary's input is the query concatenated with the explanation
        self.adversary = Adversary(input_dim=self.n_features * 2, feature_mean=self.adversary_mean, feature_std=self.adversary_std)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Re-initialize adversary and its history buffer for a new episode
        self.adversary = Adversary(input_dim=self.n_features * 2, feature_mean=self.adversary_mean, feature_std=self.adversary_std)
        self.adversary_history.clear()
        return self._get_obs(), {}

    def _get_obs(self):
        current_query = self.X_data[self.current_step]
        history_ratio = np.array([self.current_step / self.max_steps])
        return np.concatenate([current_query, history_ratio], axis=0).astype(np.float32)

    def step(self, action):
        a_t = action[0]

        x_query = self.X_data[self.current_step].reshape(1, -1)
        y_target_pred = self.y_data[self.current_step].reshape(1,)
        
        # 1. Get the true explanation and apply the agent's action (obfuscation).
        shap_values = self.explainer.shap_values(x_query)
        e_true = shap_values[0] if isinstance(shap_values, list) else shap_values
        # a_t=0 -> exact explanation, a_t=1 -> full obfuscation (real-scale noise, not a tiny perturbation).
        noise = generate_noise(e_true, self.e_std, noise_level=1.0, seed_generator=self.np_random)
        e_output = (1 - a_t) * e_true + a_t * noise

        # 2. The adversary sees the query AND the obfuscated explanation.
        adversary_input = np.concatenate([x_query, e_output], axis=1)

        # 3. Add current sample to history and calculate extraction loss over the buffer.
        self.adversary_history.append((adversary_input, y_target_pred))
        history_inputs = np.vstack([item[0] for item in self.adversary_history])
        history_labels = np.concatenate([item[1] for item in self.adversary_history])
        l_extract = self.adversary.compute_loss(history_inputs, history_labels)

        # 4. Now, allow the adversary to train on the query and its obfuscated explanation.
        self.adversary.update(adversary_input, y_target_pred)

        # 5. Calculate the utility loss and the final reward.
        # Normalize utility loss to be on a similar scale to l_extract (log_loss).
        # This prevents the penalty term from dominating the reward signal.
        e_true_norm = np.linalg.norm(e_true) + 1e-8
        utility_loss = np.linalg.norm(e_true - e_output) / e_true_norm

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