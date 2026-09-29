"""The game: history-encoded state, obfuscation, adversary session, metrics and the Gymnasium environment.

Everything here is shared by training (XAIObfuscationEnv) and evaluation (evaluate_baselines.py),
so both measure exactly the same thing.
"""
import warnings
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import spearmanr

from src.adversary import Adversary

# ---------------------------------------------------------------------------------------------
# State s_t: fixed-size encoding of the session's query history
#   window  s_t = [z_{t-k+1}, ..., z_{t-1}, z_t, t/T_max]   (dim k*d + 1)
#           the last k standardised queries; zero-padded at session start (zero = feature mean),
#           t/T_max tells the agent how many slots are real.
#   none    s_t = [z_t, t/T_max]                            (dim d + 1)
#           no history - ablation, equivalent to the original formulation.
# z = clip((x - mean_train) / std_train, -10, 10)
# ---------------------------------------------------------------------------------------------
HISTORY_MODES = ("window", "none")
CLIP = 10.0


class HistoryEncoder:
    def __init__(self, x_mean, x_std, t_max, mode="window", k=8):
        if mode not in HISTORY_MODES:
            raise ValueError(f"Unknown history mode '{mode}'. Choose from {HISTORY_MODES}.")
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_std = np.where(np.asarray(x_std) == 0, 1.0, x_std).astype(np.float64)
        self.d = len(self.x_mean)
        self.t_max = t_max
        self.k = k if mode == "window" else 1
        self.dim = self.k * self.d + 1
        self.window = deque(maxlen=self.k)

    def reset(self):
        self.window.clear()

    def observe(self, x, t):
        """Add query x (the t-th query of the session, 0-indexed) to the history and return s_t."""
        self.window.append(np.clip((np.ravel(x) - self.x_mean) / self.x_std, -CLIP, CLIP))
        return self._encode(t)

    def terminal(self, t):
        """Observation returned at episode end (no new query arrives)."""
        return self._encode(t)

    def _encode(self, t):
        pad = [np.zeros(self.d)] * (self.k - len(self.window))
        return np.concatenate(pad + list(self.window) + [np.array([t / self.t_max])]).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Obfuscation and metrics
# ---------------------------------------------------------------------------------------------
def obfuscate(e_true, a_t, e_std, rng):
    """E_out = (1 - a_t) * E_true + a_t * eps,  eps ~ N(0, diag(e_std^2)).

    a_t = 0 -> exact explanation, a_t = 1 -> pure noise at the typical explanation scale.
    """
    noise = rng.normal(loc=0.0, scale=e_std, size=np.shape(e_true))
    return (1 - a_t) * e_true + a_t * noise


def explanation_distortion(e_true, e_out):
    """Relative L2 distortion ||E_true - E_out|| / ||E_true||  (lower = more faithful explanation)."""
    return float(np.linalg.norm(e_true - e_out) / (np.linalg.norm(e_true) + 1e-8))


def spearman_rho(e_true, e_out):
    with warnings.catch_warnings():  # constant vectors (e.g. all-zero explanation) -> rho undefined, counted as 0
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(e_true, e_out)
    return 0.0 if np.isnan(rho) else float(rho)


def top_k_agreement(e_true, e_out, k=3):
    k = min(k, len(e_true))
    top_true = set(np.argsort(np.abs(e_true))[-k:])
    top_out = set(np.argsort(np.abs(e_out))[-k:])
    return len(top_true & top_out) / k


class AttackSession:
    """One adversary session: the surrogate learns online from every (x, E_out) it receives.

    Before learning from query t, the adversary error is measured on the last `adv_window`
    queries (including t) - a smoothed estimate of how well the surrogate currently imitates M.
    """

    def __init__(self, adv_mean, adv_std, adv_window=32):
        self.adversary = Adversary(input_dim=len(adv_mean), feature_mean=adv_mean, feature_std=adv_std)
        self.buffer = deque(maxlen=adv_window)

    def step(self, x, y, e_true, e_out):
        adversary_input = np.concatenate([np.ravel(x), np.ravel(e_out)]).reshape(1, -1)
        y = np.asarray(y).reshape(1,)

        self.buffer.append((adversary_input, y))
        window_inputs = np.vstack([item[0] for item in self.buffer])
        window_labels = np.concatenate([item[1] for item in self.buffer])
        adversary_error = self.adversary.error(window_inputs, window_labels)

        self.adversary.update(adversary_input, y)

        return adversary_error, explanation_distortion(e_true, e_out)


# ---------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------
class XAIObfuscationEnv(gym.Env):
    """One episode = one simulated user session of `max_steps` queries.

    Each step: the agent sees s_t, picks an obfuscation level a_t in [0, 1], the adversary learns
    from the obfuscated explanation, and the agent is rewarded with

        R_t = lambda * AdversaryError_t - mu * Distortion_t

    Explanations are precomputed (see src/utils.py: prepare) so that the environment is cheap and
    independent of SHAP/LIME - this is what makes many parallel training runs affordable.
    """

    def __init__(self, data, lambda_param=1.0, mu_param=0.05, history="window", state_window=8,
                 adv_window=32, max_steps=1000):
        super().__init__()

        self.X = data["X_pool"]
        self.y = data["y_pool"]
        self.E = data["E_pool"]
        self.e_std = data["e_std"]
        self.adv_mean = data["adv_mean"]
        self.adv_std = data["adv_std"]

        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.adv_window = adv_window
        self.max_steps = min(max_steps, len(self.X))

        self.encoder = HistoryEncoder(data["x_mean"], data["x_std"], t_max=self.max_steps, mode=history, k=state_window)

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.encoder.dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # A new session: fresh adversary, empty history and a random sequence of queries,
        # so that the agent cannot memorise one fixed query order.
        self.session_idx = self.np_random.choice(len(self.X), size=self.max_steps, replace=False)
        self.session = AttackSession(self.adv_mean, self.adv_std, self.adv_window)
        self.encoder.reset()
        return self.encoder.observe(self.X[self.session_idx[0]], 0), {}

    def step(self, action):
        a_t = float(np.clip(action[0], 0.0, 1.0))
        i = self.session_idx[self.current_step]

        e_true = self.E[i]
        e_out = obfuscate(e_true, a_t, self.e_std, self.np_random)
        adversary_error, distortion = self.session.step(self.X[i], self.y[i], e_true, e_out)

        reward = self.lambda_param * adversary_error - self.mu_param * distortion

        self.current_step += 1
        done = self.current_step >= self.max_steps
        if done:
            obs = self.encoder.terminal(self.current_step)
        else:
            obs = self.encoder.observe(self.X[self.session_idx[self.current_step]], self.current_step)

        info = {
            "adversary_error": adversary_error,
            "distortion": distortion,
            "action_a_t": a_t,
        }
        return obs, float(reward), done, False, info
