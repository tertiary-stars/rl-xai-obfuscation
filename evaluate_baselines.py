import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm
import pandas as pd

from src.utils import load_and_train_target
from src.adversary import Adversary
import src.baselines as baselines

def evaluate_strategy(strategy_fn, X_data, target_model, explainer, num_steps=500):
    """
    Evaluates a given defense strategy against an adversary.

    Args:
        strategy_fn: A function that takes (e_true, x_query, step, max_steps) and returns e_output.
        X_data: The dataset to use for evaluation.
        target_model: The black-box model being explained.
        explainer: The SHAP explainer for the target model.
        num_steps: The number of evaluation steps.

    Returns:
        A tuple of (average extraction loss, average utility loss).
    """
    adversary = Adversary(input_dim=X_data.shape[1])
    l_extract_history = []
    utility_loss_history = []

    print(f"Evaluating strategy: {strategy_fn.__name__}...")
    for i in tqdm(range(num_steps)):
        x_query = X_data[i].reshape(1, -1)

        # 1. Get true explanation
        shap_values = explainer.shap_values(x_query)
        e_true = shap_values[0] if isinstance(shap_values, list) else shap_values

        # 2. Apply defense strategy to get obfuscated explanation
        e_output = strategy_fn(e_true, x_query, i, num_steps)

        # 3. Calculate utility loss
        utility_loss = np.linalg.norm(e_true - e_output)
        utility_loss_history.append(utility_loss)

        # 4. Adversary observes the query and updates its surrogate model
        y_target_pred = target_model.predict(x_query)
        adversary.update(x_query, y_target_pred)

        # 5. Calculate extraction loss (security)
        l_extract = adversary.compute_loss(x_query, y_target_pred)
        l_extract_history.append(l_extract)

    return np.mean(l_extract_history), np.mean(utility_loss_history)

if __name__ == "__main__":
    print("1. Loading data, training target model, and loading RL agent...")
    _, X_test, target_model, explainer = load_and_train_target()
    agent = PPO.load("ppo_xai_defender_test")
    print(" -> Done.")

    # --- Define Strategy Wrappers ---
    # Wrapper for the RL Agent
    def rl_agent_strategy(e_true, x_query, step, max_steps):
        history_ratio = np.array([step / max_steps])
        obs = np.concatenate([x_query.flatten(), history_ratio]).astype(np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        a_t = action[0]
        # Use the same noise application as the environment
        noise_std = np.std(X_test, axis=0) * 0.1
        noise = np.random.normal(loc=0.0, scale=noise_std, size=e_true.shape)
        return e_true + (a_t * noise)

    # Wrappers for static baselines
    def no_defense_strategy(e_true, x_query, step, max_steps): return e_true
    def top_k_strategy(e_true, x_query, step, max_steps): return baselines.apply_top_k_truncation(e_true, k=3)
    def noise_strategy(e_true, x_query, step, max_steps): return baselines.apply_noise(e_true, noise_level=0.5)
    def precision_strategy(e_true, x_query, step, max_steps): return baselines.apply_precision_reduction(e_true, decimals=2)
    def subset_strategy(e_true, x_query, step, max_steps): return baselines.apply_random_subset(e_true, p=0.5)

    strategies = {
        "No Defense": no_defense_strategy,
        "Top-K (k=3)": top_k_strategy,
        "Gaussian Noise (lvl=0.5)": noise_strategy,
        "Precision Reduction (2 dec)": precision_strategy,
        "Random Subset (p=0.5)": subset_strategy,
        "RL Agent (PPO)": rl_agent_strategy,
    }

    results = []
    print("\n2. Starting evaluation of all defense strategies...")
    for name, strategy_fn in strategies.items():
        l_extract, utility_loss = evaluate_strategy(strategy_fn, X_test, target_model, explainer)
        results.append({
            "Strategy": name,
            "Avg. Extraction Loss (Security)": l_extract,
            "Avg. Utility Loss (Distortion)": utility_loss
        })

    print("\n3. Evaluation Complete. Results:\n")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # A good defense maximizes Security (Extraction Loss) while minimizing Distortion (Utility Loss)
    print("\nHigher 'Security' is better. Lower 'Distortion' is better.")