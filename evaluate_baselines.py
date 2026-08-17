import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_and_train_target
from src.adversary import Adversary
import src.baselines as baselines

SEED = 42


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
    # Seed for reproducible evaluation, especially for noise-based strategies
    np.random.seed(SEED)
    random.seed(SEED)

    # Adversary's input is the query concatenated with the explanation
    adversary = Adversary(input_dim=X_data.shape[1] * 2)
    l_extract_history = []
    utility_loss_history = []

    print(f"Evaluating strategy: {strategy_fn.__name__}...")
    for i in tqdm(range(num_steps)):
        x_query = X_data[i].reshape(1, -1)
        y_target_pred = target_model.predict(x_query)

        # 1. Get true explanation and apply defense strategy to get obfuscated explanation
        shap_values = explainer.shap_values(x_query)
        # The explainer may return shape (1, n_features), but baselines expect (n_features,).
        e_true = (shap_values[0] if isinstance(shap_values, list) else shap_values).flatten()
        e_output = strategy_fn(e_true, x_query, i, num_steps)

        # 2. The adversary sees the query AND the obfuscated explanation.
        e_output_2d = e_output.reshape(1, -1)
        adversary_input = np.concatenate([x_query, e_output_2d], axis=1)

        # 3. Calculate extraction loss *before* the adversary trains on the new data.
        l_extract = adversary.compute_loss(adversary_input, y_target_pred)
        l_extract_history.append(l_extract)

        # 4. Now, allow the adversary to train on the query and its obfuscated explanation.
        adversary.update(adversary_input, y_target_pred)

        # 5. Calculate utility loss (distortion)
        utility_loss = np.linalg.norm(e_true - e_output)
        utility_loss_history.append(utility_loss)

    return np.mean(l_extract_history), np.mean(utility_loss_history)

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("1. Loading data, training target model, and loading RL agent...")
    X_train, X_test, target_model, explainer = load_and_train_target()
    agent = PPO.load("ppo_xai_defender_test")
    # We must use the standard deviation of the TRAINING data for noise generation
    # as this is what the agent was trained on.
    X_train_std = np.std(X_train, axis=0)
    print(" -> Done.")

    # --- Define Strategy Wrappers ---
    # Wrapper for the RL Agent
    def rl_agent_strategy(e_true, x_query, step, max_steps):
        history_ratio = np.array([step / max_steps])
        obs = np.concatenate([x_query.flatten(), history_ratio]).astype(np.float32) 
        action, _ = agent.predict(obs, deterministic=True)
        a_t = action[0]
        # Use the same noise application as the environment for a fair comparison
        # The agent's action `a_t` acts as the noise_level.
        noise = baselines.generate_noise(e_true, X_train_std, noise_level=a_t)
        return e_true + (a_t * noise)

    # Wrappers for static baselines
    def no_defense_strategy(e_true, x_query, step, max_steps): return e_true
    def top_k_strategy(e_true, x_query, step, max_steps): return baselines.apply_top_k_truncation(e_true, k=3)
    def noise_strategy(e_true, x_query, step, max_steps):
        # Use the consistent data_std for noise generation
        return baselines.apply_noise(e_true, data_std=X_train_std, noise_level=0.5)
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

    print("\n4. Generating plot...")

    fig, ax = plt.subplots(figsize=(11, 8))

    # Use a colormap to get different colors for each strategy
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))

    for i, row in results_df.iterrows():
        ax.scatter(
            row["Avg. Utility Loss (Distortion)"],
            row["Avg. Extraction Loss (Security)"],
            label=row["Strategy"],
            s=200,  # Marker size
            c=[colors[i]],
            alpha=0.8,
            edgecolors='k'
        )
        ax.text(
            row["Avg. Utility Loss (Distortion)"] + 0.005,  # Offset text slightly
            row["Avg. Extraction Loss (Security)"],
            row["Strategy"],
            fontsize=10
        )

    ax.set_title("Security vs. Distortion Trade-off of Defense Strategies", fontsize=16, pad=20)
    ax.set_xlabel("Distortion (Lower is Better →)", fontsize=12)
    ax.set_ylabel("Security (Higher is Better ↑)", fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plot_filename = "security_vs_distortion_tradeoff.png"
    plt.savefig(plot_filename)
    print(f" -> Plot saved to {plot_filename}")
    plt.show()