import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import load_and_train_target
from src.adversary import Adversary

from collections import deque


def generate_noise(shap_values, data_std, noise_level=1.0, seed_generator=np.random):
    noise = seed_generator.normal(loc=0.0, scale=data_std, size=shap_values.shape)
    return noise * noise_level

def apply_top_k_truncation(shap_values, k=3):
    shap_values = np.array(shap_values, dtype=np.float32)
    if k >= len(shap_values): return shap_values.copy()
    top_k_indices = np.argsort(np.abs(shap_values))[-k:]
    e_output = np.zeros_like(shap_values)
    e_output[top_k_indices] = shap_values[top_k_indices]
    return e_output

def apply_noise(shap_values, data_std, noise_level=0.1):
    noise = generate_noise(shap_values, data_std, noise_level=noise_level)
    return shap_values + noise

def apply_precision_reduction(shap_values, decimals=2):
    shap_values = np.array(shap_values, dtype=np.float32)
    return np.round(shap_values, decimals=decimals)

def apply_random_subset(shap_values, p=0.5):
    shap_values = np.array(shap_values, dtype=np.float32)
    mask = np.random.choice([0, 1], size=shap_values.shape, p=[1-p, p])
    return shap_values * mask

SEED = 42


def evaluate_strategy(strategy_fn, X_data, target_model, explainer, num_steps=500, history_len=32):
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

    # Typical SHAP magnitude, used to normalize the adversary's input.
    sample = X_data[:min(200, len(X_data))]
    sample_shap = explainer.shap_values(sample)
    sample_shap = sample_shap[0] if isinstance(sample_shap, list) else sample_shap
    e_std = np.std(sample_shap, axis=0)
    e_std[e_std == 0] = 1e-6
    feature_std = np.concatenate([np.std(X_data, axis=0), e_std])
    feature_std[feature_std == 0] = 1.0
    feature_mean = np.concatenate([np.mean(X_data, axis=0), np.mean(sample_shap, axis=0)])

    # Adversary's input is the query concatenated with the explanation
    adversary = Adversary(input_dim=X_data.shape[1] * 2, feature_mean=feature_mean, feature_std=feature_std)
    adversary_history = deque(maxlen=history_len) # Initialize the history buffer
    
    l_extract_history = []
    utility_loss_history = []

    print(f"Evaluating strategy: {strategy_fn.__name__}...")
    for i in tqdm(range(num_steps)):
        x_query = X_data[i].reshape(1, -1)
        y_target_pred = target_model.predict(x_query)

        shap_values = explainer.shap_values(x_query)
        e_true = (shap_values[0] if isinstance(shap_values, list) else shap_values).flatten()
        e_output = strategy_fn(e_true, x_query, i, num_steps)

        e_output_2d = e_output.reshape(1, -1)
        adversary_input = np.concatenate([x_query, e_output_2d], axis=1)

        # 3. Add to history buffer and calculate extraction loss over the batch
        adversary_history.append((adversary_input, y_target_pred))
        history_inputs = np.vstack([item[0] for item in adversary_history])
        history_labels = np.concatenate([item[1] for item in adversary_history])
        
        l_extract = adversary.compute_loss(history_inputs, history_labels)
        l_extract_history.append(l_extract)

        # 4. Now, allow the adversary to train on the new data
        adversary.update(adversary_input, y_target_pred)

        # 5. Calculate utility loss
        e_true_norm = np.linalg.norm(e_true) + 1e-8
        utility_loss = np.linalg.norm(e_true - e_output) / e_true_norm
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
    print(" -> Done.")

    # Must match XAIObfuscationEnv's e_std definition for a fair comparison.
    _sample = X_train[:min(200, len(X_train))]
    _sample_shap = explainer.shap_values(_sample)
    _sample_shap = _sample_shap[0] if isinstance(_sample_shap, list) else _sample_shap
    e_std = np.std(_sample_shap, axis=0)
    e_std[e_std == 0] = 1e-6

    # --- Define Strategy Wrappers ---
    # Wrapper for the RL Agent
    def rl_agent_strategy(e_true, x_query, step, max_steps):
        history_ratio = np.array([step / max_steps])
        obs = np.concatenate([x_query.flatten(), history_ratio]).astype(np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        a_t = action[0]
        # Match the environment's obfuscation formula (Eq. 2).
        noise = generate_noise(e_true, e_std, noise_level=1.0)
        return (1 - a_t) * e_true + a_t * noise

    # Wrappers for static baselines
    def no_defense_strategy(e_true, x_query, step, max_steps): return e_true
    def top_k_strategy(e_true, x_query, step, max_steps): return apply_top_k_truncation(e_true, k=3)
    def noise_strategy(e_true, x_query, step, max_steps):
        # Scale to e_std (SHAP magnitude), not raw feature std - different units.
        return apply_noise(e_true, data_std=e_std, noise_level=0.5)
    def precision_strategy(e_true, x_query, step, max_steps): return apply_precision_reduction(e_true, decimals=2)
    def subset_strategy(e_true, x_query, step, max_steps): return apply_random_subset(e_true, p=0.5)

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

    print("\n5. Recording a_t over a single session (dynamic behavior check)...")
    # Averages can't show adaptive behavior - trace a_t and l_extract per step instead.
    np.random.seed(SEED)
    trace_steps = 200
    # Same normalization as Adversary, needed for the dynamics trace below.
    feature_std = np.concatenate([np.std(X_test, axis=0), e_std])
    feature_std[feature_std == 0] = 1.0
    feature_mean = np.concatenate([np.mean(X_test, axis=0), np.mean(_sample_shap, axis=0)])
    # ... [Keep previous trace setup code] ...
    trace_adversary = Adversary(input_dim=X_test.shape[1] * 2, feature_mean=feature_mean, feature_std=feature_std)
    trace_history = deque(maxlen=32) # Add history buffer for the trace
    
    a_t_trace, l_extract_trace = [], []
    for i in range(trace_steps):
        x_query = X_test[i].reshape(1, -1)
        y_target_pred = target_model.predict(x_query)
        shap_values = explainer.shap_values(x_query)
        e_true = (shap_values[0] if isinstance(shap_values, list) else shap_values).flatten()

        history_ratio = np.array([i / trace_steps])
        obs = np.concatenate([x_query.flatten(), history_ratio]).astype(np.float32)
        action, _ = agent.predict(obs, deterministic=True)
        a_t = action[0]
        a_t_trace.append(a_t)

        noise = generate_noise(e_true, e_std, noise_level=1.0)
        e_output = (1 - a_t) * e_true + a_t * noise
        adversary_input = np.concatenate([x_query, e_output.reshape(1, -1)], axis=1)
        
        # Buffer the trace extraction loss calculation
        trace_history.append((adversary_input, y_target_pred))
        history_inputs = np.vstack([item[0] for item in trace_history])
        history_labels = np.concatenate([item[1] for item in trace_history])
        
        l_extract_trace.append(trace_adversary.compute_loss(history_inputs, history_labels))
        trace_adversary.update(adversary_input, y_target_pred)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(a_t_trace, color="tab:orange")
    ax1.set_ylabel("Obfuscation intensity (a_t)")
    ax1.set_title("RL Agent's Obfuscation Intensity and Adversary's Extraction Loss Over a Session", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(l_extract_trace, color="tab:blue")
    ax2.set_ylabel("Extraction loss (l_extract)")
    ax2.set_xlabel("Step in session")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    trace_filename = "rl_agent_dynamics_over_session.png"
    plt.savefig(trace_filename)
    print(f" -> Plot saved to {trace_filename}")
    plt.show()