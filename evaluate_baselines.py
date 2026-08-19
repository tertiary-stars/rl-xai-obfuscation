import numpy as np
from stable_baselines3 import PPO
from tqdm import tqdm
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

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
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    print("1. Loading data and training target model...")
    X_train, X_test, target_model, explainer = load_and_train_target()
    print(" -> Done.")

    _sample = X_train[:min(200, len(X_train))]
    _sample_shap = explainer.shap_values(_sample)
    _sample_shap = _sample_shap[0] if isinstance(_sample_shap, list) else _sample_shap
    e_std = np.std(_sample_shap, axis=0)
    e_std[e_std == 0] = 1e-6

    # --- Static Strategies ---
    def no_defense_strategy(e_true, x_query, step, max_steps): return e_true
    def top_k_strategy(e_true, x_query, step, max_steps): return apply_top_k_truncation(e_true, k=3)
    def noise_strategy(e_true, x_query, step, max_steps): return apply_noise(e_true, data_std=e_std, noise_level=0.5)
    def precision_strategy(e_true, x_query, step, max_steps): return apply_precision_reduction(e_true, decimals=2)
    def subset_strategy(e_true, x_query, step, max_steps): return apply_random_subset(e_true, p=0.5)

    static_strategies = {
        "No Defense": no_defense_strategy,
        "Top-K (k=3)": top_k_strategy,
        "Gaussian Noise (lvl=0.5)": noise_strategy,
        "Precision Reduction (2 dec)": precision_strategy,
        "Random Subset (p=0.5)": subset_strategy,
    }

    results = []
    print("\n2. Starting evaluation of static defense strategies...")
    for name, strategy_fn in static_strategies.items():
        l_extract, utility_loss = evaluate_strategy(strategy_fn, X_test, target_model, explainer)
        results.append({
            "Strategy": name,
            "Avg. Extraction Loss (Security)": l_extract,
            "Avg. Utility Loss (Distortion)": utility_loss,
            "Type": "Static Baseline"
        })

    # --- RL Ablation Agents ---
    ablation_pairs = [
        (0.0, 1.0), (1.0, 0.5), (1.0, 0.2), (1.0, 0.1), 
        (1.0, 0.05), (1.0, 0.01), (1.0, 0.0)
    ]

    print("\n3. Starting evaluation of RL ablation agents...")
    for lam, mu in ablation_pairs:
        agent_name = f"ppo_xai_defender_l{lam}_m{mu}"
        try:
            current_agent = PPO.load(agent_name)
        except Exception as e:
            print(f"Could not load {agent_name}. Skipping. Error: {e}")
            continue
            
        def make_rl_strategy(loaded_agent):
            def rl_agent_strategy(e_true, x_query, step, max_steps):
                history_ratio = np.array([step / max_steps])
                obs = np.concatenate([x_query.flatten(), history_ratio]).astype(np.float32)
                action, _ = loaded_agent.predict(obs, deterministic=True)
                a_t = action[0]
                noise = generate_noise(e_true, e_std, noise_level=1.0)
                return (1 - a_t) * e_true + a_t * noise
            return rl_agent_strategy
            
        strat_fn = make_rl_strategy(current_agent)
        strat_fn.__name__ = f"rl_l{lam}_m{mu}"
        name = f"RL (λ={lam}, μ={mu})"
        l_extract, utility_loss = evaluate_strategy(strat_fn, X_test, target_model, explainer)
        
        results.append({
            "Strategy": name,
            "Avg. Extraction Loss (Security)": l_extract,
            "Avg. Utility Loss (Distortion)": utility_loss,
            "Type": "RL Agent"
        })

    print("\n4. Evaluation Complete. Results:\n")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\n5. Generating publication-ready Pareto front plot...")
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap

    # Set seaborn style for a cleaner academic look
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 7))

    static_df = results_df[results_df["Type"] == "Static Baseline"].copy()
    rl_df = results_df[results_df["Type"] == "RL Agent"].copy()
    
    # Sort RL data by utility loss to draw the line correctly
    rl_df = rl_df.sort_values(by="Avg. Utility Loss (Distortion)")

    # 1. Plot Static Baselines (using subdued blue/gray squares)
    for i, row in static_df.iterrows():
        # Avoid double-labeling the origin if "No Defense" overlaps with collapsed RL
        label_text = row["Strategy"] if row["Strategy"] != "No Defense" else ""
        
        ax.scatter(row["Avg. Utility Loss (Distortion)"], row["Avg. Extraction Loss (Security)"], 
                   s=150, c='#4A708B', alpha=0.8, edgecolors='white', linewidth=1.5, marker='s', zorder=2)
        
        if label_text:
            ax.text(row["Avg. Utility Loss (Distortion)"] + 0.02, row["Avg. Extraction Loss (Security)"], 
                    label_text, fontsize=10, color='#2F4F4F', verticalalignment='center', zorder=4)

    # 2. Plot RL Pareto Curve
    ax.plot(rl_df["Avg. Utility Loss (Distortion)"], rl_df["Avg. Extraction Loss (Security)"], 
            linestyle='--', color='#A9A9A9', linewidth=2, alpha=0.7, zorder=1)

    # 3. Plot RL Agents with a color gradient (dark red -> light coral) to show parameter progression
    cmap = sns.color_palette("flare", as_cmap=True)
    
    # Track collapsed points to label them once
    collapsed_x, collapsed_y = 0.0, 0.0
    has_collapsed = False

    for i, row in rl_df.iterrows():
        x_val = row["Avg. Utility Loss (Distortion)"]
        y_val = row["Avg. Extraction Loss (Security)"]
        
        # Color based on index in the sorted dataframe
        color = cmap(i / len(rl_df))
        
        ax.scatter(x_val, y_val, s=200, color=color, alpha=0.9, edgecolors='white', linewidth=1.5, marker='o', zorder=3)
        
        # Handle labels
        if x_val < 0.001 and y_val < 0.25:
            has_collapsed = True
            collapsed_x, collapsed_y = x_val, y_val
        else:
            # Shift labels slightly to avoid clipping the line
            y_offset = 0.002 if i % 2 == 0 else -0.003
            ax.text(x_val + 0.02, y_val + y_offset, row["Strategy"], 
                    fontsize=10, color='#8B0000', fontweight='bold', verticalalignment='center', zorder=4)

    # Add the single combined label for the overlapping origin points
    if has_collapsed:
        ax.text(collapsed_x + 0.02, collapsed_y, "No Defense / RL (μ ≥ 0.1)", 
                fontsize=10, color='black', fontweight='bold', verticalalignment='center', zorder=4)

    # 4. Clean up axes and legends
    ax.set_title("Security vs. Distortion Trade-off: RL Adaptive Policy vs. Static Defenses", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Average Utility Loss (Distortion) $\\rightarrow$ Lower is Better", fontsize=11, fontweight='bold')
    ax.set_ylabel("Average Extraction Loss (Security) $\\uparrow$ Higher is Better", fontsize=11, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Static Baselines', markerfacecolor='#4A708B', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='RL Agents (Pareto Front)', markerfacecolor='#D95F5F', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, shadow=True, fontsize=11)

    # Despine for a cleaner look
    sns.despine(trim=True, offset=5)
    
    plt.tight_layout()
    plot_filename = "security_vs_distortion_academic.png"
    plt.savefig(plot_filename, dpi=300) # High DPI for paper inclusion
    print(f" -> Academic plot saved to {plot_filename}")