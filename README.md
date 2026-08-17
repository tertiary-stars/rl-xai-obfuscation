# XAI Obfuscation using Reinforcement Learning

This project implements a Reinforcement Learning (RL) agent that learns to defend a machine learning model's explanations (XAI) from being used by an adversary for model extraction attacks. The agent learns a dynamic strategy to add noise to SHAP explanations, balancing the trade-off between confusing the adversary (security) and preserving the explanation's usefulness (utility).

## Project Structure

```
rl-agent-xai/
├── ppo_xai_defender_test.zip   # Saved trained RL agent model
├── train_agent.py              # Main script to train the RL agent
├── evaluate_baselines.py       # Script to evaluate the RL agent against other static strategies
├── main.py                     # (Deprecated) Old monolithic script
├── README.md                   # This file: project overview and documentation
└── src/
    ├── environment.py          # Contains the custom OpenAI Gym environment for the RL agent
    ├── adversary.py            # Defines the adversary's surrogate model and learning loop
    ├── utils.py                # Helper functions for data loading and model training
    └── baselines.py            # Implementations of static defense strategies
```

---

## How to Run

1.  **Train the Agent**:
    Run the training script. This will load the dataset, train the target model, and then train the PPO agent, saving the result as `ppo_xai_defender_test.zip`.
    ```bash
    python train_agent.py
    ```

2.  **Evaluate All Strategies**:
    Run the evaluation script. This will load the trained agent and compare its performance against several baseline defense strategies.
    ```bash
    python evaluate_baselines.py
    ```

---

## Key Concepts

The core of this project is the trade-off between two competing goals:

*   **Security (Extraction Loss)**: We want to make it difficult for an adversary to train a surrogate model that mimics our private target model. A high extraction loss means the adversary's model is inaccurate, which is good for security.
*   **Utility (Distortion Loss)**: We want the obfuscated explanations to remain as faithful as possible to the original ones. A low utility loss (or distortion) means the modified explanation is still useful and understandable to a legitimate user.

The RL agent's goal is to learn a policy that **maximizes security** while **minimizing distortion**.

---

## File-by-File Breakdown

### `train_agent.py`

This is the main entry point for training the RL agent.

*   **Purpose**: To set up and run the training loop for the PPO (Proximal Policy Optimization) agent.
*   **Why this way?**:
    *   **Seeding (`SEED = 42`)**: We set global seeds for `numpy`, `random`, and `torch` to ensure that every part of the process—from data splits and model initializations to the agent's own randomness—is reproducible. This is critical for fair experiments.
    *   **`make_vec_env`**: This utility from `stable-baselines3` wraps our custom environment. While we only use `n_envs=1`, it's a standard way to instantiate environments and correctly handles seeding.
    *   **PPO Agent**: PPO is a robust, state-of-the-art RL algorithm that works well with continuous action spaces (our agent chooses a noise level `a_t` between 0.0 and 1.0).
    *   **Saving the Model**: The trained agent's policy is saved to a file (`.zip`) so it can be loaded later for evaluation without needing to be retrained.

### `evaluate_baselines.py`

This script benchmarks our trained RL agent against other, simpler defense strategies.

*   **`evaluate_strategy(strategy_fn, ...)`**:
    *   **Purpose**: To provide a standardized "gauntlet" for any given defense strategy. It simulates an adversary's query process over a number of steps and measures the resulting average security and utility loss.
    *   **Why this way?**: It abstracts the evaluation logic, allowing us to easily plug in any new strategy (`strategy_fn`) without rewriting the simulation loop. It ensures every strategy is tested under the exact same conditions (same data, same adversary).

*   **Strategy Wrappers (`rl_agent_strategy`, `no_defense_strategy`, etc.)**:
    *   **Purpose**: To adapt each defense mechanism to the common interface required by `evaluate_strategy`.
    *   **Why this way?**: This "adapter" pattern makes the code clean. The `rl_agent_strategy` wrapper, for example, handles getting an observation and calling `agent.predict()`, while the `top_k_strategy` wrapper simply calls the corresponding function from `baselines.py`.

*   **Results (`pd.DataFrame`)**:
    *   **Purpose**: To collect and display the performance metrics for all strategies in a clear, tabular format.

### `src/environment.py`

This file defines the custom simulation environment where our agent learns.

*   **`class XAIObfuscationEnv(gym.Env)`**:
    *   **Purpose**: To model the XAI obfuscation problem as a standard reinforcement learning environment, compatible with libraries like `stable-baselines3`.
    *   **`__init__(...)`**:
        *   **Why this way?**: It initializes the state of the world: the dataset (`X_data`), the model to protect (`target_model`), and the explainer (`explainer`). The `lambda_param` and `mu_param` are crucial hyperparameters that allow us to weigh the importance of security vs. utility in the reward signal.
    *   **`reset(...)`**:
        *   **Why this way?**: This method is called at the start of every training episode. It resets the step counter and, importantly, creates a **new adversary**. This ensures the agent learns a general strategy, not one that just overfits to a single adversary's learning trajectory.
    *   **`_get_obs()`**:
        *   **Why this way?**: The observation tells the agent everything it needs to know to make a decision. We provide the current data point being queried (`current_query`) and a sense of time (`history_ratio`). The time component can help the agent learn a non-stationary policy (e.g., apply more noise later in the interaction as the adversary gets stronger).
    *   **`step(action)`**:
        *   **Purpose**: This is the core of the environment. It takes the agent's action, simulates one time-step, and returns the result.
        *   **Why this way?**:
            1.  **Get Action**: The agent provides an action `a_t` (noise level).
            2.  **Generate Noise**: `self.np_random.normal(...)` is used. This is critical. It uses the environment's internal, seedable random number generator, making the agent's interaction with the environment deterministic and reproducible during evaluation.
            3.  **Calculate Reward**: The reward `(lambda * l_extract) - (mu * utility_loss)` directly encodes the trade-off. The agent is rewarded for increasing the adversary's loss (`l_extract`) and penalized for distorting the explanation (`utility_loss`).
            4.  **Update Adversary**: The adversary sees the original query (but not the explanation) and the model's prediction, and updates its internal surrogate.
            5.  **Return**: It returns the next observation, the calculated reward, and status flags (`done`, `truncated`).

### `src/adversary.py`

This file defines the opponent our RL agent is trying to defeat.

*   **`class Adversary`**:
    *   **Purpose**: To simulate an attacker attempting a model extraction attack by training a surrogate model.
    *   **`__init__(...)`**:
        *   **Why this way?**: We use an `MLPClassifier` (a simple neural network) as the surrogate model. It's a reasonable choice for a general-purpose attacker. `random_state=42` ensures the adversary's own initialization is deterministic.
    *   **`update(X_batch, y_batch)`**:
        *   **Why this way?**: It uses `partial_fit`, which updates the surrogate model with a single data point at a time. This simulates an "online" learning scenario where the adversary queries the target model sequentially and learns from each result.
    *   **`compute_loss(X_batch, y_true)`**:
        *   **Why this way?**: It calculates the `log_loss` of the surrogate model on the latest query. This loss is a direct measure of how well the adversary is "extracting" the target model's decision boundary. A high loss means the adversary is failing, which is our security goal. We return `1.0` before the adversary is initialized to provide a consistent reward signal.

### `src/utils.py`

This file contains reusable utility functions.

*   **`load_and_train_target()`**:
    *   **Purpose**: To encapsulate the data loading and preprocessing logic.
    *   **Why this way?**: It isolates the data setup from the RL and evaluation logic, making the main scripts cleaner. It uses the "Adult" dataset, a standard benchmark. It trains an `XGBClassifier` as the "black-box" model we aim to protect and creates a `shap.TreeExplainer` to generate explanations for it.

### `src/baselines.py`

This file implements simple, non-learning defense strategies to compare against our RL agent.

*   **Purpose**: To provide a set of reference points for performance. If our complex RL agent can't beat these simple strategies, it's not providing much value.
*   **Why this way?**: Each function is a pure, stateless transformation of the SHAP values, representing a fixed defense policy.
    *   `apply_top_k_truncation`: A common method to simplify explanations by only showing the most important features.
    *   `apply_noise`: A naive noise-adding strategy with a fixed noise level.
    *   `apply_precision_reduction`: Reduces information by rounding values.
    *   `apply_random_subset`: Randomly hides some feature importances.
