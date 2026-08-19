# Troubleshooting and Design Log

This document tracks the key issues encountered during the development of the RL agent and the solutions implemented to resolve them.

---

### 1. Issue: RL Agent Converged to a Trivial Local Optimum (Zero Distortion)

**Symptom:**
The PPO agent quickly learned to always output an action `a_t` of 0. This corresponds to a "do nothing" policy, where it never adds noise to the explanations. While this resulted in zero utility loss (distortion), it also provided zero security, failing the primary objective.

**Root Causes & Solutions:**

**A. Unbalanced Reward Scaling:**
*   **Problem:** The `utility_loss` was calculated as the raw L2 norm (`np.linalg.norm(e_true - e_output)`), which produced values on a much larger scale (e.g., 1.0-10.0) than the `l_extract` (log-loss, typically 0.5-1.5). The penalty for distortion was so high that it completely overpowered any potential reward from increasing security.
*   **Solution:** We normalized the `utility_loss` to bring it to a similar mathematical scale as the extraction loss. The new calculation is `np.linalg.norm(e_true - e_output) / (np.linalg.norm(e_true) + 1e-8)`. Additionally, we lowered the default `mu_param` (the weight for the utility penalty) from `1.0` to `0.05` to further rebalance the reward signal.

**B. Noisy and Unstable Extraction Loss Signal:**
*   **Problem:** The `l_extract` was calculated based on the adversary's performance on a single data point. The `log_loss` for one sample is highly volatile and provides a very noisy gradient for the PPO agent, making it difficult to learn a stable policy.
*   **Solution:** We implemented a rolling history buffer (`deque` of size 32) in the `XAIObfuscationEnv`. The `l_extract` is now calculated as the average loss over this entire buffer. This smooths out the reward signal, providing a much more stable gradient for the agent to learn from.

---

### 2. Issue: Gaussian Noise Baseline Produced Extreme Distortion

**Symptom:**
When evaluating the "Gaussian Noise" baseline strategy, the utility loss was orders of magnitude higher than any other strategy, making it an invalid comparison point.

**Root Cause:**
*   **Problem:** The `generate_noise` function was incorrectly scaling the noise. It was using the standard deviation of the raw dataset features (e.g., `age`, `capital-gain`), which have a very large scale, and applying that noise to the SHAP values, which are on a much smaller scale (e.g., -0.5 to 0.5). This resulted in adding gigantic, disproportionate noise.
*   **Solution:** We refactored the `generate_noise` function to remove the confusing `scale_fraction` parameter. We then ensured that all calls to this function from the environment and evaluation scripts pass `e_std` (the standard deviation of the *SHAP values*), not the standard deviation of the dataset features. This ensures the noise is scaled appropriately to the explanations themselves.

---

### 3. Issue (Design Choice): How Should the Adversary Receive Input?

**Symptom:**
A concern was raised that if the adversary receives both the feature vector (`x_query`) and the obfuscated explanation (`e_output`), it might learn to simply ignore the noisy `e_output` and rely only on `x_query`, neutralizing the agent's defense.

**Root Cause:**
*   **Problem:** This is a fundamental design choice in simulating an explanation-aware attack. Does the adversary use the explanation to augment its knowledge, or does it rely on it entirely?
*   **Solution (Decision):** We decided to **preserve the original input structure** (`np.concatenate([x_query, e_output], axis=1)`). The goal is to model an adversary that *is* explanation-aware and uses them to improve its attack. Forcing the adversary to rely *only* on the explanation would change the problem definition. The fixes for reward scaling and loss stability (see Issue #1) were deemed sufficient to force the agent to learn a meaningful policy without altering this core interaction model.