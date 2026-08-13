import numpy as np

def apply_top_k_truncation(shap_values, k=3):
    shap_values = np.array(shap_values, dtype=np.float32)
    if k >= len(shap_values): return shap_values.copy()
    top_k_indices = np.argsort(np.abs(shap_values))[-k:]
    e_output = np.zeros_like(shap_values)
    e_output[top_k_indices] = shap_values[top_k_indices]
    return e_output

def apply_noise(shap_values, noise_level=0.1):
    """
    Adds Gaussian noise to the explanation values.
    The noise is scaled by the standard deviation of the original values.
    """
    shap_values = np.array(shap_values, dtype=np.float32)
    noise_std = np.std(shap_values) * noise_level
    noise = np.random.normal(loc=0.0, scale=noise_std, size=shap_values.shape)
    return shap_values + noise

def apply_precision_reduction(shap_values, decimals=2):
    """
    Generalizes explanations by rounding them to a specified number of decimals.
    """
    shap_values = np.array(shap_values, dtype=np.float32)
    return np.round(shap_values, decimals=decimals)

def apply_random_subset(shap_values, p=0.5):
    """
    Limits API output by randomly setting a fraction of feature importances to zero.
    'p' is the probability of keeping a feature's importance value.
    """
    shap_values = np.array(shap_values, dtype=np.float32)
    mask = np.random.choice([0, 1], size=shap_values.shape, p=[1-p, p])
    return shap_values * mask