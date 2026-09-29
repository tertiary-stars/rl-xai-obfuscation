import os

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import lime.lime_tabular
import xgboost as xgb
import shap
from tqdm import tqdm

class LimeWrapper:
    """Wraps a LIME explainer to perfectly mimic the SHAP TreeExplainer API."""
    def __init__(self, lime_explainer, predict_proba_fn, num_features, num_samples=500):
        self.explainer = lime_explainer
        self.predict_proba_fn = predict_proba_fn
        self.num_features = num_features
        self.num_samples = num_samples  # Drastically speeds up execution

    def shap_values(self, X):
        X_array = np.array(X)
        is_single_instance = X_array.ndim == 1
        
        if is_single_instance:
            X_array = X_array.reshape(1, -1)
            
        batch_size, n_features = X_array.shape
        explanations = np.zeros((batch_size, n_features))
        
        for i in range(batch_size):
            exp = self.explainer.explain_instance(
                X_array[i], 
                self.predict_proba_fn, 
                num_features=self.num_features,
                num_samples=self.num_samples  # Applied here
            )
            
            if 1 in exp.local_exp:
                for feature_idx, weight in exp.local_exp[1]:
                    explanations[i, feature_idx] = weight
                    
        return explanations[0] if is_single_instance else explanations

def load_and_train_credit_dnn():
    """Second Pipeline: Credit Card + DNN + LIME"""
    credit = fetch_openml(data_id=1597, as_frame=True, parser="auto")
    X = credit.data.select_dtypes(include=[np.number]).dropna().values
    y = (credit.target == '1').astype(int).values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    target_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
    target_model.fit(X_train, y_train)

    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train, 
        mode='classification', 
        random_state=42
    )
    
    # This is the crucial wrapper step!
    explainer = LimeWrapper(
        lime_explainer, 
        target_model.predict_proba, 
        num_features=X_train.shape[1]
    )

    return X_train, X_test, target_model, explainer
    
def load_and_train_adult_xgb():
    """Original Pipeline: Adult Income + XGBoost + SHAP"""
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    X = adult.data.select_dtypes(include=[np.number]).dropna()
    y = (adult.target.loc[X.index] == '>50K').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    target_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss', random_state=42)
    target_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(target_model)

    return X_train.values, X_test.values, target_model, explainer


# ---------------------------------------------------------------------------------------------
# Synthetic-query attacker: owns only a few seed samples and synthesises new queries by perturbing
# earlier ones (random-sign steps on a random subset of features, in the spirit of the JbDA / T-RND
# augmentations that PRADA was designed to detect). The natural-query attacker simply queries
# natural data in random order, as in training.
# ---------------------------------------------------------------------------------------------
ATTACKS = ("natural", "synthetic")


def synthetic_queries(X_seed_pool, n_queries, x_std, rng, n_seed=10, step=0.1, feature_frac=0.3):
    d = X_seed_pool.shape[1]
    x_std = np.where(x_std == 0, 1.0, x_std)
    queries = list(X_seed_pool[rng.choice(len(X_seed_pool), size=n_seed, replace=False)])
    while len(queries) < n_queries:
        base = queries[rng.integers(len(queries))]
        mask = rng.random(d) < feature_frac
        if not mask.any():
            mask[rng.integers(d)] = True
        direction = rng.choice([-1.0, 1.0], size=d) * mask
        queries.append(base + step * x_std * direction)
    return np.asarray(queries)


# ---------------------------------------------------------------------------------------------
# One-off preparation: train the target model, precompute explanations and cache everything.
# Explanations (SHAP / LIME) dominate the cost of a step and do not depend on the defence, so they
# are computed once. Every training / evaluation job then only loads this cache, which keeps
# parallel jobs cheap and guarantees they all see identical data.
# ---------------------------------------------------------------------------------------------
DATASETS = {
    "adult": ("Adult Income (XGBoost + SHAP)", load_and_train_adult_xgb),
    "credit": ("Credit Card (DNN + LIME)", load_and_train_credit_dnn),
}
CACHE_DIR = "cache"


def cache_path(dataset):
    return os.path.join(CACHE_DIR, f"{dataset}.joblib")


def explain(explainer, X, batch=50, desc="explaining"):
    out = []
    for start in tqdm(range(0, len(X), batch), desc=desc, leave=False):
        values = explainer.shap_values(X[start:start + batch])
        values = values[0] if isinstance(values, list) else values
        out.append(np.asarray(values, dtype=np.float64).reshape(len(X[start:start + batch]), -1))
    return np.vstack(out)


def prepare(dataset, n_pool=5000, n_eval=1000, n_synthetic_streams=3, synthetic_len=500, seed=42, force=False):
    path = cache_path(dataset)
    if os.path.exists(path) and not force:
        print(f"[prepare] {path} already exists (use --force to rebuild).")
        return path

    rng = np.random.default_rng(seed)
    pipeline_name, load_fn = DATASETS[dataset]
    print(f"[prepare] {pipeline_name}: loading data and training target model...")
    X_train, X_test, target_model, explainer = load_fn()
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)

    # Training pool (sessions are sampled from it) and held-out evaluation queries.
    X_pool = X_train[rng.choice(len(X_train), size=min(n_pool, len(X_train)), replace=False)]
    X_eval = X_test[rng.choice(len(X_test), size=min(n_eval, len(X_test)), replace=False)]
    E_pool = explain(explainer, X_pool, desc="explaining pool")
    E_eval = explain(explainer, X_eval, desc="explaining eval")

    # Synthetic-query attacker streams (seeded from held-out data).
    X_synth, E_synth = [], []
    for s in range(n_synthetic_streams):
        Xs = synthetic_queries(X_eval, synthetic_len, x_std, np.random.default_rng(seed + 1000 + s))
        X_synth.append(Xs)
        E_synth.append(explain(explainer, Xs, desc=f"explaining synthetic stream {s}"))

    e_std = E_pool.std(axis=0)
    adv_std = np.concatenate([x_std, e_std])
    adv_std[adv_std == 0] = 1.0
    e_std[e_std == 0] = 1e-6

    data = {
        "dataset": dataset,
        "pipeline_name": pipeline_name,
        "X_pool": X_pool, "y_pool": target_model.predict(X_pool).astype(int), "E_pool": E_pool,
        "X_eval": X_eval, "y_eval": target_model.predict(X_eval).astype(int), "E_eval": E_eval,
        "X_synth": np.stack(X_synth),
        "y_synth": np.stack([target_model.predict(Xs).astype(int) for Xs in X_synth]),
        "E_synth": np.stack(E_synth),
        "x_mean": x_mean, "x_std": x_std,
        "e_std": e_std,
        "adv_mean": np.concatenate([x_mean, E_pool.mean(axis=0)]),
        "adv_std": adv_std,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump(data, path)
    print(f"[prepare] saved {path}")
    return path


def load_cache(dataset):
    path = cache_path(dataset)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run `python train.py prepare --datasets {dataset}` first.")
    return joblib.load(path)
