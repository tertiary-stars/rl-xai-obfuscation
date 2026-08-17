import numpy as np
import shap
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_train_target():
    """Loads Adult Income dataset and trains an XGBoost black-box model."""
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")

    X = adult.data.select_dtypes(include=[np.number]).dropna()
    y = (adult.target.loc[X.index] == '>50K').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    target_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss', random_state=42)
    target_model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(target_model)

    return X_train.values, X_test.values, target_model, explainer