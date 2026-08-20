import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import lime.lime_tabular
import xgboost as xgb
import shap

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
