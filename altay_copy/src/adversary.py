import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss


class Adversary:
    """Online surrogate model S trained on (query, obfuscated explanation) -> target label."""

    def __init__(self, input_dim, feature_mean=None, feature_std=None, random_state=42):
        self.input_dim = input_dim
        self.surrogate = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=1,
            random_state=random_state
        )
        self.is_initialized = False
        self.classes = np.array([0, 1])
        # x and e live on very different scales - without this the MLP saturates and the error becomes meaningless.
        self.feature_mean = feature_mean if feature_mean is not None else 0.0
        self.feature_std = feature_std if feature_std is not None else 1.0

    def _scale(self, X_batch):
        return (X_batch - self.feature_mean) / self.feature_std

    def update(self, X_batch, y_batch):
        # Always pass classes to partial_fit when processing single samples
        self.surrogate.partial_fit(self._scale(X_batch), y_batch, classes=self.classes)
        self.is_initialized = True

    def error(self, X_batch, y_true):
        """Adversary error: cross-entropy between the surrogate's predictions and the target model's labels.

        This is the quantity the *defender* wants to be HIGH (the surrogate imitates the target badly).
        """
        if not self.is_initialized:
            return 1.0

        # Cross-entropy is well defined for single-class windows as long as `labels` is given.
        # (A former constant 1.0 for such windows made the metric flat on imbalanced data like Credit Card Fraud.)
        surrogate_probs = self.surrogate.predict_proba(self._scale(X_batch))
        return log_loss(y_true, surrogate_probs, labels=self.classes)
