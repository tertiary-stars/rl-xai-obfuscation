import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier

class Adversary:
    def __init__(self, input_dim):
        self.surrogate = MLPClassifier(
            hidden_layer_sizes=(64, 32), 
            max_iter=1, 
            random_state=42 
        )
        self.is_initialized = False
        self.classes = np.array([0, 1])
        
    def update(self, X_batch, y_batch):
        self.surrogate.partial_fit(X_batch, y_batch, classes=self.classes)
        self.is_initialized = True
            
    def compute_loss(self, X_batch, y_true):
        if not self.is_initialized:
            return 1.0  
        surrogate_probs = self.surrogate.predict_proba(X_batch)
        return log_loss(y_true, surrogate_probs, labels=self.classes)