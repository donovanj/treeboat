from abc import ABC, abstractmethod
import torch.nn as nn

class PredictionModel(ABC):
    """Base interface for all prediction models."""
    
    @abstractmethod
    def train(self, features, targets, **params):
        """Train the model."""
        pass
        
    @abstractmethod
    def predict(self, features):
        """Generate predictions."""
        pass
        
    @abstractmethod
    def evaluate(self, features, targets):
        """Evaluate model performance."""
        pass
        
    @abstractmethod
    def save(self, path):
        """Save model to disk."""
        pass
        
    @abstractmethod
    def load(self, path):
        """Load model from disk."""
        pass 