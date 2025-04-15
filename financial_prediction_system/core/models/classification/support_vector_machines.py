from typing import Dict, Any, Optional, Union
import numpy as np
import joblib
import torch
from sklearn.svm import SVC

from ..base import PredictionModel
from ..factory import ModelFactory


class SVMModel(PredictionModel):
    """PyTorch-compatible wrapper for Support Vector Machines.
    
    Effective for binary classification of market movements with 
    good performance in high-dimensional spaces.
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: Optional[Union[str, float]] = 'scale',
                 probability: bool = True,
                 random_state: Optional[int] = None,
                 class_weight: Optional[Union[str, Dict]] = None):
        """Initialize the SVM model.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type to be used ('linear', 'poly', 'rbf', 'sigmoid')
            degree: Degree of polynomial kernel function
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            probability: Whether to enable probability estimates
            random_state: Random state for reproducibility
            class_weight: Weights associated with classes
        """
        self.model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            class_weight=class_weight
        )
        
    def train(self, features, targets, **params):
        """Train the SVM model.
        
        Args:
            features: Input features (numpy array or torch tensor)
            targets: Target values (numpy array or torch tensor)
            **params: Additional parameters for training
            
        Returns:
            self: The trained model
        """
        # Convert torch tensors to numpy if necessary
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        # Train the model
        self.model.fit(features, targets)
        return self
        
    def predict(self, features):
        """Generate predictions from the SVM model.
        
        Args:
            features: Input features (numpy array or torch tensor)
            
        Returns:
            torch.Tensor: Model predictions
        """
        # Convert torch tensors to numpy if necessary
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            is_torch = True
        else:
            is_torch = False
            
        # Generate predictions
        predictions = self.model.predict(features)
        
        # Convert back to torch tensor if input was a torch tensor
        if is_torch:
            predictions = torch.tensor(predictions)
            
        return predictions
    
    def predict_proba(self, features):
        """Generate class probabilities.
        
        Args:
            features: Input features (numpy array or torch tensor)
            
        Returns:
            torch.Tensor or numpy.ndarray: Class probabilities
        """
        # Convert torch tensors to numpy if necessary
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            is_torch = True
        else:
            is_torch = False
            
        # Generate probabilities
        if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
            probabilities = self.model.predict_proba(features)
        else:
            raise ValueError("Model was not trained with probability=True")
        
        # Convert back to torch tensor if input was a torch tensor
        if is_torch:
            probabilities = torch.tensor(probabilities)
            
        return probabilities
        
    def evaluate(self, features, targets):
        """Evaluate the SVM model.
        
        Args:
            features: Input features (numpy array or torch tensor)
            targets: Target values (numpy array or torch tensor)
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Convert torch tensors to numpy if necessary
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        # Calculate accuracy
        accuracy = self.model.score(features, targets)
        
        # Calculate number of support vectors
        n_support = self.model.n_support_.tolist() if hasattr(self.model, "n_support_") else None
        
        # Additional metrics could be added here
        metrics = {
            "accuracy": accuracy,
            "n_support_vectors": n_support
        }
        
        return metrics
        
    def save(self, path):
        """Save the SVM model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load the SVM model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model
        """
        self.model = joblib.load(path)
        return self


# Register the model with the factory
ModelFactory.register("svm", SVMModel)
