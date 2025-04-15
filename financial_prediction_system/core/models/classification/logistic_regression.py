from typing import Dict, Any, Optional, Union
import numpy as np
import joblib
import torch
from sklearn.linear_model import LogisticRegression

from ..base import PredictionModel
from ..factory import ModelFactory


class LogisticRegressionModel(PredictionModel):
    """PyTorch-compatible wrapper for Logistic Regression.
    
    Simple but powerful for probability-based trading signals 
    with high interpretability.
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 penalty: str = 'l2',
                 solver: str = 'lbfgs',
                 max_iter: int = 100,
                 multi_class: str = 'auto',
                 random_state: Optional[int] = None,
                 class_weight: Optional[Union[str, Dict]] = None):
        """Initialize the Logistic Regression model.
        
        Args:
            C: Inverse of regularization strength
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet' or 'none')
            solver: Algorithm for optimization
            max_iter: Maximum number of iterations for solver
            multi_class: Strategy for multi-class classification
            random_state: Random state for reproducibility
            class_weight: Weights associated with classes
        """
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=random_state,
            class_weight=class_weight
        )
        
    def train(self, features, targets, **params):
        """Train the Logistic Regression model.
        
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
        """Generate predictions from the Logistic Regression model.
        
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
        probabilities = self.model.predict_proba(features)
        
        # Convert back to torch tensor if input was a torch tensor
        if is_torch:
            probabilities = torch.tensor(probabilities)
            
        return probabilities
        
    def evaluate(self, features, targets):
        """Evaluate the Logistic Regression model.
        
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
        
        # Get coefficients for interpretation
        coefficients = self.model.coef_.tolist() if hasattr(self.model, "coef_") else None
        intercept = self.model.intercept_.tolist() if hasattr(self.model, "intercept_") else None
        
        # Additional metrics could be added here
        metrics = {
            "accuracy": accuracy,
            "coefficients": coefficients,
            "intercept": intercept
        }
        
        return metrics
        
    def save(self, path):
        """Save the Logistic Regression model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load the Logistic Regression model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model
        """
        self.model = joblib.load(path)
        return self


# Register the model with the factory
ModelFactory.register("logistic_regression", LogisticRegressionModel)
