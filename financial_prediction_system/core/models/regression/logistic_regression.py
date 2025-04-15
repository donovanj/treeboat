from typing import Dict, Any, Optional, Union
import numpy as np
import joblib
import torch
from sklearn.linear_model import LogisticRegression, LinearRegression

from ..base import PredictionModel
from ..factory import ModelFactory


class LogisticRegressionModel(PredictionModel):
    """PyTorch-compatible wrapper for Logistic Regression or Linear Regression.
    
    Simple but powerful model with high interpretability.
    - Logistic Regression for classification (probability-based trading signals)
    - Linear Regression for continuous targets (price prediction)
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 penalty: str = 'l2',
                 solver: str = 'lbfgs',
                 max_iter: int = 100,
                 multi_class: str = 'auto',
                 random_state: Optional[int] = None,
                 class_weight: Optional[Union[str, Dict]] = None,
                 regression: bool = False):
        """Initialize the Regression model.
        
        Args:
            C: Inverse of regularization strength (logistic only)
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet' or 'none') (logistic only)
            solver: Algorithm for optimization (logistic only)
            max_iter: Maximum number of iterations for solver (logistic only)
            multi_class: Strategy for multi-class classification (logistic only)
            random_state: Random state for reproducibility (logistic only)
            class_weight: Weights associated with classes (logistic only)
            regression: Whether to use LinearRegression instead of LogisticRegression
        """
        self.regression = regression
        
        if self.regression:
            # For continuous targets, use LinearRegression
            self.model = LinearRegression()
        else:
            # For classification, use LogisticRegression
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
        """Train the Regression model.
        
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
            
        # Auto-detect regression if the target has continuous values
        if not self.regression:
            # Check if targets appear to be continuous
            unique_targets = np.unique(targets)
            if len(unique_targets) > 10 or np.issubdtype(targets.dtype, np.floating):
                print("Detected continuous targets, switching to LinearRegression")
                # Reinitialize as LinearRegression
                self.model = LinearRegression()
                self.regression = True
            
        # Train the model
        self.model.fit(features, targets)
        return self
        
    def predict(self, features):
        """Generate predictions from the Regression model.
        
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
        """Generate class probabilities (classification only).
        
        Args:
            features: Input features (numpy array or torch tensor)
            
        Returns:
            torch.Tensor or numpy.ndarray: Class probabilities
        """
        if self.regression:
            raise ValueError("predict_proba is only available for classification models")
            
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
        """Evaluate the Regression model.
        
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
            
        # Calculate score (r² for regression, accuracy for classification)
        score = self.model.score(features, targets)
        
        # Prepare appropriate metrics based on model type
        if self.regression:
            predictions = self.model.predict(features)
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            # Get coefficients for interpretation
            coefficients = self.model.coef_.tolist() if hasattr(self.model, "coef_") else None
            intercept = self.model.intercept_.tolist() if hasattr(self.model, "intercept_") else None
            
            metrics = {
                "r2": score,
                "mse": mse,
                "mae": mae,
                "coefficients": coefficients,
                "intercept": intercept
            }
        else:
            # Get coefficients for interpretation
            coefficients = self.model.coef_.tolist() if hasattr(self.model, "coef_") else None
            intercept = self.model.intercept_.tolist() if hasattr(self.model, "intercept_") else None
            
            metrics = {
                "accuracy": score,
                "coefficients": coefficients,
                "intercept": intercept
            }
        
        return metrics
        
    def save(self, path):
        """Save the Regression model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load the Regression model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model
        """
        self.model = joblib.load(path)
        self.regression = isinstance(self.model, LinearRegression)
        return self


# Register the model with the factory
ModelFactory.register("logistic_regression", LogisticRegressionModel)
