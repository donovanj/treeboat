from typing import Dict, Any, Optional, Union
import numpy as np
import joblib
import torch
from sklearn.svm import SVC, SVR

from ..base import PredictionModel
from ..factory import ModelFactory


class SVMModel(PredictionModel):
    """PyTorch-compatible wrapper for Support Vector Machines.
    
    Can be used for both classification (market direction) and regression (price prediction)
    with good performance in high-dimensional spaces.
    """
    
    def __init__(self, 
                 C: float = 1.0,
                 kernel: str = 'rbf',
                 degree: int = 3,
                 gamma: Optional[Union[str, float]] = 'scale',
                 probability: bool = True,
                 random_state: Optional[int] = None,
                 class_weight: Optional[Union[str, Dict]] = None,
                 epsilon: float = 0.1,
                 regression: bool = False):
        """Initialize the SVM model.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type to be used ('linear', 'poly', 'rbf', 'sigmoid')
            degree: Degree of polynomial kernel function
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            probability: Whether to enable probability estimates (classification only)
            random_state: Random state for reproducibility (classification only)
            class_weight: Weights associated with classes (classification only)
            epsilon: Epsilon in the epsilon-SVR model (regression only)
            regression: Whether to use SVR instead of SVC
        """
        self.regression = regression
        
        # Common parameters for both SVC and SVR
        common_params = {
            'C': C,
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
        }
        
        if self.regression:
            # For regression task (SVR)
            # Note: SVR doesn't accept random_state or probability parameters
            common_params['epsilon'] = epsilon
            self.model = SVR(**common_params)
        else:
            # For classification task (SVC)
            common_params['probability'] = probability
            common_params['class_weight'] = class_weight
            common_params['random_state'] = random_state
            self.model = SVC(**common_params)
        
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
            
        # Auto-detect regression if the target has continuous values
        if not self.regression:
            # Check if targets appear to be continuous
            unique_targets = np.unique(targets)
            if len(unique_targets) > 10 or np.issubdtype(targets.dtype, np.floating):
                print("Detected continuous targets, switching to SVR (regression mode)")
                # Reinitialize as SVR with similar parameters
                # Note: SVR doesn't accept random_state or probability parameters
                regressor_params = {
                    'C': self.model.C,
                    'kernel': self.model.kernel,
                    'degree': self.model.degree,
                    'gamma': self.model.gamma,
                    'epsilon': 0.1  # Default epsilon value
                }
                self.model = SVR(**regressor_params)
                self.regression = True
            
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
            
        # Calculate score
        score = self.model.score(features, targets)
        
        # Prepare appropriate metrics based on model type
        if self.regression:
            predictions = self.model.predict(features)
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            metrics = {
                "r2": score,
                "mse": mse,
                "mae": mae
            }
        else:
            # Get number of support vectors
            n_support = self.model.n_support_.tolist() if hasattr(self.model, "n_support_") else None
            
            metrics = {
                "accuracy": score,
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
        self.regression = isinstance(self.model, SVR)
        return self


# Register the model with the factory
ModelFactory.register("svm", SVMModel)
