from typing import Dict, Any, Optional, Union
import numpy as np
import joblib
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..base import PredictionModel
from ..factory import ModelFactory


class RandomForestModel(PredictionModel):
    """PyTorch-compatible wrapper for Random Forest.
    
    Can be used for both classification (predicting market direction up/down)
    and regression (predicting continuous price values) tasks.
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[str, int, float]] = 'sqrt',
                 random_state: Optional[int] = None,
                 class_weight: Optional[Union[str, Dict]] = None,
                 regression: bool = False):
        """Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random state for reproducibility
            class_weight: Weights associated with classes (classification only)
            regression: Whether to use RandomForestRegressor instead of RandomForestClassifier
        """
        self.regression = regression
        
        # Parameters that apply to both regressor and classifier
        common_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state,
        }
        
        if self.regression:
            self.model = RandomForestRegressor(**common_params)
        else:
            common_params['class_weight'] = class_weight
            self.model = RandomForestClassifier(**common_params)
        
    def train(self, features, targets, **params):
        """Train the Random Forest model.
        
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
                print("Detected continuous targets, switching to regression mode")
                # Reinitialize as a regressor with the same parameters
                regressor_params = {
                    'n_estimators': self.model.n_estimators,
                    'max_depth': self.model.max_depth,
                    'min_samples_split': self.model.min_samples_split,
                    'min_samples_leaf': self.model.min_samples_leaf,
                    'max_features': self.model.max_features,
                    'random_state': self.model.random_state,
                }
                self.model = RandomForestRegressor(**regressor_params)
                self.regression = True
            
        # Train the model
        self.model.fit(features, targets)
        return self
        
    def predict(self, features):
        """Generate predictions from the Random Forest model.
        
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
        """Evaluate the Random Forest model.
        
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
            
        # Calculate accuracy or R² depending on model type
        score = self.model.score(features, targets)
        
        # Prepare appropriate metrics based on model type
        if self.regression:
            predictions = self.model.predict(features)
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            metrics = {
                "r2": score,
                "mse": mse,
                "mae": mae,
                "feature_importances": self.model.feature_importances_.tolist() if hasattr(self.model, "feature_importances_") else None
            }
        else:
            metrics = {
                "accuracy": score,
                "feature_importances": self.model.feature_importances_.tolist() if hasattr(self.model, "feature_importances_") else None
            }
        
        return metrics
        
    def save(self, path):
        """Save the Random Forest model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load the Random Forest model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model
        """
        self.model = joblib.load(path)
        self.regression = isinstance(self.model, RandomForestRegressor)
        return self


# Register the model with the factory
ModelFactory.register("random_forest", RandomForestModel)
