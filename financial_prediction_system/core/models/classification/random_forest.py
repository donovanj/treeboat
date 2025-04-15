from typing import Dict, Any, Optional, Union
import numpy as np
import joblib
import torch
from sklearn.ensemble import RandomForestClassifier

from ..base import PredictionModel
from ..factory import ModelFactory


class RandomForestModel(PredictionModel):
    """PyTorch-compatible wrapper for Random Forest.
    
    Excellent for predicting market direction (up/down) while handling 
    non-linear relationships and feature importance analysis.
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Optional[Union[str, int, float]] = 'sqrt',
                 random_state: Optional[int] = None,
                 class_weight: Optional[Union[str, Dict]] = None):
        """Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random state for reproducibility
            class_weight: Weights associated with classes
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            class_weight=class_weight
        )
        
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
            
        # Calculate accuracy
        accuracy = self.model.score(features, targets)
        
        # Additional metrics could be added here
        metrics = {
            "accuracy": accuracy,
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
        return self


# Register the model with the factory
ModelFactory.register("random_forest", RandomForestModel)
