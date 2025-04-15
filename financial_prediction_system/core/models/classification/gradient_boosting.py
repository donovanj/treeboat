from typing import Dict, Any, Optional, Union, List
import numpy as np
import joblib
import torch
from sklearn.ensemble import GradientBoostingClassifier

from ..base import PredictionModel
from ..factory import ModelFactory


class GradientBoostingModel(PredictionModel):
    """PyTorch-compatible wrapper for Gradient Boosting.
    
    Strong performance for predicting price movements by 
    combining multiple weak learners.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 max_features: Optional[Union[str, int, float]] = None,
                 random_state: Optional[int] = None):
        """Initialize the Gradient Boosting model.
        
        Args:
            n_estimators: Number of boosting stages to perform
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual regression estimators
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            subsample: Fraction of samples to be used for fitting the individual trees
            max_features: Number of features to consider when looking for the best split
            random_state: Random state for reproducibility
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state
        )
        
    def train(self, features, targets, **params):
        """Train the Gradient Boosting model.
        
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
        """Generate predictions from the Gradient Boosting model.
        
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
    
    def staged_predict_proba(self, features):
        """Generate class probabilities for each boosting iteration.
        
        Args:
            features: Input features (numpy array or torch tensor)
            
        Returns:
            List[numpy.ndarray] or List[torch.Tensor]: Class probabilities per stage
        """
        # Convert torch tensors to numpy if necessary
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
            is_torch = True
        else:
            is_torch = False
            
        # Generate staged probabilities
        staged_probs = list(self.model.staged_predict_proba(features))
        
        # Convert back to torch tensors if input was a torch tensor
        if is_torch:
            staged_probs = [torch.tensor(probs) for probs in staged_probs]
            
        return staged_probs
        
    def evaluate(self, features, targets):
        """Evaluate the Gradient Boosting model.
        
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
        
        # Get feature importances
        feature_importances = self.model.feature_importances_.tolist() if hasattr(self.model, "feature_importances_") else None
        
        # Additional metrics could be added here
        metrics = {
            "accuracy": accuracy,
            "feature_importances": feature_importances,
            "train_score": self.model.train_score_.tolist() if hasattr(self.model, "train_score_") else None
        }
        
        return metrics
        
    def save(self, path):
        """Save the Gradient Boosting model to disk.
        
        Args:
            path: Path to save the model
        """
        joblib.dump(self.model, path)
        
    def load(self, path):
        """Load the Gradient Boosting model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model
        """
        self.model = joblib.load(path)
        return self


# Register the model with the factory
ModelFactory.register("gradient_boosting", GradientBoostingModel)
