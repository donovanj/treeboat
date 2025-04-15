from typing import Dict, Any, Optional, Union, List
import numpy as np
import joblib
import torch
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ..base import PredictionModel
from ..factory import ModelFactory


class GradientBoostingModel(PredictionModel):
    """PyTorch-compatible wrapper for Gradient Boosting.
    
    Can be used for both classification (market direction) and regression (price prediction)
    by combining multiple weak learners for strong predictive performance.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 subsample: float = 1.0,
                 max_features: Optional[Union[str, int, float]] = None,
                 random_state: Optional[int] = None,
                 regression: bool = False):
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
            regression: Whether to use GradientBoostingRegressor instead of GradientBoostingClassifier
        """
        self.regression = regression
        
        # Common parameters for both classifier and regressor
        common_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'subsample': subsample,
            'max_features': max_features,
            'random_state': random_state
        }
        
        if self.regression:
            self.model = GradientBoostingRegressor(**common_params)
        else:
            self.model = GradientBoostingClassifier(**common_params)
        
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
            
        # Auto-detect regression if the target has continuous values
        if not self.regression:
            # Check if targets appear to be continuous
            unique_targets = np.unique(targets)
            if len(unique_targets) > 10 or np.issubdtype(targets.dtype, np.floating):
                print("Detected continuous targets, switching to regression mode")
                # Reinitialize as a regressor with the same parameters
                regressor_params = {
                    'n_estimators': self.model.n_estimators,
                    'learning_rate': self.model.learning_rate,
                    'max_depth': self.model.max_depth,
                    'min_samples_split': self.model.min_samples_split,
                    'min_samples_leaf': self.model.min_samples_leaf,
                    'subsample': self.model.subsample,
                    'max_features': self.model.max_features,
                    'random_state': self.model.random_state
                }
                self.model = GradientBoostingRegressor(**regressor_params)
                self.regression = True
            
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
    
    def staged_predict_proba(self, features):
        """Generate class probabilities for each boosting iteration (classification only).
        
        Args:
            features: Input features (numpy array or torch tensor)
            
        Returns:
            List[numpy.ndarray] or List[torch.Tensor]: Class probabilities per stage
        """
        if self.regression:
            raise ValueError("staged_predict_proba is only available for classification models")
            
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
            
        # Calculate score
        score = self.model.score(features, targets)
        
        # Get feature importances
        feature_importances = self.model.feature_importances_.tolist() if hasattr(self.model, "feature_importances_") else None
        
        # Prepare appropriate metrics based on model type
        if self.regression:
            predictions = self.model.predict(features)
            mse = np.mean((predictions - targets) ** 2)
            mae = np.mean(np.abs(predictions - targets))
            
            metrics = {
                "r2": score,
                "mse": mse,
                "mae": mae,
                "feature_importances": feature_importances,
                "train_score": self.model.train_score_.tolist() if hasattr(self.model, "train_score_") else None
            }
        else:
            metrics = {
                "accuracy": score,
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
        self.regression = isinstance(self.model, GradientBoostingRegressor)
        return self


# Register the model with the factory
ModelFactory.register("gradient_boosting", GradientBoostingModel)
