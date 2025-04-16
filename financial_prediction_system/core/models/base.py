from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np

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

    def explain(self, features: Union[pd.DataFrame, np.ndarray], 
               background_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
               num_samples: int = 100) -> Dict[str, Any]:
        """Explain model predictions using SHAP.
        
        This method creates a ModelExplainer based on the model type
        and uses it to generate SHAP explanations.
        
        Args:
            features: Data to generate explanations for
            background_data: Optional background data for SHAP
            num_samples: Number of samples to use for explanation
            
        Returns:
            Dictionary with SHAP explanation results
            
        Note:
            Concrete model classes may override this with model-specific
            explanation logic.
        """
        from financial_prediction_system.core.evaluation.model_explainer import ModelExplainer
        
        # Use a sample of features if too large
        if hasattr(features, 'shape') and features.shape[0] > num_samples:
            if isinstance(features, pd.DataFrame):
                sample_features = features.sample(num_samples, random_state=42)
            else:
                indices = np.random.RandomState(42).choice(features.shape[0], num_samples, replace=False)
                sample_features = features[indices]
        else:
            sample_features = features
            
        # Get feature names if available
        feature_names = list(features.columns) if hasattr(features, 'columns') else None
        
        # Create explainer based on model type
        explainer = ModelExplainer(self, self.__class__.__name__.lower(), feature_names)
        
        # Set background data if provided
        if background_data is not None:
            explainer.set_background_data(background_data)
            
        # Generate explanations
        explanation = explainer.explain(sample_features)
        
        return explanation 