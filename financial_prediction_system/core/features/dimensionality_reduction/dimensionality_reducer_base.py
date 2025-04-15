from abc import ABC, abstractmethod
import pandas as pd

class FeatureTransformer(ABC):
    """Abstract base class for feature transformers"""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'FeatureTransformer':
        """Fit the transformer to the data"""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data"""
        self.fit(data)
        return self.transform(data)

class DimensionalityReducer(FeatureTransformer):
    """Base class for dimensionality reduction transformers"""
    
    def __init__(self, n_components: int = None, prefix: str = ""):
        """
        Initialize the dimensionality reducer
        
        Parameters
        ----------
        n_components : int, optional
            Number of components to keep
        prefix : str, optional
            Prefix to add to feature names
        """
        self.n_components = n_components
        self.prefix = prefix
        self.model = None
        self.feature_names = None
    
    @abstractmethod
    def _create_model(self):
        """Create the underlying model"""
        pass