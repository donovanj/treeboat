"""
Feature selection utilities
"""

import pandas as pd
from financial_prediction_system.core.features.dimensionality_reduction.principal_component_analysis import PCAReducer
from financial_prediction_system.core.features.dimensionality_reduction.kmeans_cluster import KMeansReducer

class FeatureSelector:
    """Unified interface for feature selection methods"""
    
    def __init__(self, method: str = 'variance_threshold', **kwargs):
        """
        Initialize the feature selector
        
        Parameters
        ----------
        method : str, default='variance_threshold'
            The feature selection method to use:
            - 'variance_threshold': Remove features with low variance
            - 'pca': Use PCA for dimensionality reduction
            - 'kmeans': Use KMeans clustering for dimensionality reduction
        **kwargs
            Additional parameters for the selected method
        """
        self.method = method
        self.params = kwargs
        self.transformer = None
    
    def select_features(self, features: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        """
        Select features from the input dataset
        
        Parameters
        ----------
        features : pd.DataFrame
            The input features
        target : pd.Series, optional
            The target variable, if needed by the selection method
        
        Returns
        -------
        pd.DataFrame
            The selected features
        """
        if self.method == 'variance_threshold':
            threshold = self.params.get('threshold', 0.0)
            return self._select_by_variance(features, threshold)
        
        elif self.method == 'pca':
            n_components = self.params.get('n_components', None)
            transformer = PCAReducer(
                n_components=n_components,
                whiten=self.params.get('whiten', False),
                random_state=self.params.get('random_state', None)
            )
            self.transformer = transformer
            return transformer.fit_transform(features)
        
        elif self.method == 'kmeans':
            n_clusters = self.params.get('n_clusters', 8)
            transformer = KMeansReducer(
                n_clusters=n_clusters,
                random_state=self.params.get('random_state', None)
            )
            self.transformer = transformer
            return transformer.fit_transform(features)
        
        else:
            raise ValueError(f"Unsupported selection method: {self.method}")
    
    def _select_by_variance(self, features: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
        """
        Select features by variance threshold
        
        Parameters
        ----------
        features : pd.DataFrame
            The input features
        threshold : float, default=0.0
            Features with a variance lower than this threshold will be removed
        
        Returns
        -------
        pd.DataFrame
            The selected features
        """
        variances = features.var()
        selected_features = features.loc[:, variances > threshold]
        return selected_features

