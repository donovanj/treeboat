"""
Feature Director to orchestrate the feature building process
"""

import pandas as pd
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

class FeatureDirector:
    """Director class that orchestrates the feature building process"""
    
    def __init__(self, builder: FeatureBuilder):
        """
        Initialize the director
        
        Parameters
        ----------
        builder : FeatureBuilder
            The feature builder to use
        """
        self.builder = builder
    
    def build_technical_features(self) -> pd.DataFrame:
        """
        Build a set of technical features
        
        Returns
        -------
        pd.DataFrame
            The built features
        """
        return (self.builder
                .add_feature_set('technical')
                .add_feature_set('volume')
                .build())
    
    def build_dimensionality_reduced_features(
        self, 
        n_components: int = None, 
        method: str = 'pca'
    ) -> pd.DataFrame:
        """
        Build features with dimensionality reduction
        
        Parameters
        ----------
        n_components : int, optional
            Number of components to keep
        method : str, default='pca'
            Dimensionality reduction method ('pca' or 'kmeans')
        
        Returns
        -------
        pd.DataFrame
            The built features
        """
        builder = self.builder.add_feature_set('technical').add_feature_set('volume')
        
        if method == 'pca':
            builder = builder.add_pca_reduction(n_components=n_components)
        elif method == 'kmeans':
            builder = builder.add_kmeans_reduction(n_clusters=n_components or 8)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
        return builder.build()
    
    def build_full_feature_set(self) -> pd.DataFrame:
        """
        Build a complete feature set with all available features
        
        Returns
        -------
        pd.DataFrame
            The built features
        """
        return (self.builder
                .add_feature_set('technical')
                .add_feature_set('volume')
                .add_feature_set('date')
                .build()) 