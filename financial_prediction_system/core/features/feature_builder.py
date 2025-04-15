"""
Feature Builder Module

This module implements the Builder pattern for creating feature sets with flexible
composition and dimensionality reduction capabilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any

# Import dimensionality reduction classes
from financial_prediction_system.core.features.dimensionality_reduction.dimensionality_reducer_base import FeatureTransformer
from financial_prediction_system.core.features.dimensionality_reduction.principal_component_analysis import PCAReducer
from financial_prediction_system.core.features.dimensionality_reduction.kmeans_cluster import KMeansReducer

class FeatureBuilder:
    """Builder for creating feature sets with flexible composition"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the feature builder
        
        Parameters
        ----------
        data : pd.DataFrame
            The input data
        """
        self.data = data
        self.features = {}
        self.transformers = {}    

    
    def add_pca_reduction(
        self, 
        columns: List[str] = None, 
        n_components: int = None,
        whiten: bool = False,
        random_state: int = None,
        transformer_name: str = 'pca'
    ) -> 'FeatureBuilder':
        """
        Add PCA dimensionality reduction
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to apply PCA to. If None, use all numeric columns
        n_components : int, optional
            Number of components to keep
        whiten : bool, default=False
            When True, whiten the components
        random_state : int, optional
            Random state for reproducibility
        transformer_name : str, default='pca'
            Name for the transformer
        
        Returns
        -------
        self : FeatureBuilder
            The builder instance for method chaining
        """
        if columns is None:
            # Use all numeric columns
            numeric_cols = self.data.select_dtypes(include='number').columns.tolist()
            columns = numeric_cols
        
        if not columns:
            return self  # Skip if no columns selected
        
        # Create the transformer
        transformer = PCAReducer(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )
        
        # Store the transformer for later use
        self.transformers[transformer_name] = transformer
        
        # Fit and transform the data
        transformed = transformer.fit_transform(self.data[columns])
        
        # Add the transformed features
        for col in transformed.columns:
            self.features[col] = transformed[col]
        
        return self
    
    def add_kmeans_reduction(
        self, 
        columns: List[str] = None, 
        n_clusters: int = 8,
        random_state: int = None,
        transformer_name: str = 'kmeans'
    ) -> 'FeatureBuilder':
        """
        Add KMeans clustering dimensionality reduction
        
        Parameters
        ----------
        columns : List[str], optional
            Columns to apply KMeans to. If None, use all numeric columns
        n_clusters : int, default=8
            Number of clusters to form
        random_state : int, optional
            Random state for reproducibility
        transformer_name : str, default='kmeans'
            Name for the transformer
        
        Returns
        -------
        self : FeatureBuilder
            The builder instance for method chaining
        """
        if columns is None:
            # Use all numeric columns
            numeric_cols = self.data.select_dtypes(include='number').columns.tolist()
            columns = numeric_cols
        
        if not columns:
            return self  # Skip if no columns selected
        
        # Create the transformer
        transformer = KMeansReducer(
            n_clusters=n_clusters,
            random_state=random_state
        )
        
        # Store the transformer for later use
        self.transformers[transformer_name] = transformer
        
        # Fit and transform the data
        transformed = transformer.fit_transform(self.data[columns])
        
        # Add the transformed features
        for col in transformed.columns:
            self.features[col] = transformed[col]
        
        # Add cluster assignment as a feature
        self.features[f'{transformer_name}_cluster'] = transformer.predict(self.data[columns])
        
        return self
    
    def add_feature_set(self, feature_set_name: str, **kwargs) -> 'FeatureBuilder':
        """
        Add a predefined set of features
        
        Parameters
        ----------
        feature_set_name : str
            Name of the feature set to add
        **kwargs : dict
            Parameters to pass to the feature generator
        
        Returns
        -------
        self : FeatureBuilder
            The builder instance for method chaining
        """
        # This would be expanded to support loading different feature sets
        # based on the feature_set_name parameter
        if feature_set_name == 'technical':
            from financial_prediction_system.core.features.technical.trend import add_technical_features
            add_technical_features(self, **kwargs)
        elif feature_set_name == 'volume':
            from financial_prediction_system.core.features.technical.volume import add_volume_features
            add_volume_features(self, **kwargs)
        elif feature_set_name == 'date':
            from financial_prediction_system.core.features.time_seasonality.day_of_week import add_date_features
            add_date_features(self, **kwargs)
        elif feature_set_name == 'treasury_rate':
            from financial_prediction_system.core.features.treasury_yield.yields import add_treasury_yield_features
            add_treasury_yield_features(self, **kwargs) 
        elif feature_set_name == 'treasury_rate_equity_relationship':
            from financial_prediction_system.core.features.treasury_yield.rates_equity_relationship import add_rates_equity_features
            add_rates_equity_features(self, **kwargs)
        elif feature_set_name == 'volatility':
            from financial_prediction_system.core.features.technical.volatility import add_volatility_features
            add_volatility_features(self, **kwargs)
        elif feature_set_name == 'price_action_ranges':
            from financial_prediction_system.core.features.price_action_pattens.ranges import add_price_range_features
            add_price_range_features(self, **kwargs)  
        elif feature_set_name == 'price_action_gaps':
            from financial_prediction_system.core.features.price_action_pattens.gaps import add_price_gap_features
            add_price_gap_features(self, **kwargs)
        elif feature_set_name == 'sector_behavior':
            from financial_prediction_system.core.features.market.sector_behavior import add_sector_relative_features
            add_sector_relative_features(self, **kwargs)
        elif feature_set_name == 'market_index_relationship':
            from financial_prediction_system.core.features.market.index_features import add_market_index_features
            add_market_index_features(self, **kwargs)
        elif feature_set_name == 'market_regime':
            from financial_prediction_system.core.features.market.market_regime import add_market_regime_features
            add_market_regime_features(self, **kwargs)
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")
        
        return self
    
    def build(self) -> pd.DataFrame:
        """
        Build the feature set
        
        Returns
        -------
        pd.DataFrame
            The complete feature set
        """
        # Convert features dictionary to DataFrame
        features_df = pd.DataFrame(self.features)
        
        # Drop NaN values that may have been introduced by rolling windows
        features_df = features_df.dropna()
        
        return features_df
    
    def get_transformer(self, name: str) -> Optional[FeatureTransformer]:
        """
        Get a transformer by name
        
        Parameters
        ----------
        name : str
            The name of the transformer
        
        Returns
        -------
        Optional[FeatureTransformer]
            The transformer or None if not found
        """
        return self.transformers.get(name)