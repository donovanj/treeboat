"""
Model Feature Builder Module

This module integrates the FeatureBuilder with the models system by creating
a model-compatible feature pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from financial_prediction_system.core.features.feature_builder import (
    FeatureBuilder, 
    FeatureSelector,
    FeatureDirector
)

class ModelFeaturePipeline:
    """
    Pipeline that connects feature building to the model system
    
    This class prepares features for model training and prediction
    while ensuring compatibility with the model interfaces.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model feature pipeline
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary for the pipeline
        """
        self.config = config or {}
        self.feature_builder = None
        self.feature_selector = None
        self.features = None
        self.feature_columns = None
        
    def prepare_features(self, data: pd.DataFrame, target_col: str = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features from input data
        
        Parameters
        ----------
        data : pd.DataFrame
            The input data containing raw features
        target_col : str, optional
            The name of the target column if available
            
        Returns
        -------
        Tuple[pd.DataFrame, Optional[pd.Series]]
            A tuple of (features, target) where target may be None if target_col is None
        """
        # Extract target if provided
        target = None
        if target_col and target_col in data.columns:
            target = data[target_col].copy()
            data = data.drop(columns=[target_col])
            
        # Create feature builder
        self.feature_builder = FeatureBuilder(data)
        
        # Add features based on configuration
        if self.config.get('use_technical_features', True):
            self.feature_builder.add_feature_set('technical', 
                                                window_sizes=self.config.get('window_sizes', [5, 10, 20]))
            
        if self.config.get('use_volume_features', True):
            self.feature_builder.add_feature_set('volume')
            
        if self.config.get('use_date_features', True):
            self.feature_builder.add_feature_set('date')
            
        if self.config.get('use_price_action_features', False):
            self.feature_builder.add_feature_set('price_action_ranges')
            self.feature_builder.add_feature_set('price_action_gaps')
            
        if self.config.get('use_market_regime_features', False):
            self.feature_builder.add_feature_set('market_regime')
            
        if self.config.get('use_sector_features', False):
            self.feature_builder.add_feature_set('sector_behavior')
        
        # Apply dimensionality reduction if configured
        dim_reduction_method = self.config.get('dimensionality_reduction')
        if dim_reduction_method:
            n_components = self.config.get('n_components')
            
            if dim_reduction_method == 'pca':
                self.feature_builder.add_pca_reduction(
                    n_components=n_components,
                    whiten=self.config.get('whiten', False),
                    random_state=self.config.get('random_state')
                )
            elif dim_reduction_method == 'kmeans':
                self.feature_builder.add_kmeans_reduction(
                    n_clusters=n_components or 8,
                    random_state=self.config.get('random_state')
                )
        
        # Build features
        self.features = self.feature_builder.build()
        
        # Apply feature selection if configured
        selection_method = self.config.get('feature_selection')
        if selection_method:
            selection_params = self.config.get('selection_params', {})
            self.feature_selector = FeatureSelector(
                method=selection_method,
                **selection_params
            )
            self.features = self.feature_selector.select_features(self.features, target)
        
        # Save feature columns for future reference
        self.feature_columns = self.features.columns.tolist()
        
        return self.features, target
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the existing feature pipeline
        
        Parameters
        ----------
        data : pd.DataFrame
            New data to transform
            
        Returns
        -------
        pd.DataFrame
            Transformed features
        """
        if self.feature_builder is None:
            raise ValueError("Feature pipeline not initialized. Call prepare_features first.")
        
        # Create a new feature builder for the new data
        new_builder = FeatureBuilder(data)
        
        # Apply the same transformations
        if self.config.get('use_technical_features', True):
            new_builder.add_feature_set('technical', 
                                       window_sizes=self.config.get('window_sizes', [5, 10, 20]))
            
        if self.config.get('use_volume_features', True):
            new_builder.add_feature_set('volume')
            
        if self.config.get('use_date_features', True):
            new_builder.add_feature_set('date')
            
        if self.config.get('use_price_action_features', False):
            new_builder.add_feature_set('price_action_ranges')
            new_builder.add_feature_set('price_action_gaps')
            
        if self.config.get('use_market_regime_features', False):
            new_builder.add_feature_set('market_regime')
            
        if self.config.get('use_sector_features', False):
            new_builder.add_feature_set('sector_behavior')
        
        # Apply same dimensionality reduction
        dim_reduction_method = self.config.get('dimensionality_reduction')
        if dim_reduction_method:
            # Use the transformers from the original feature builder
            for name, transformer in self.feature_builder.transformers.items():
                # Apply transformer to new data
                if dim_reduction_method == 'pca' and name == 'pca':
                    columns = data.select_dtypes(include='number').columns.tolist()
                    transformed = transformer.transform(data[columns])
                    for col in transformed.columns:
                        new_builder.features[col] = transformed[col]
                elif dim_reduction_method == 'kmeans' and name == 'kmeans':
                    columns = data.select_dtypes(include='number').columns.tolist()
                    transformed = transformer.transform(data[columns])
                    for col in transformed.columns:
                        new_builder.features[col] = transformed[col]
                    # Add cluster assignment
                    new_builder.features[f'{name}_cluster'] = transformer.predict(data[columns])
        
        features = new_builder.build()
        
        # Ensure only the original feature columns are included
        if self.feature_columns:
            # Get intersection of available columns
            available_cols = [col for col in self.feature_columns if col in features.columns]
            features = features[available_cols]
            
            # Check if any columns are missing
            missing_cols = [col for col in self.feature_columns if col not in features.columns]
            if missing_cols:
                print(f"Warning: {len(missing_cols)} features are missing from the new data: {missing_cols[:5]}...")
        
        return features
    
    def prepare_sequence_data(self, features: pd.DataFrame, targets: Optional[pd.Series] = None, 
                              sequence_length: int = 20) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequence data for time series models like LSTM or Transformer
        
        Parameters
        ----------
        features : pd.DataFrame
            Feature data
        targets : pd.Series, optional
            Target data
        sequence_length : int, default=20
            Length of sequences to create
            
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Tuple of (feature_sequences, target_values) where target_values may be None
        """
        if features is None or len(features) < sequence_length:
            raise ValueError(f"Not enough data to create sequences of length {sequence_length}")
        
        # Convert to numpy arrays
        features_np = features.values
        
        # Create sequences
        n_samples, n_features = features_np.shape
        n_sequences = n_samples - sequence_length + 1
        
        # Initialize sequences array
        sequences = np.zeros((n_sequences, sequence_length, n_features))
        
        # Fill sequences
        for i in range(n_sequences):
            sequences[i] = features_np[i:i + sequence_length]
        
        # Process targets if provided
        target_values = None
        if targets is not None:
            # For each sequence, we use the target value at the end of the sequence
            target_values = targets.iloc[sequence_length-1:].values
            
            # Ensure target_values has the right shape
            if len(target_values) != n_sequences:
                raise ValueError(f"Target values length {len(target_values)} does not match "
                                 f"number of sequences {n_sequences}")
        
        return sequences, target_values
    
    def get_feature_importance(self, model=None) -> pd.DataFrame:
        """
        Get feature importance if available
        
        Parameters
        ----------
        model : object, optional
            The trained model that provides feature importance
            
        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and their importance
        """
        if self.feature_columns is None:
            raise ValueError("Feature pipeline not initialized. Call prepare_features first.")
            
        # Try to get feature importance from the model
        importance = None
        if model is not None:
            # Try different attributes commonly used for feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                if importance.ndim > 1 and importance.shape[0] == 1:
                    importance = importance.ravel()
            
            if importance is not None:
                # Create DataFrame with feature names and importance
                if len(importance) == len(self.feature_columns):
                    importance_df = pd.DataFrame({
                        'feature': self.feature_columns,
                        'importance': importance
                    })
                    return importance_df.sort_values('importance', ascending=False)
        
        # If no importance available, return empty DataFrame
        return pd.DataFrame(columns=['feature', 'importance']) 