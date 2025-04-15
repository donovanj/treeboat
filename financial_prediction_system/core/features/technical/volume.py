"""
Volume-based features module
"""

from typing import List
import pandas as pd
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_volume_features(builder: FeatureBuilder, window_sizes: List[int] = [5, 10, 20]) -> FeatureBuilder:
    """
    Add volume-based features
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    window_sizes : List[int], default=[5, 10, 20]
        Window sizes for volume indicators
    
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    volume_cols = [col for col in data.columns if 'volume' in col.lower()]
    
    if not volume_cols:
        return builder  # Skip if no volume data available
    
    volume_col = volume_cols[0]
    
    # Volume moving averages
    for window in window_sizes:
        builder.features[f'volume_ma_{window}'] = data[volume_col].rolling(window=window).mean()
    
    # Volume momentum
    for window in window_sizes:
        builder.features[f'volume_change_{window}'] = data[volume_col].pct_change(periods=window)
    
    # Volume relative to moving average
    for window in window_sizes:
        ma = data[volume_col].rolling(window=window).mean()
        builder.features[f'volume_rel_ma_{window}'] = data[volume_col] / ma
    
    return builder