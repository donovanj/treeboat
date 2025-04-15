"""
Technical trend features module
"""

from typing import List
import pandas as pd
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_technical_features(builder: FeatureBuilder, window_sizes: List[int] = [5, 10, 20]) -> FeatureBuilder:
    """
    Add technical indicators as features
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    window_sizes : List[int], default=[5, 10, 20]
        Window sizes for moving averages and other indicators
    
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    price_cols = [col for col in data.columns if 'price' in col.lower() or 'close' in col.lower()]
    
    if not price_cols:
        raise ValueError("No price columns found in data")
    
    price_col = price_cols[0]
    
    # Moving averages
    for window in window_sizes:
        builder.features[f'ma_{window}'] = data[price_col].rolling(window=window).mean()
        
    # Momentum
    for window in window_sizes:
        builder.features[f'momentum_{window}'] = data[price_col].pct_change(periods=window)
    
    # Volatility
    for window in window_sizes:
        builder.features[f'volatility_{window}'] = data[price_col].rolling(window=window).std()
    
    return builder