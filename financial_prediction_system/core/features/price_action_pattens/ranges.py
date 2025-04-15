"""
Price action range features module
"""
from typing import List
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_price_range_features(builder: FeatureBuilder, window_sizes: List[int] = [5, 10, 20, 30]) -> FeatureBuilder:
    """
    Add price action range-based features
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    window_sizes : List[int], default=[5, 10, 20, 30]
        Window sizes for range indicators
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    # Check if OHLC data is available
    has_ohlc = all(col in data.columns for col in ['open', 'high', 'low', 'close'])
    if not has_ohlc:
        return builder  # Skip if no OHLC data available
    
    # Daily trading range
    builder.features['daily_range'] = data['high'] - data['low']
    builder.features['daily_range_pct'] = (data['high'] - data['low']) / data['close']
    
    # Body size (absolute and relative)
    builder.features['body_size'] = abs(data['close'] - data['open'])
    builder.features['body_size_pct'] = abs(data['close'] - data['open']) / data['close']
    
    # Upper and lower shadows (wicks)
    builder.features['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    builder.features['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
    
    # Shadow to body ratio
    shadows_sum = builder.features['upper_shadow'] + builder.features['lower_shadow']
    body_size = builder.features['body_size']
    builder.features['shadow_to_body_ratio'] = np.where(
        body_size > 0, 
        shadows_sum / body_size, 
        np.nan
    )
    
    # Moving window ranges
    for window in window_sizes:
        # N-day high-low range
        builder.features[f'range_{window}_day'] = data['high'].rolling(window=window).max() - data['low'].rolling(window=window).min()
        builder.features[f'range_{window}_day_pct'] = builder.features[f'range_{window}_day'] / data['close']
        
        # Position within N-day range (0-1)
        high_n = data['high'].rolling(window=window).max()
        low_n = data['low'].rolling(window=window).min()
        range_n = high_n - low_n
        builder.features[f'pos_in_{window}d_range'] = np.where(
            range_n > 0,
            (data['close'] - low_n) / range_n,
            0.5
        )
        
        # N-day moving average of daily ranges
        builder.features[f'avg_daily_range_{window}'] = builder.features['daily_range'].rolling(window=window).mean()
        builder.features[f'avg_daily_range_pct_{window}'] = builder.features['daily_range_pct'].rolling(window=window).mean()
    
    # Identify contracting and expanding ranges
    for window in window_sizes:
        # Range contraction/expansion
        builder.features[f'range_contraction_{window}'] = builder.features[f'avg_daily_range_{window}'] / builder.features[f'avg_daily_range_{window}'].shift(window)
        
        # Narrowest range N (NR signal)
        rolling_min_range = builder.features['daily_range'].rolling(window=window).min()
        builder.features[f'is_NR{window}'] = (builder.features['daily_range'] == rolling_min_range).astype(int)
        
        # Widest range N (WR signal)
        rolling_max_range = builder.features['daily_range'].rolling(window=window).max()
        builder.features[f'is_WR{window}'] = (builder.features['daily_range'] == rolling_max_range).astype(int)
    
    # Inside/outside day patterns
    builder.features['inside_day'] = ((data['high'] <= data['high'].shift(1)) & 
                                      (data['low'] >= data['low'].shift(1))).astype(int)
    
    builder.features['outside_day'] = ((data['high'] >= data['high'].shift(1)) & 
                                       (data['low'] <= data['low'].shift(1))).astype(int)
    
    # Consecutive inside days
    for window in [2, 3, 4]:
        builder.features[f'inside_day_{window}'] = builder.features['inside_day'].rolling(window=window).sum().apply(lambda x: 1 if x == window else 0)
    
    # Trading ranges and breakouts
    for window in window_sizes:
        # Identify if price is making new highs or lows relative to previous windows
        builder.features[f'new_high_{window}'] = (data['high'] > data['high'].shift(1).rolling(window=window).max()).astype(int)
        builder.features[f'new_low_{window}'] = (data['low'] < data['low'].shift(1).rolling(window=window).min()).astype(int)
    
    return builder