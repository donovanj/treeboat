"""
Price action gap features module
"""
from typing import List
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_price_gap_features(builder: FeatureBuilder, lookback_periods: List[int] = [1, 5, 10, 20]) -> FeatureBuilder:
    """
    Add price action gap-based features
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    lookback_periods : List[int], default=[1, 5, 10, 20]
        Periods to analyze for gap patterns
        
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
    
    # Gap up: today's open higher than yesterday's high
    builder.features['gap_up'] = (data['open'] > data['high'].shift(1)).astype(int)
    
    # Gap down: today's open lower than yesterday's low
    builder.features['gap_down'] = (data['open'] < data['low'].shift(1)).astype(int)
    
    # Calculate gap size (as percentage)
    builder.features['gap_size'] = np.where(
        builder.features['gap_up'] == 1,
        (data['open'] - data['high'].shift(1)) / data['high'].shift(1),
        np.where(
            builder.features['gap_down'] == 1,
            (data['open'] - data['low'].shift(1)) / data['low'].shift(1),
            0
        )
    )
    
    # Gap fill tracking
    # Gap up filled if low touches or goes below previous high
    builder.features['gap_up_filled'] = np.where(
        builder.features['gap_up'] == 1,
        (data['low'] <= data['high'].shift(1)).astype(int),
        0
    )
    
    # Gap down filled if high touches or goes above previous low
    builder.features['gap_down_filled'] = np.where(
        builder.features['gap_down'] == 1,
        (data['high'] >= data['low'].shift(1)).astype(int),
        0
    )
    
    # Track days to fill gap
    for lookback in lookback_periods:
        # Initialize tracking arrays
        gap_up_days_to_fill = np.zeros(len(data))
        gap_down_days_to_fill = np.zeros(len(data))
        
        # Loop to find how many days it took to fill each gap
        for i in range(lookback, len(data)):
            if builder.features['gap_up'].iloc[i-lookback] == 1:
                # Check each day forward if gap was filled
                for j in range(1, lookback+1):
                    if i-lookback+j < len(data) and data['low'].iloc[i-lookback+j] <= data['high'].iloc[i-lookback-1]:
                        gap_up_days_to_fill[i-lookback] = j
                        break
            
            if builder.features['gap_down'].iloc[i-lookback] == 1:
                # Check each day forward if gap was filled
                for j in range(1, lookback+1):
                    if i-lookback+j < len(data) and data['high'].iloc[i-lookback+j] >= data['low'].iloc[i-lookback-1]:
                        gap_down_days_to_fill[i-lookback] = j
                        break
        
        builder.features[f'gap_up_days_to_fill_{lookback}'] = pd.Series(gap_up_days_to_fill, index=data.index)
        builder.features[f'gap_down_days_to_fill_{lookback}'] = pd.Series(gap_down_days_to_fill, index=data.index)
    
    # Recent gap features
    for period in lookback_periods:
        # Number of gaps in past N periods
        builder.features[f'num_gaps_{period}'] = (
            builder.features['gap_up'].rolling(window=period).sum() + 
            builder.features['gap_down'].rolling(window=period).sum()
        )
        
        # Average gap size over past N periods
        mask = (builder.features['gap_up'] | builder.features['gap_down']).astype(bool)
        gap_sizes = builder.features['gap_size'][mask]
        
        if not gap_sizes.empty:
            builder.features[f'avg_gap_size_{period}'] = builder.features['gap_size'].replace(0, np.nan).rolling(
                window=period, min_periods=1).mean().fillna(0)
        
        # Unfilled gaps count
        gap_up_unfilled = (builder.features['gap_up'] - builder.features['gap_up_filled']).clip(lower=0)
        gap_down_unfilled = (builder.features['gap_down'] - builder.features['gap_down_filled']).clip(lower=0)
        
        builder.features[f'unfilled_gaps_{period}'] = (
            gap_up_unfilled.rolling(window=period).sum() +
            gap_down_unfilled.rolling(window=period).sum()
        )
    
    # Gap island patterns
    # Gap up followed by gap down
    builder.features['island_top'] = (
        (builder.features['gap_up'] == 1) & 
        (builder.features['gap_down'].shift(-1) == 1)
    ).astype(int)
    
    # Gap down followed by gap up
    builder.features['island_bottom'] = (
        (builder.features['gap_down'] == 1) & 
        (builder.features['gap_up'].shift(-1) == 1)
    ).astype(int)
    
    # Runaway gaps (followed by another gap in same direction)
    builder.features['runaway_gap_up'] = (
        (builder.features['gap_up'] == 1) & 
        (builder.features['gap_up'].shift(-1) == 1)
    ).astype(int)
    
    builder.features['runaway_gap_down'] = (
        (builder.features['gap_down'] == 1) & 
        (builder.features['gap_down'].shift(-1) == 1)
    ).astype(int)
    
    # Exhaustion gaps (after trend, large gap, followed by reversal)
    # Simplified version - this could be enhanced with trend detection
    builder.features['exhaustion_gap_up'] = (
        (builder.features['gap_up'] == 1) & 
        (builder.features['gap_size'] > builder.features['gap_size'].rolling(window=5).mean() * 1.5) &
        (data['close'].shift(-1) < data['open'].shift(-1))
    ).astype(int)
    
    builder.features['exhaustion_gap_down'] = (
        (builder.features['gap_down'] == 1) & 
        (abs(builder.features['gap_size']) > abs(builder.features['gap_size'].rolling(window=5).mean() * 1.5)) &
        (data['close'].shift(-1) > data['open'].shift(-1))
    ).astype(int)
    
    return builder