"""
Volatility features module
"""
from typing import List
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_volatility_features(builder: FeatureBuilder, window_sizes: List[int] = [5, 10, 20, 30]) -> FeatureBuilder:
    """
    Add volatility-based features
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    window_sizes : List[int], default=[5, 10, 20, 30]
        Window sizes for volatility indicators
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    price_cols = [col for col in data.columns if any(x in col.lower() for x in ['close', 'price', 'adj_close'])]
    
    if not price_cols:
        return builder  # Skip if no price data available
    
    price_col = price_cols[0]
    
    # Standard deviation-based volatility
    for window in window_sizes:
        builder.features[f'volatility_std_{window}'] = data[price_col].rolling(window=window).std()
    
    # ATR (Average True Range)
    if all(col in data.columns for col in ['high', 'low', 'close']):
        for window in window_sizes:
            # True Range calculation
            tr1 = abs(data['high'] - data['low'])
            tr2 = abs(data['high'] - data['close'].shift(1))
            tr3 = abs(data['low'] - data['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR calculation
            builder.features[f'atr_{window}'] = true_range.rolling(window=window).mean()
            
            # Normalized ATR (ATR/Close price)
            builder.features[f'atr_pct_{window}'] = builder.features[f'atr_{window}'] / data['close']
    
    # Bollinger Bands Width
    for window in window_sizes:
        rolling_mean = data[price_col].rolling(window=window).mean()
        rolling_std = data[price_col].rolling(window=window).std()
        builder.features[f'bb_width_{window}'] = (rolling_std * 2) / rolling_mean
    
    # Historical Volatility (annualized)
    for window in window_sizes:
        log_returns = np.log(data[price_col] / data[price_col].shift(1))
        builder.features[f'hist_vol_{window}'] = log_returns.rolling(window=window).std() * np.sqrt(252)
    
    # GARCH-like volatility proxy (squared returns)
    log_returns = np.log(data[price_col] / data[price_col].shift(1))
    squared_returns = log_returns ** 2
    for window in window_sizes:
        builder.features[f'squared_returns_ma_{window}'] = squared_returns.rolling(window=window).mean()
    
    # Garman-Klass volatility estimator
    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        log_ho = np.log(data['high'] / data['open'])
        log_lo = np.log(data['low'] / data['open'])
        log_co = np.log(data['close'] / data['open'])
        gk_vol = 0.5 * (log_ho - log_lo)**2 - (2*np.log(2) - 1) * log_co**2
        
        for window in window_sizes:
            builder.features[f'garman_klass_vol_{window}'] = gk_vol.rolling(window=window).mean() * np.sqrt(252)
    
    return builder