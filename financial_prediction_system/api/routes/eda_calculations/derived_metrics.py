import pandas as pd
import numpy as np


def calculate_volume_metrics(data: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Calculates various volume-related metrics."""
    metrics = pd.DataFrame(index=data.index)
    
    # Ensure volume exists and is numeric
    if 'volume' not in data.columns or not pd.api.types.is_numeric_dtype(data['volume']):
        return metrics # Return empty DataFrame if no valid volume data
        
    volume = data['volume'].replace(0, np.nan) # Avoid division by zero

    # volume Change
    metrics['volume_change'] = volume.pct_change()

    # Normalized volume
    rolling_mean_volume = volume.rolling(window=window, min_periods=1).mean()
    metrics['volume_normalized'] = volume / rolling_mean_volume

    # Dollar volume (Price-volume Metric)
    if 'close' in data.columns and pd.api.types.is_numeric_dtype(data['close']):
        metrics['dollar_volume'] = data['close'] * data['volume'] # Use original volume here
        
    return metrics.replace([np.inf, -np.inf], np.nan)

def calculate_price_pattern_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates metrics based on OHLC price patterns."""
    metrics = pd.DataFrame(index=data.index)
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in data.columns and pd.api.types.is_numeric_dtype(data[col]) for col in required_cols):
        return metrics # Return empty if required columns are missing or not numeric
        
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']

    # Candlestick Body Size (Normalized by close)
    metrics['body_size_norm'] = (open_ - close).abs() / close

    # Upper Shadow (Normalized by close)
    metrics['upper_shadow_norm'] = (high - np.maximum(open_, close)) / close

    # lower Shadow (Normalized by close)
    metrics['lower_shadow_norm'] = (np.minimum(open_, close) - low) / close

    # close Position within Range
    hl_range = high - low
    # Avoid division by zero if high == low
    metrics['close_position'] = ((close - low) / hl_range).replace([np.inf, -np.inf], np.nan).fillna(0.5) # Fill NaNs where H=L, assume mid-point

    return metrics.replace([np.inf, -np.inf], np.nan)

def calculate_return_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates various return-based metrics."""
    metrics = pd.DataFrame(index=data.index)
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in data.columns and pd.api.types.is_numeric_dtype(data[col]) for col in required_cols):
        return metrics
        
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    close_prev = close.shift(1)

    # Daily Log Returns
    metrics['log_return'] = np.log(close / close_prev)

    # Absolute Daily Returns
    metrics['abs_return'] = (close / close_prev - 1).abs()

    # Squared Daily Returns
    metrics['sq_return'] = (close / close_prev - 1)**2

    # Range-Based Returns (Normalized by open)
    metrics['range_return_open'] = (high - low) / open_

    return metrics.replace([np.inf, -np.inf], np.nan)

def calculate_volatility_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates various intraday and range-based volatility metrics."""
    metrics = pd.DataFrame(index=data.index)
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in data.columns and pd.api.types.is_numeric_dtype(data[col]) for col in required_cols):
        return metrics
        
    open_ = data['open']
    high = data['high']
    low = data['low']
    close = data['close']
    close_prev = close.shift(1)

    # Daily Price Range (Normalized by close)
    metrics['price_range_norm'] = (high - low) / close

    # True Range
    hl = high - low
    hc = (high - close_prev).abs()
    lc = (low - close_prev).abs()
    metrics['true_range'] = np.maximum(hl, np.maximum(hc, lc))

    # Garman-Klass Volatility (requires log)
    log_hl = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(close / open_).replace([np.inf, -np.inf], np.nan)
    metrics['vol_garman_klass'] = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    # Set negative results to zero (variance can't be negative)
    metrics['vol_garman_klass'] = metrics['vol_garman_klass'].clip(lower=0)

    # Rogers-Satchell Volatility (requires log)
    log_hc = np.log(high / close).replace([np.inf, -np.inf], np.nan)
    log_ho = np.log(high / open_).replace([np.inf, -np.inf], np.nan)
    log_lc = np.log(low / close).replace([np.inf, -np.inf], np.nan)
    log_lo = np.log(low / open_).replace([np.inf, -np.inf], np.nan)
    metrics['vol_rogers_satchell'] = log_hc * log_ho + log_lc * log_lo
    # Ensure non-negative
    metrics['vol_rogers_satchell'] = metrics['vol_rogers_satchell'].clip(lower=0)

    return metrics.replace([np.inf, -np.inf], np.nan)

def calculate_all_derived_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates all derived metrics and returns them in a single DataFrame."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame()
        
    volume_metrics = calculate_volume_metrics(data)
    pattern_metrics = calculate_price_pattern_metrics(data)
    return_metrics = calculate_return_metrics(data)
    volatility_metrics = calculate_volatility_metrics(data)
    
    # Combine all metrics
    all_metrics = pd.concat([volume_metrics, pattern_metrics, return_metrics, volatility_metrics], axis=1)
    
    return all_metrics 