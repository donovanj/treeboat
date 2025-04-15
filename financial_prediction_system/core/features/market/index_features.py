"""
Market index relationship features module
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_market_index_features(
    builder: FeatureBuilder, 
    index_data: Dict[str, pd.DataFrame],
    correlation_windows: List[int] = [5, 10, 20, 60],
    beta_windows: List[int] = [20, 60, 120, 252],
    relative_strength_windows: List[int] = [5, 10, 20, 60]
) -> FeatureBuilder:
    """
    Add features capturing the relationship between the security and major market indexes
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    index_data : Dict[str, pd.DataFrame]
        Dictionary of index data frames, with keys like 'spx_prices', 'ndx_prices', etc.
        Each DataFrame should have a datetime index matching the builder's data
    correlation_windows : List[int], default=[5, 10, 20, 60]
        Window sizes for calculating rolling correlations
    beta_windows : List[int], default=[20, 60, 120, 252]
        Window sizes for calculating rolling betas
    relative_strength_windows : List[int], default=[5, 10, 20, 60]
        Window sizes for calculating relative strength
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    # Find the closing price column in the main data
    close_cols = [col for col in data.columns if 'close' in col.lower()]
    if not close_cols:
        return builder  # Skip if no close data available
    price_col = close_cols[0]
    
    # Calculate returns for the main security
    returns = data[price_col].pct_change()
    log_returns = np.log(data[price_col] / data[price_col].shift(1))
    
    # Process each index
    for index_name, index_df in index_data.items():
        # Skip if the index data doesn't have enough data points
        if len(index_df) < 20:
            continue
            
        # Find the closing price column in the index data
        index_close_cols = [col for col in index_df.columns if 'close' in col.lower()]
        if not index_close_cols:
            continue
        index_price_col = index_close_cols[0]
        
        # Ensure index data is aligned with main data
        index_df = index_df.reindex(data.index, method='ffill')
        
        # Calculate returns for the index
        index_returns = index_df[index_price_col].pct_change()
        index_log_returns = np.log(index_df[index_price_col] / index_df[index_price_col].shift(1))
        
        # Relative price levels
        index_prefix = index_name.split('_')[0].lower()
        builder.features[f'rel_to_{index_prefix}'] = data[price_col] / index_df[index_price_col]
        
        # Normalize both series to start at 100 for relative performance tracking
        norm_base = 100
        for window in relative_strength_windows:
            if len(returns) >= window:
                # Get relative performance over window periods
                sec_perf = (1 + returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1, raw=True))
                idx_perf = (1 + index_returns.rolling(window=window).apply(lambda x: (1 + x).prod() - 1, raw=True))
                builder.features[f'rel_strength_{index_prefix}_{window}'] = sec_perf / idx_perf
        
        # Rolling correlation
        for window in correlation_windows:
            if len(returns) >= window:
                builder.features[f'corr_{index_prefix}_{window}'] = returns.rolling(
                    window=window).corr(index_returns)
                
        # Rolling beta (regression against index)
        for window in beta_windows:
            if len(log_returns) >= window:
                # Using numpy's polyfit to calculate beta
                def rolling_beta(y, x, window):
                    beta = np.full(len(y), np.nan)
                    for i in range(window-1, len(y)):
                        if i >= window-1:
                            xy = np.vstack([x[i-window+1:i+1], np.ones(window)]).T
                            try:
                                beta[i] = np.linalg.lstsq(xy, y[i-window+1:i+1], rcond=None)[0][0]
                            except:
                                pass
                    return beta
                
                # Calculate beta using the function
                if len(log_returns) > window:  # Ensure we have enough data
                    betas = rolling_beta(log_returns.values, index_log_returns.values, window)
                    builder.features[f'beta_{index_prefix}_{window}'] = pd.Series(betas, index=log_returns.index)
        
        # VIX-specific features (if available)
        if 'vix' in index_name.lower():
            # VIX percentile over lookback periods
            for window in [30, 60, 90, 252]:
                if len(index_df) >= window:
                    vix_pct_rank = index_df[index_price_col].rolling(window=window).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
                    )
                    builder.features[f'vix_percentile_{window}'] = vix_pct_rank
            
            # VIX rate of change
            for period in [1, 3, 5, 10]:
                builder.features[f'vix_roc_{period}'] = index_df[index_price_col].pct_change(periods=period)
                
            # VIX moving averages crossovers
            vix_ma5 = index_df[index_price_col].rolling(window=5).mean()
            vix_ma20 = index_df[index_price_col].rolling(window=20).mean()
            builder.features['vix_ma5_gt_ma20'] = (vix_ma5 > vix_ma20).astype(int)
        
        # Detect if security is outperforming/underperforming the index
        for window in [5, 10, 20]:
            sec_return = returns.rolling(window=window).sum()
            idx_return = index_returns.rolling(window=window).sum()
            builder.features[f'outperforming_{index_prefix}_{window}'] = (sec_return > idx_return).astype(int)
        
        # Index specific features for major indexes
        if any(idx in index_name.lower() for idx in ['spx', 'ndx', 'dji']):
            # Distance from index moving averages (in %)
            for ma_period in [50, 100, 200]:
                ma = index_df[index_price_col].rolling(window=ma_period).mean()
                builder.features[f'{index_prefix}_dist_ma{ma_period}'] = (index_df[index_price_col] - ma) / ma
            
            # Breadth indicators (if available)
            # These would typically come from external data sources but can be approximated
            # with trend measures of the index itself
            for window in [5, 10, 20]:
                # Trend strength
                builder.features[f'{index_prefix}_trend_{window}'] = index_returns.rolling(window=window).sum()
                
                # Momentum
                builder.features[f'{index_prefix}_mom_{window}'] = index_df[index_price_col].pct_change(periods=window)
    
    return builder