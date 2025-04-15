"""
Rates-equity relationship features module
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_rates_equity_features(
    builder: FeatureBuilder,
    yields_data: pd.DataFrame,
    index_data: Dict[str, pd.DataFrame],
    correlation_windows: List[int] = [20, 60, 120],
    sector_sensitivity_windows: List[int] = [60, 120, 252]
) -> FeatureBuilder:
    """
    Add features capturing the relationship between interest rates and equities
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    yields_data : pd.DataFrame
        DataFrame containing Treasury yield data with date as index
    index_data : Dict[str, pd.DataFrame]
        Dictionary of index data frames, with keys like 'spx_prices', 'ndx_prices', etc.
    correlation_windows : List[int], default=[20, 60, 120]
        Window sizes for calculating rolling correlations
    sector_sensitivity_windows : List[int], default=[60, 120, 252]
        Window sizes for calculating sector rate sensitivity
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    # Check if yields_data is empty
    if yields_data is None or yields_data.empty:
        return builder
    
    # Find the closing price column in the main data
    close_cols = [col for col in data.columns if 'close' in col.lower()]
    if not close_cols:
        return builder  # Skip if no close data available
    price_col = close_cols[0]
    
    # Clean up yields data
    yields = yields_data.copy()
    # Replace any potential inf or NaN values
    yields = yields.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
    # Align with the main data's index
    yields = yields.reindex(data.index, method='ffill')
    
    # Key yield columns to focus on
    focus_yields = ['mo3', 'yr2', 'yr10', 'yr30']
    available_yields = [col for col in focus_yields if col in yields.columns]
    
    if not available_yields:
        return builder  # No valid yield columns
    
    # Key spread to focus on (10yr-2yr)
    if 'yr10' in yields.columns and 'yr2' in yields.columns:
        yields['spread_10y2y'] = yields['yr10'] - yields['yr2']
        available_yields.append('spread_10y2y')
    
    # Calculate security returns
    security_returns = data[price_col].pct_change()
    
    # 1. Correlation between security returns and yield changes
    for yield_col in available_yields:
        if yield_col in yields.columns:
            yield_changes = yields[yield_col].diff()
            
            for window in correlation_windows:
                if len(security_returns) >= window and len(yield_changes) >= window:
                    builder.features[f'corr_{yield_col}_{window}d'] = security_returns.rolling(
                        window=window).corr(yield_changes)
    
    # 2. Rate sensitivity (beta to rate changes)
    # Similar to market beta, but using rate changes instead of market returns
    for yield_col in available_yields:
        if yield_col in yields.columns:
            yield_changes = yields[yield_col].diff()
            
            for window in correlation_windows:
                if len(security_returns) >= window and len(yield_changes) >= window:
                    # Calculate rolling beta to rate changes using regression
                    rolling_rate_betas = []
                    
                    for i in range(window - 1, len(security_returns)):
                        if i >= window - 1:
                            sec_window = security_returns.iloc[i-window+1:i+1].values
                            rate_window = yield_changes.iloc[i-window+1:i+1].values
                            
                            # Filter out NaN values
                            valid_indices = ~(np.isnan(sec_window) | np.isnan(rate_window))
                            if sum(valid_indices) > window // 2:
                                try:
                                    X = rate_window[valid_indices].reshape(-1, 1)
                                    X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
                                    y = sec_window[valid_indices]
                                    
                                    # Check for empty arrays after filtering
                                    if X.size > 0 and y.size > 0:
                                        beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
                                        rolling_rate_betas.append(beta)
                                    else:
                                        rolling_rate_betas.append(np.nan)
                                except Exception:
                                    rolling_rate_betas.append(np.nan)
                            else:
                                rolling_rate_betas.append(np.nan)
                        else:
                            rolling_rate_betas.append(np.nan)
                    
                    if rolling_rate_betas:  # Check if list is not empty
                        builder.features[f'rate_beta_{yield_col}_{window}d'] = pd.Series(
                            rolling_rate_betas, index=security_returns.index
                        )
    
    # 3. Sector-specific rate sensitivity
    # Different sectors have different sensitivity to rate changes
    if 'spx_prices' in index_data and index_data['spx_prices'] is not None and not index_data['spx_prices'].empty:
        if len(index_data['spx_prices']) > 0:  # Check there are rows
            max_window = max(sector_sensitivity_windows) if sector_sensitivity_windows else 0
            if len(index_data['spx_prices']) > max_window:
                spx_df = index_data['spx_prices'].copy().reindex(data.index, method='ffill')
                spx_close_cols = [col for col in spx_df.columns if 'close' in col.lower()]
                
                if spx_close_cols:
                    spx_price_col = spx_close_cols[0]
                    spx_returns = spx_df[spx_price_col].pct_change()
                    
                    # Sector indexes to analyze
                    sector_indexes = {
                        'ndx': 'technology',
                        'sox': 'semiconductors',
                        'osx': 'energy',
                        'rut': 'small_cap',
                        'dji': 'industrial'
                    }
                    
                    # Key rates for sector analysis
                    if 'yr10' in yields.columns:
                        rate_col = 'yr10'
                        rate_changes = yields[rate_col].diff()
                        
                        for sector_prefix, sector_name in sector_indexes.items():
                            sector_key = next((k for k in index_data.keys() if k.startswith(sector_prefix)), None)
                            
                            if sector_key and sector_key in index_data and index_data[sector_key] is not None:
                                if not index_data[sector_key].empty:
                                    sector_df = index_data[sector_key].copy().reindex(data.index, method='ffill')
                                    sector_close_cols = [col for col in sector_df.columns if 'close' in col.lower()]
                                    
                                    if sector_close_cols:
                                        sector_price_col = sector_close_cols[0]
                                        sector_returns = sector_df[sector_price_col].pct_change()
                                        
                                        # Calculate sector rate sensitivity
                                        for window in sector_sensitivity_windows:
                                            if len(sector_returns) >= window:
                                                try:
                                                    # Calculate correlation between sector returns and rate changes
                                                    sector_rate_corr = sector_returns.rolling(window=window).corr(rate_changes)
                                                    builder.features[f'{sector_name}_rate_corr_{window}d'] = sector_rate_corr
                                                    
                                                    # See if security is more or less sensitive than its sector
                                                    security_rate_corr = security_returns.rolling(window=window).corr(rate_changes)
                                                    builder.features[f'rel_to_{sector_name}_rate_sens_{window}d'] = (
                                                        security_rate_corr - sector_rate_corr
                                                    )
                                                except Exception:
                                                    # Skip this correlation if there's an error
                                                    pass
    
    # 4. Rate regime features
    # Create regime indicators based on rate levels and changes
    if 'yr10' in yields.columns:
        # Filter out any NaN values in yr10 yields
        valid_yields = yields['yr10'].dropna()
        if not valid_yields.empty:
            # Yield level regimes
            yr10_thresholds = [1.5, 2.5, 3.5, 4.5]
            try:
                yield_levels = pd.cut(
                    valid_yields, 
                    bins=[0] + yr10_thresholds + [float('inf')],
                    labels=[1, 2, 3, 4, 5]  # 1=very low, 5=very high
                ).astype(float)
                
                # Reindex to match original data
                yield_levels = yield_levels.reindex(yields.index)
                builder.features['yield_regime'] = yield_levels
            except Exception:
                # Skip this feature if categorization fails
                pass
        
        # Yield change regimes (falling, stable, rising, rapidly rising)
        for window in [20, 60]:
            if len(yields) > window:
                try:
                    yield_change = yields['yr10'].diff(window)
                    # Filter out any NaN or inf values
                    valid_changes = yield_change.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not valid_changes.empty:
                        change_thresholds = [-0.5, -0.1, 0.1, 0.5]
                        
                        yield_change_regime = pd.cut(
                            valid_changes,
                            bins=[-float('inf')] + change_thresholds + [float('inf')],
                            labels=[-2, -1, 0, 1, 2]  # -2=rapidly falling, 0=stable, 2=rapidly rising
                        ).astype(float)
                        
                        # Reindex to match original data
                        yield_change_regime = yield_change_regime.reindex(yields.index)
                        builder.features[f'yield_change_regime_{window}d'] = yield_change_regime
                except Exception:
                    # Skip this feature if categorization fails
                    pass
    
    # 5. Comparative performance in different rate regimes
    # Calculate relative performance in rising/falling rate environments
    if 'yr10' in yields.columns:
        rate_col = 'yr10'
        
        for window in [20, 60]:
            if len(yields) > window:
                try:
                    # Define rate environment
                    yield_change = yields[rate_col].diff(window)
                    # Handle NaN values
                    yield_change = yield_change.replace([np.inf, -np.inf], np.nan)
                    rising_rates = (yield_change > 0).astype(int)
                    falling_rates = (yield_change < 0).astype(int)
                    
                    # Calculate security performance in rising/falling rate periods
                    security_perf = data[price_col].pct_change(window)
                    
                    # SPX performance for comparison
                    if 'spx_prices' in index_data and index_data['spx_prices'] is not None and not index_data['spx_prices'].empty:
                        spx_df = index_data['spx_prices'].copy().reindex(data.index, method='ffill')
                        spx_close_cols = [col for col in spx_df.columns if 'close' in col.lower()]
                        
                        if spx_close_cols:
                            spx_price_col = spx_close_cols[0]
                            spx_perf = spx_df[spx_price_col].pct_change(window)
                            
                            # Relative performance in different rate environments
                            rising_rates_rel_perf = np.where(
                                rising_rates == 1,
                                security_perf - spx_perf,
                                np.nan
                            )
                            
                            falling_rates_rel_perf = np.where(
                                falling_rates == 1,
                                security_perf - spx_perf,
                                np.nan
                            )
                            
                            # Calculate rolling average of relative performance in each regime
                            # First, create Series of the performance values
                            rising_series = pd.Series(rising_rates_rel_perf, index=data.index)
                            falling_series = pd.Series(falling_rates_rel_perf, index=data.index)
                            
                            # Replace any inf values with NaN
                            rising_series = rising_series.replace([np.inf, -np.inf], np.nan)
                            falling_series = falling_series.replace([np.inf, -np.inf], np.nan)
                            
                            # Calculate rolling mean of non-NaN values
                            for avg_window in [60, 120]:
                                if len(rising_series) > avg_window:
                                    rising_avg = rising_series.rolling(window=avg_window, min_periods=avg_window//4).mean()
                                    falling_avg = falling_series.rolling(window=avg_window, min_periods=avg_window//4).mean()
                                    
                                    builder.features[f'rising_rates_rel_perf_{window}d_{avg_window}d_avg'] = rising_avg
                                    builder.features[f'falling_rates_rel_perf_{window}d_{avg_window}d_avg'] = falling_avg
                except Exception:
                    # Skip this feature if there's an error
                    pass
    
    # 6. Equity risk premium approximation
    # ERP = Earnings Yield - 10Y Treasury Yield
    # We can approximate earnings yield with 1/PE ratio or use a market index earnings yield
    # Here we'll use a simple approximation based on SPX
    if 'yr10' in yields.columns and 'spx_prices' in index_data:
        if index_data['spx_prices'] is not None and not index_data['spx_prices'].empty:
            spx_df = index_data['spx_prices'].copy().reindex(data.index, method='ffill')
            spx_close_cols = [col for col in spx_df.columns if 'close' in col.lower()]
            
            if spx_close_cols:
                # Use average historical earnings yield of ~4-5% as a rough proxy
                # In a real system, you would use actual earnings data
                approx_earnings_yield = 0.045  # 4.5%
                
                # Handle cases where yr10 might be zero
                yr10_values = yields['yr10'].replace(0, np.nan) / 100
                builder.features['erp_approx'] = approx_earnings_yield - yr10_values
    
    return builder