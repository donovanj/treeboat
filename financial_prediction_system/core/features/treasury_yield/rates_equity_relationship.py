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
    
    # Find the closing price column in the main data
    close_cols = [col for col in data.columns if 'close' in col.lower()]
    if not close_cols:
        return builder  # Skip if no close data available
    price_col = close_cols[0]
    
    # Clean up yields data
    yields = yields_data.copy().ffill().bfill()
    
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
                                X = rate_window[valid_indices].reshape(-1, 1)
                                X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
                                y = sec_window[valid_indices]
                                
                                try:
                                    beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
                                    rolling_rate_betas.append(beta)
                                except:
                                    rolling_rate_betas.append(np.nan)
                            else:
                                rolling_rate_betas.append(np.nan)
                        else:
                            rolling_rate_betas.append(np.nan)
                    
                    builder.features[f'rate_beta_{yield_col}_{window}d'] = pd.Series(
                        rolling_rate_betas, index=security_returns.index
                    )
    
    # 3. Sector-specific rate sensitivity
    # Different sectors have different sensitivity to rate changes
    if 'spx_prices' in index_data and len(index_data['spx_prices']) > max(sector_sensitivity_windows):
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
                    
                    if sector_key and sector_key in index_data:
                        sector_df = index_data[sector_key].copy().reindex(data.index, method='ffill')
                        sector_close_cols = [col for col in sector_df.columns if 'close' in col.lower()]
                        
                        if sector_close_cols:
                            sector_price_col = sector_close_cols[0]
                            sector_returns = sector_df[sector_price_col].pct_change()
                            
                            # Calculate sector rate sensitivity
                            for window in sector_sensitivity_windows:
                                if len(sector_returns) >= window:
                                    # Calculate correlation between sector returns and rate changes
                                    sector_rate_corr = sector_returns.rolling(window=window).corr(rate_changes)
                                    builder.features[f'{sector_name}_rate_corr_{window}d'] = sector_rate_corr
                                    
                                    # See if security is more or less sensitive than its sector
                                    security_rate_corr = security_returns.rolling(window=window).corr(rate_changes)
                                    builder.features[f'rel_to_{sector_name}_rate_sens_{window}d'] = (
                                        security_rate_corr - sector_rate_corr
                                    )
    
    # 4. Rate regime features
    # Create regime indicators based on rate levels and changes
    if 'yr10' in yields.columns:
        # Yield level regimes
        yr10_thresholds = [1.5, 2.5, 3.5, 4.5]
        yield_levels = pd.cut(
            yields['yr10'], 
            bins=[0] + yr10_thresholds + [float('inf')],
            labels=[1, 2, 3, 4, 5]  # 1=very low, 5=very high
        ).astype(float)
        
        builder.features['yield_regime'] = yield_levels
        
        # Yield change regimes (falling, stable, rising, rapidly rising)
        for window in [20, 60]:
            yield_change = yields['yr10'].diff(window)
            change_thresholds = [-0.5, -0.1, 0.1, 0.5]
            
            yield_change_regime = pd.cut(
                yield_change,
                bins=[-float('inf')] + change_thresholds + [float('inf')],
                labels=[-2, -1, 0, 1, 2]  # -2=rapidly falling, 0=stable, 2=rapidly rising
            ).astype(float)
            
            builder.features[f'yield_change_regime_{window}d'] = yield_change_regime
    
    # 5. Comparative performance in different rate regimes
    # Calculate relative performance in rising/falling rate environments
    if 'yr10' in yields.columns:
        rate_col = 'yr10'
        
        for window in [20, 60]:
            # Define rate environment
            yield_change = yields[rate_col].diff(window)
            rising_rates = (yield_change > 0).astype(int)
            falling_rates = (yield_change < 0).astype(int)
            
            # Calculate security performance in rising/falling rate periods
            security_perf = data[price_col].pct_change(window)
            
            # SPX performance for comparison
            if 'spx_prices' in index_data:
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
                    
                    # Calculate rolling mean of non-NaN values
                    for avg_window in [60, 120]:
                        rising_avg = rising_series.rolling(window=avg_window, min_periods=avg_window//4).mean()
                        falling_avg = falling_series.rolling(window=avg_window, min_periods=avg_window//4).mean()
                        
                        builder.features[f'rising_rates_rel_perf_{window}d_{avg_window}d_avg'] = rising_avg
                        builder.features[f'falling_rates_rel_perf_{window}d_{avg_window}d_avg'] = falling_avg
    
    # 6. Equity risk premium approximation
    # ERP = Earnings Yield - 10Y Treasury Yield
    # We can approximate earnings yield with 1/PE ratio or use a market index earnings yield
    # Here we'll use a simple approximation based on SPX
    if 'yr10' in yields.columns and 'spx_prices' in index_data:
        spx_df = index_data['spx_prices'].copy().reindex(data.index, method='ffill')
        spx_close_cols = [col for col in spx_df.columns if 'close' in col.lower()]
        
        if spx_close_cols:
            # Use average historical earnings yield of ~4-5% as a rough proxy
            # In a real system, you would use actual earnings data
            approx_earnings_yield = 0.045  # 4.5%
            builder.features['erp_approx'] = approx_earnings_yield - (yields['yr10'] / 100)
    
    return builder