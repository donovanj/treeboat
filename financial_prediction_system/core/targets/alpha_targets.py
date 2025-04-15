"""
Alpha targets module
"""
import pandas as pd
import numpy as np
from financial_prediction_system.core.targets.target_builder import TargetBuilder

def add_alpha_targets(builder: TargetBuilder,
                      price_col: str = 'close',
                      benchmark_col: str = 'benchmark',
                      periods: list = [1, 5, 10, 20],
                      risk_adjusted: bool = False,
                      alpha_type: str = 'standard',
                      annualize: bool = False,
                      classification: bool = False) -> TargetBuilder:
    """
    Add alpha-based targets (excess return over benchmark)
    
    Parameters
    ----------
    builder : TargetBuilder
        The target builder instance
    price_col : str, default='close'
        Column name for the price data
    benchmark_col : str, default='benchmark'
        Column name for the benchmark price data
    periods : list, default=[1, 5, 10, 20]
        List of forward periods to calculate alpha for
    risk_adjusted : bool, default=False
        If True, calculate risk-adjusted alpha (alpha/volatility)
    alpha_type : str, default='standard'
        Type of alpha to calculate: 'standard', 'jensen', or 'information_ratio'
    annualize : bool, default=False
        If True, annualize the alpha values
    classification : bool, default=False
        If True, create classification targets based on alpha quartiles
        
    Returns
    -------
    TargetBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if price_col not in data.columns:
        return builder  # Skip if price column not found
    if benchmark_col not in data.columns:
        return builder  # Skip if benchmark column not found
        
    # Calculate returns
    asset_returns = data[price_col].pct_change()
    benchmark_returns = data[benchmark_col].pct_change()
    
    # Create targets for each period
    for period in periods:
        # Calculate forward returns
        forward_asset_returns = asset_returns.shift(-period).rolling(period).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        forward_benchmark_returns = benchmark_returns.shift(-period).rolling(period).apply(
            lambda x: (1 + x).prod() - 1, raw=True
        )
        
        if alpha_type == 'standard':
            # Calculate alpha (excess return)
            alpha = forward_asset_returns - forward_benchmark_returns
            target_name_base = 'alpha'
            
        elif alpha_type == 'jensen':
            # Calculate Jensen's alpha
            # Use a 60-period lookback to calculate beta
            lookback = min(60, len(data))
            
            # Calculate beta using rolling regression
            def calculate_beta(asset_ret, bench_ret):
                if len(asset_ret) < 5 or len(bench_ret) < 5:  # Need at least 5 points for regression
                    return np.nan
                covariance = np.cov(asset_ret, bench_ret)[0, 1]
                benchmark_variance = np.var(bench_ret)
                if benchmark_variance == 0:
                    return np.nan
                return covariance / benchmark_variance
            
            # Calculate rolling beta
            rolling_window = pd.DataFrame({
                'asset': asset_returns,
                'benchmark': benchmark_returns
            }).rolling(lookback)
            
            betas = rolling_window.apply(
                lambda x: calculate_beta(x['asset'], x['benchmark']), raw=False
            )
            
            # Calculate Jensen's alpha
            alpha = forward_asset_returns - (betas * forward_benchmark_returns)
            target_name_base = 'jensen_alpha'
            
        elif alpha_type == 'information_ratio':
            # Calculate Information Ratio (excess return / tracking error)
            # Tracking error is the standard deviation of the difference between returns
            tracking_error = (asset_returns - benchmark_returns).rolling(period).std() * np.sqrt(period)
            alpha = (forward_asset_returns - forward_benchmark_returns) / tracking_error
            target_name_base = 'information_ratio'
            
        else:
            raise ValueError(f"Invalid alpha_type: {alpha_type}. Choose from 'standard', 'jensen', or 'information_ratio'")
        
        # Apply risk adjustment if requested
        if risk_adjusted and alpha_type != 'information_ratio':  # IR is already risk-adjusted
            # Use rolling standard deviation of daily returns for risk adjustment
            rolling_vol = asset_returns.rolling(period).std() * np.sqrt(period)
            alpha = alpha / rolling_vol
            target_name_prefix = f'{target_name_base}_risk_adj'
        else:
            target_name_prefix = target_name_base
            
        # Annualize if requested
        if annualize:
            # Annualize alpha values (assuming 252 trading days)
            annualization_factor = 252 / period
            alpha = alpha * annualization_factor
            target_name = f'{target_name_prefix}_annual_{period}d'
        else:
            target_name = f'{target_name_prefix}_{period}d'
            
        # Add the primary alpha target
        builder.targets[target_name] = alpha
        
        # Add classification target if requested
        if classification:
            # Create quartile-based classification
            alpha_class = pd.qcut(alpha, 4, labels=[0, 1, 2, 3], duplicates='drop')
            builder.targets[f'{target_name}_class'] = alpha_class
    
    return builder