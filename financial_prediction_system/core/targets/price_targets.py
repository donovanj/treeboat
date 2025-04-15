"""
Price targets module
"""
import pandas as pd
import numpy as np
from financial_prediction_system.core.targets.target_builder import TargetBuilder

def add_price_targets(builder: TargetBuilder,
                      price_col: str = 'close',
                      periods: list = [1, 5, 10, 20],
                      return_type: str = 'log',
                      bins: int = None,
                      lookback_normalization: bool = False,
                      lookback_window: int = 252,
                      threshold_based: bool = False,
                      thresholds: list = None) -> TargetBuilder:
    """
    Add price-based return targets
    
    Parameters
    ----------
    builder : TargetBuilder
        The target builder instance
    price_col : str, default='close'
        Column name for the price data
    periods : list, default=[1, 5, 10, 20]
        List of forward periods to calculate returns for
    return_type : str, default='log'
        Type of return to calculate: 'simple', 'log', 'direction', 'categorical', or 'zscore'
    bins : int, optional
        Number of bins for categorical returns, only used if return_type='categorical'
    lookback_normalization : bool, default=False
        If True, normalize returns based on historical volatility
    lookback_window : int, default=252
        Number of days to use for lookback normalization
    threshold_based : bool, default=False
        If True, create threshold-based classification instead of quantile-based
    thresholds : list, optional
        Custom thresholds for classification, e.g. [-0.05, -0.02, 0.02, 0.05]
        
    Returns
    -------
    TargetBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if price_col not in data.columns:
        return builder  # Skip if price column not found
        
    prices = data[price_col]
    
    for period in periods:
        # Calculate future price and base returns
        future_price = prices.shift(-period)
        
        if return_type == 'simple':
            # Calculate simple return: (future_price / current_price) - 1
            returns = (future_price / prices) - 1
            returns_name = f'return_{period}d'
            
        elif return_type == 'log':
            # Calculate log return: log(future_price / current_price)
            returns = np.log(future_price / prices)
            returns_name = f'log_return_{period}d'
            
        elif return_type == 'compound':
            # Calculate compound return over the period
            # (different from simple return when there are multiple steps)
            daily_returns = prices.pct_change()
            returns = daily_returns.shift(-period).rolling(period).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )
            returns_name = f'compound_return_{period}d'
            
        elif return_type == 'zscore':
            # Calculate returns normalized by historical standard deviation
            simple_returns = (future_price / prices) - 1
            
            # Calculate rolling standard deviation
            rolling_std = prices.pct_change().rolling(lookback_window).std()
            
            # Normalize returns by standard deviation
            returns = simple_returns / (rolling_std * np.sqrt(period))
            returns_name = f'zscore_return_{period}d'
            
        elif return_type == 'direction':
            # Calculate price direction: 1 if price increases, 0 if it decreases
            returns = (future_price > prices).astype(int)
            returns_name = f'direction_{period}d'
            
            # Add the basic direction target
            builder.targets[returns_name] = returns
            
            # Add additional directional strength target (optional)
            simple_returns = (future_price / prices) - 1
            
            # Create a 3-class target: down significantly, flat, up significantly
            if threshold_based and thresholds:
                # Use custom thresholds
                cuts = [-np.inf] + thresholds + [np.inf]
                labels = list(range(len(cuts) - 1))
            else:
                # Default thresholds
                std_dev = simple_returns.std()
                cuts = [-np.inf, -std_dev * 0.5, std_dev * 0.5, np.inf]
                labels = [0, 1, 2]  # 0: down, 1: flat, 2: up
                
            directional_strength = pd.cut(simple_returns, bins=cuts, labels=labels)
            builder.targets[f'direction_strength_{period}d'] = directional_strength
            
            continue  # Skip the standard target addition for direction type
            
        elif return_type == 'categorical':
            # Calculate returns for categorization
            simple_returns = (future_price / prices) - 1
            
            if bins:
                # Create categorical target using quantile bins
                if threshold_based and thresholds:
                    # Use custom thresholds
                    cuts = [-np.inf] + thresholds + [np.inf]
                    labels = list(range(len(cuts) - 1))
                    categorized = pd.cut(simple_returns, bins=cuts, labels=labels)
                    returns_name = f'return_cat_{period}d_custom'
                else:
                    # Quantile-based bins
                    labels = list(range(bins))
                    categorized = pd.qcut(simple_returns, bins, labels=labels, duplicates='drop')
                    returns_name = f'return_cat_{period}d_{bins}bins'
            else:
                # Default categorization: down big, down, flat, up, up big
                if threshold_based and thresholds:
                    cuts = [-np.inf] + thresholds + [np.inf]
                    labels = list(range(len(cuts) - 1))
                else:
                    # Default thresholds
                    cuts = [-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf]
                    labels = [0, 1, 2, 3, 4]  # 0: down big, 1: down, 2: flat, 3: up, 4: up big
                    
                categorized = pd.cut(simple_returns, bins=cuts, labels=labels)
                returns_name = f'return_cat_{period}d'
                
            returns = categorized
        else:
            raise ValueError(f"Invalid return_type: {return_type}. Choose from 'simple', 'log', 'compound', 'zscore', 'direction', or 'categorical'")
        
        # Apply lookback normalization if requested (except for categorical and zscore)
        if lookback_normalization and return_type not in ['categorical', 'zscore', 'direction']:
            # Normalize returns by historical volatility
            rolling_std = prices.pct_change().rolling(lookback_window).std()
            returns = returns / (rolling_std * np.sqrt(period))
            returns_name = f'{returns_name}_normalized'
        
        # Add the target
        builder.targets[returns_name] = returns
        
        # Add percentile rank target for continuous returns
        if return_type in ['simple', 'log', 'compound', 'zscore']:
            # Calculate rank over a lookback window
            ranks = returns.rolling(lookback_window, min_periods=int(lookback_window/2)).rank(pct=True)
            builder.targets[f'{returns_name}_rank'] = ranks
    
    return builder


def add_drawdown_targets(builder: TargetBuilder,
                         price_col: str = 'close',
                         periods: list = [5, 10, 20],
                         classification: bool = False) -> TargetBuilder:
    """
    Add maximum drawdown targets for future periods
    
    Parameters
    ----------
    builder : TargetBuilder
        The target builder instance
    price_col : str, default='close'
        Column name for the price data
    periods : list, default=[5, 10, 20]
        List of forward periods to calculate drawdowns for
    classification : bool, default=False
        If True, create classification targets based on drawdown severity
        
    Returns
    -------
    TargetBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if price_col not in data.columns:
        return builder  # Skip if price column not found
        
    prices = data[price_col]
    
    for period in periods:
        # Create a DataFrame to hold future prices for each day
        future_prices = pd.DataFrame(index=prices.index)
        
        # Populate with future prices
        for i in range(1, period + 1):
            future_prices[f'day_{i}'] = prices.shift(-i)
            
        # Calculate maximum drawdown for each future period
        def calculate_max_drawdown(row):
            prices_window = row.dropna().values
            if len(prices_window) == 0:
                return np.nan
                
            peak = prices_window[0]  # Start from the first value
            max_drawdown = 0
            
            for price in prices_window:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
                
            return max_drawdown
            
        max_drawdowns = future_prices.apply(calculate_max_drawdown, axis=1)
        builder.targets[f'max_drawdown_{period}d'] = max_drawdowns
        
        # Add classification target if requested
        if classification:
            # Create categories based on drawdown severity
            cuts = [-0.001, 0.01, 0.03, 0.05, 0.10, np.inf]  # Adjust these thresholds as needed
            labels = [0, 1, 2, 3, 4]  # 0: minimal, 1: small, 2: medium, 3: large, 4: severe
            drawdown_class = pd.cut(max_drawdowns, bins=cuts, labels=labels)
            builder.targets[f'max_drawdown_{period}d_class'] = drawdown_class
    
    return builder