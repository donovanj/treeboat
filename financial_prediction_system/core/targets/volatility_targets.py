"""
Volatility targets module
"""
import pandas as pd
import numpy as np
from financial_prediction_system.core.targets.target_builder import TargetBuilder

def add_volatility_targets(builder: TargetBuilder,
                          price_col: str = 'close',
                          periods: list = [5, 10, 20],
                          volatility_type: str = 'realized',
                          classification: bool = False,
                          high_col: str = 'high',
                          low_col: str = 'low',
                          open_col: str = 'open',
                          annualize: bool = True,
                          relative_vol: bool = False,
                          lookback_window: int = 60,
                          vol_of_vol: bool = False,
                          skew_kurtosis: bool = False,
                          range_targets: bool = False) -> TargetBuilder:
    """
    Add volatility-based targets
    
    Parameters
    ----------
    builder : TargetBuilder
        The target builder instance
    price_col : str, default='close'
        Column name for the price data
    periods : list, default=[5, 10, 20]
        List of forward periods to calculate volatility for
    volatility_type : str, default='realized'
        Type of volatility to calculate: 'realized', 'parkinson', 'garman_klass', 
        'rogers_satchell', 'yang_zhang', or 'garch'
    classification : bool, default=False
        If True, create classification targets based on volatility quartiles
    high_col : str, optional
        Column name for high price data, required for some volatility types
    low_col : str, optional
        Column name for low price data, required for some volatility types
    open_col : str, optional
        Column name for open price data, required for 'yang_zhang' volatility
    annualize : bool, default=True
        If True, annualize volatility values (multiply by sqrt(252))
    relative_vol : bool, default=False
        If True, calculate volatility relative to historical average
    lookback_window : int, default=60
        Number of days for lookback when calculating relative volatility
    vol_of_vol : bool, default=False
        If True, calculate the volatility of volatility
    skew_kurtosis : bool, default=False
        If True, add skewness and kurtosis targets for returns distribution
    range_targets : bool, default=False
        If True, create targets for min-max price range forecasts
        
    Returns
    -------
    TargetBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if price_col not in data.columns:
        return builder  # Skip if price column not found
    
    # Calculate returns
    returns = data[price_col].pct_change()
    
    # Calculate annualization factor
    annual_factor = np.sqrt(252) if annualize else 1.0
    
    # For relative volatility, calculate historical volatility
    if relative_vol:
        historical_vol = returns.rolling(lookback_window).std() * annual_factor
    
    for period in periods:
        if volatility_type == 'realized':
            # Calculate future realized volatility (standard deviation of returns)
            future_vol = returns.shift(-period).rolling(period).apply(
                lambda x: np.std(x) * annual_factor, raw=True
            )
            target_name = f'realized_vol_{period}d'
            
        elif volatility_type == 'parkinson':
            # Parkinson volatility estimator using high-low range
            if high_col is None or low_col is None or high_col not in data.columns or low_col not in data.columns:
                continue  # Skip if high/low columns not available
                
            future_range = np.log(data[high_col].shift(-period) / data[low_col].shift(-period))
            future_vol = np.sqrt((252 if annualize else 1) / (4 * period * np.log(2)) * future_range.rolling(period).apply(
                lambda x: (x ** 2).sum(), raw=True
            ))
            target_name = f'parkinson_vol_{period}d'
            
        elif volatility_type == 'garman_klass':
            # Garman-Klass volatility estimator
            if high_col is None or low_col is None or high_col not in data.columns or low_col not in data.columns:
                continue  # Skip if high/low columns not available
                
            # Calculate log returns for close prices
            log_returns = np.log(data[price_col] / data[price_col].shift(1))
            
            # Calculate high-low range
            log_hl = np.log(data[high_col] / data[low_col])
            
            # Future Garman-Klass volatility
            future_vol = np.sqrt((252 if annualize else 1) * (0.5 * log_hl.shift(-period).rolling(period).apply(
                lambda x: (x ** 2).mean(), raw=True
            ) - (2 * np.log(2) - 1) * log_returns.shift(-period).rolling(period).apply(
                lambda x: (x ** 2).mean(), raw=True
            )))
            target_name = f'garman_klass_vol_{period}d'
            
        elif volatility_type == 'rogers_satchell':
            # Rogers-Satchell volatility estimator
            if high_col is None or low_col is None or high_col not in data.columns or low_col not in data.columns:
                continue  # Skip if high/low columns not available
                
            # Calculate log price ratios
            log_ho = np.log(data[high_col] / data[open_col]) if open_col in data.columns else np.log(data[high_col] / data[price_col].shift(1))
            log_lo = np.log(data[low_col] / data[open_col]) if open_col in data.columns else np.log(data[low_col] / data[price_col].shift(1))
            log_co = np.log(data[price_col] / data[open_col]) if open_col in data.columns else np.log(data[price_col] / data[price_col].shift(1))
            
            # Future Rogers-Satchell volatility
            rs_term = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
            future_vol = np.sqrt((252 if annualize else 1) * rs_term.shift(-period).rolling(period).apply(
                lambda x: x.mean(), raw=True
            ))
            target_name = f'rogers_satchell_vol_{period}d'
            
        elif volatility_type == 'yang_zhang':
            # Yang-Zhang volatility estimator (open-high-low-close)
            if high_col is None or low_col is None or open_col is None or high_col not in data.columns or low_col not in data.columns or open_col not in data.columns:
                continue  # Skip if required columns not available
                
            # Calculate overnight volatility (close to open)
            log_co = np.log(data[open_col] / data[price_col].shift(1))
            overnight_vol = log_co.shift(-period).rolling(period).apply(
                lambda x: x.var(), raw=True
            )
            
            # Calculate open-to-close volatility components
            log_ho = np.log(data[high_col] / data[open_col])
            log_lo = np.log(data[low_col] / data[open_col])
            log_co_intraday = np.log(data[price_col] / data[open_col])
            
            rs_term = log_ho * (log_ho - log_co_intraday) + log_lo * (log_lo - log_co_intraday)
            open_close_vol = rs_term.shift(-period).rolling(period).apply(
                lambda x: x.mean(), raw=True
            )
            
            # Weight factors from Yang-Zhang paper
            k = 0.34 / (1.34 + (period + 1) / (period - 1))
            future_vol = np.sqrt((252 if annualize else 1) * (overnight_vol + k * open_close_vol))
            target_name = f'yang_zhang_vol_{period}d'
            
        elif volatility_type == 'garch':
            # Simple GARCH(1,1)-like rolling estimator
            # This is a simplified approximation, not a true GARCH model
            alpha = 0.1  # ARCH parameter
            beta = 0.85  # GARCH parameter
            omega = 0.05 * returns.var()  # Long-run variance weight
            
            # Create a function to simulate one-step GARCH forecasts
            def rolling_garch_forecast(historical_returns, forecast_horizon):
                if len(historical_returns) < 10:
                    return np.nan
                
                # Initialize with historical variance
                last_var = historical_returns.var()
                
                # For each day in the horizon, forecast one step ahead
                for _ in range(forecast_horizon):
                    # GARCH(1,1) forecast equation
                    last_var = omega + alpha * historical_returns.iloc[-1]**2 + beta * last_var
                
                return np.sqrt(last_var * (252 if annualize else 1))
            
            # Apply the GARCH forecaster to future windows
            future_vol = returns.shift(-period).rolling(
                window=period, min_periods=max(10, period//2)
            ).apply(
                lambda x: rolling_garch_forecast(x, period), raw=False
            )
            target_name = f'garch_vol_{period}d'
            
        else:
            raise ValueError(f"Invalid volatility_type: {volatility_type}. Choose from 'realized', 'parkinson', 'garman_klass', 'rogers_satchell', 'yang_zhang', or 'garch'")
        
        # Add the continuous volatility target
        builder.targets[target_name] = future_vol
        
        # Calculate relative volatility if requested
        if relative_vol:
            rel_vol = future_vol / historical_vol
            builder.targets[f'relative_{target_name}'] = rel_vol
        
        # Add classification target if requested
        if classification:
            # Create quartile-based classification
            vol_class = pd.qcut(future_vol, 4, labels=[0, 1, 2, 3], duplicates='drop')
            builder.targets[f'{target_name}_class'] = vol_class
        
        # Add volatility of volatility if requested
        if vol_of_vol:
            # Calculate historical volatility series
            hist_vol = returns.rolling(period).std() * annual_factor
            
            # Calculate volatility of volatility
            vol_of_vol_series = hist_vol.shift(-period).rolling(period).apply(
                lambda x: np.std(x) * np.sqrt(252 / period), raw=True
            )
            builder.targets[f'vol_of_vol_{period}d'] = vol_of_vol_series
        
        # Add skewness and kurtosis targets if requested
        if skew_kurtosis:
            # Calculate future skewness
            future_skew = returns.shift(-period).rolling(period).apply(
                lambda x: pd.Series(x).skew(), raw=True
            )
            builder.targets[f'skewness_{period}d'] = future_skew
            
            # Calculate future excess kurtosis
            future_kurt = returns.shift(-period).rolling(period).apply(
                lambda x: pd.Series(x).kurtosis(), raw=True
            )
            builder.targets[f'kurtosis_{period}d'] = future_kurt
        
        # Add high-low range targets if requested
        if range_targets:
            if high_col is None or low_col is None or high_col not in data.columns or low_col not in data.columns:
                continue  # Skip if high/low columns not available
            
            # Calculate future high-low range as percentage of current price
            future_high = data[high_col].shift(-period).rolling(period).max()
            future_low = data[low_col].shift(-period).rolling(period).min()
            future_range_pct = (future_high - future_low) / data[price_col]
            
            builder.targets[f'price_range_{period}d'] = future_range_pct
    
    return builder


def add_jump_risk_targets(builder: TargetBuilder,
                         price_col: str = 'close',
                         periods: list = [5, 10, 20],
                         jump_threshold: float = 0.02,
                         classification: bool = True) -> TargetBuilder:
    """
    Add targets for jump risk and extreme price movements
    
    Parameters
    ----------
    builder : TargetBuilder
        The target builder instance
    price_col : str, default='close'
        Column name for the price data
    periods : list, default=[5, 10, 20]
        List of forward periods to calculate jump risk for
    jump_threshold : float, default=0.02
        Threshold for considering a daily return as a jump (e.g., 2%)
    classification : bool, default=True
        If True, create classification targets for jump occurrence
        
    Returns
    -------
    TargetBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if price_col not in data.columns:
        return builder  # Skip if price column not found
    
    # Calculate returns
    returns = data[price_col].pct_change()
    
    for period in periods:
        # Get future returns for the period
        future_returns = returns.shift(-period).rolling(period).apply(
            lambda x: x, raw=False
        )
        
        # Calculate maximum absolute daily return in the future period
        max_abs_return = future_returns.apply(lambda x: np.max(np.abs(x)) if len(x) > 0 else np.nan)
        builder.targets[f'max_abs_return_{period}d'] = max_abs_return
        
        # Count number of jumps in the future period
        jump_count = future_returns.apply(
            lambda x: np.sum(np.abs(x) > jump_threshold) if len(x) > 0 else np.nan
        )
        builder.targets[f'jump_count_{period}d'] = jump_count
        
        # Calculate jump probability (binary classification)
        if classification:
            jump_probability = (jump_count > 0).astype(int)
            builder.targets[f'jump_probability_{period}d'] = jump_probability
            
            # Multi-class jump severity
            def classify_jump_severity(returns_series):
                if len(returns_series) == 0:
                    return np.nan
                max_jump = np.max(np.abs(returns_series))
                if max_jump <= jump_threshold:
                    return 0  # No jump
                elif max_jump <= 2 * jump_threshold:
                    return 1  # Small jump
                elif max_jump <= 3 * jump_threshold:
                    return 2  # Medium jump
                else:
                    return 3  # Large jump
            
            jump_severity = future_returns.apply(classify_jump_severity)
            builder.targets[f'jump_severity_{period}d'] = jump_severity
    
    return builder


def add_tail_risk_targets(builder: TargetBuilder,
                         price_col: str = 'close',
                         periods: list = [10, 20, 60],
                         quantiles: list = [0.01, 0.05],
                         lookback_window: int = 252) -> TargetBuilder:
    """
    Add targets for tail risk estimation
    
    Parameters
    ----------
    builder : TargetBuilder
        The target builder instance
    price_col : str, default='close'
        Column name for the price data
    periods : list, default=[10, 20, 60]
        List of forward periods to calculate tail risk for
    quantiles : list, default=[0.01, 0.05]
        Quantiles to use for Value at Risk (VaR) calculation
    lookback_window : int, default=252
        Number of days for historical VaR calculation
        
    Returns
    -------
    TargetBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if price_col not in data.columns:
        return builder  # Skip if price column not found
    
    # Calculate returns
    returns = data[price_col].pct_change()
    
    for period in periods:
        # Calculate future minimum return (worst-case scenario)
        future_min_return = returns.shift(-period).rolling(period).apply(
            lambda x: np.min(x) if len(x) > 0 else np.nan, raw=True
        )
        builder.targets[f'min_return_{period}d'] = future_min_return
        
        # Calculate expected shortfall (average of tail events)
        def expected_shortfall(returns_series, quantile=0.05):
            if len(returns_series) < 10:
                return np.nan
            cutoff = np.quantile(returns_series, quantile)
            tail_returns = returns_series[returns_series <= cutoff]
            if len(tail_returns) == 0:
                return np.nan
            return np.mean(tail_returns)
            
        for q in quantiles:
            # Calculate future expected shortfall
            future_es = returns.shift(-period).rolling(period).apply(
                lambda x: expected_shortfall(x, q), raw=False
            )
            builder.targets[f'expected_shortfall_{int(q*100)}pct_{period}d'] = future_es
            
            # Calculate whether future minimum return exceeds historical VaR
            # First calculate historical VaR
            historical_var = returns.rolling(lookback_window).apply(
                lambda x: np.quantile(x, q), raw=True
            )
            
            # Then check if future minimum return exceeds this VaR
            var_breach = (future_min_return < historical_var).astype(int)
            builder.targets[f'var_breach_{int(q*100)}pct_{period}d'] = var_breach
    
    return builder