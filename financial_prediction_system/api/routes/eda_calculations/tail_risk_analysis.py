import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import genpareto, t
import json
from plotly.utils import PlotlyJSONEncoder

# Import the configured logger
from financial_prediction_system.logging_config import logger

def calculate_returns(data: pd.DataFrame) -> pd.Series:
    """Calculates daily log returns."""
    # Define potential close column names
    close_col = None
    if 'Close' in data.columns:
        close_col = 'Close'
    elif 'close' in data.columns:
        close_col = 'close'
    elif 'Adj Close' in data.columns: # Added common alternative
        close_col = 'Adj Close'
    elif 'adj_close' in data.columns: # Added common alternative
        close_col = 'adj_close'
    # Add other potential names if needed

    # Check for close column
    if close_col is None:
        logger.error(f"Tail Risk: Close column ('Close', 'close', 'Adj Close', etc.) not found in input DataFrame. Columns: {data.columns.tolist()}")
        return pd.Series(dtype=float) # Return empty series
    
    # Check if data is sufficient
    if len(data) < 2:
        logger.warning("Tail Risk: Insufficient data (less than 2 rows) to calculate returns.")
        return pd.Series(dtype=float)
    
    try:
        close_prices = pd.to_numeric(data[close_col], errors='coerce')
        shifted_close = close_prices.shift(1)
        # Avoid division by zero or log(negative/zero)
        valid_mask = (close_prices > 1e-9) & (shifted_close > 1e-9)
        returns = pd.Series(np.nan, index=data.index) # Initialize with original index
        returns.loc[valid_mask] = np.log(close_prices[valid_mask] / shifted_close[valid_mask])
        # Drop NaN resulting from shift and any potential existing NaNs
        returns = returns.dropna()
        logger.info(f"Tail Risk: Calculated {len(returns)} returns using column '{close_col}'.")
        return returns
    except Exception as e:
         logger.error(f"Tail Risk: Error calculating returns using column '{close_col}': {e}", exc_info=True)
         return pd.Series(dtype=float)

def calculate_historical_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculates historical Value at Risk (VaR)."""
    if returns is None or returns.empty:
        logger.warning("Tail Risk (Hist VaR): Input returns series is empty.")
        return np.nan
    try:
        var_val = returns.quantile(1 - confidence_level)
        if pd.isna(var_val):
            logger.warning("Tail Risk (Hist VaR): Calculation resulted in NaN (possibly due to all NaN input or insufficient data).")
        # logger.debug(f"Tail Risk (Hist VaR): Calculated {var_val:.4f} at {confidence_level*100:.0f}% level.")
        return var_val
    except Exception as e:
         logger.error(f"Tail Risk (Hist VaR): Error calculating: {e}", exc_info=True)
         return np.nan

def calculate_historical_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """Calculates historical Conditional Value at Risk (CVaR)."""
    if returns is None or returns.empty:
        logger.warning("Tail Risk (Hist CVaR): Input returns series is empty.")
        return np.nan
    try:
        var = calculate_historical_var(returns, confidence_level)
        if pd.isna(var):
             logger.warning("Tail Risk (Hist CVaR): Underlying VaR is NaN. Cannot calculate CVaR.")
             return np.nan
        # Calculate CVaR only on returns less than or equal to VaR
        tail_returns = returns[returns <= var]
        if tail_returns.empty:
            logger.warning(f"Tail Risk (Hist CVaR): No returns found at or below VaR ({var:.4f}). Returning VaR as CVaR.")
            # In this edge case, CVaR is often considered equal to VaR, or could return NaN
            return var # Or np.nan depending on desired behavior
        cvar_val = tail_returns.mean()
        if pd.isna(cvar_val):
             logger.warning("Tail Risk (Hist CVaR): Calculation resulted in NaN.")
        # logger.debug(f"Tail Risk (Hist CVaR): Calculated {cvar_val:.4f} based on {len(tail_returns)} tail returns.")
        return cvar_val
    except Exception as e:
         logger.error(f"Tail Risk (Hist CVaR): Error calculating: {e}", exc_info=True)
         return np.nan

def calculate_parametric_var(returns: pd.Series, confidence_level: float = 0.95, distribution: str = 'normal') -> float:
    """Calculates parametric VaR using normal or t-distribution."""
    if returns is None or returns.empty or len(returns.dropna()) < 2:
        logger.warning(f"Tail Risk (Parametric VaR - {distribution}): Insufficient valid data ({len(returns.dropna()) if returns is not None else 0}).")
        return np.nan
        
    valid_returns = returns.dropna()
    mean = valid_returns.mean()
    std = valid_returns.std()
    if std == 0 or pd.isna(std):
         logger.warning(f"Tail Risk (Parametric VaR - {distribution}): Standard deviation is zero or NaN. Cannot calculate.")
         return np.nan
         
    try:
        if distribution == 'normal':
            z_score = stats.norm.ppf(1 - confidence_level)
            var_val = mean + z_score * std
            # logger.debug(f"Tail Risk (Parametric VaR - Normal): Calculated {var_val:.4f}")
            return var_val
        elif distribution == 't':
            # Fit t-distribution, handle potential errors
            try:
                 df, loc, scale = stats.t.fit(valid_returns)
                 if pd.isna(df) or df <= 0:
                     logger.warning(f"Tail Risk (Parametric VaR - t): Invalid degrees of freedom ({df}) from t-fit. Falling back to Normal.")
                     return calculate_parametric_var(returns, confidence_level, 'normal') # Fallback
                 # logger.debug(f"Tail Risk (Parametric VaR - t): Fit df={df:.2f}, loc={loc:.4f}, scale={scale:.4f}")
                 t_score = stats.t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
                 # Parametric VaR using fitted params: t_score is already the quantile
                 var_val = t_score
                 # logger.debug(f"Tail Risk (Parametric VaR - t): Calculated {var_val:.4f}")
                 return var_val
            except Exception as fit_err:
                 logger.warning(f"Tail Risk (Parametric VaR - t): Failed to fit t-distribution ({fit_err}). Falling back to Normal.", exc_info=False)
                 return calculate_parametric_var(returns, confidence_level, 'normal') # Fallback
        else:
            logger.error(f"Tail Risk (Parametric VaR): Invalid distribution specified: {distribution}")
            raise ValueError("Distribution must be 'normal' or 't'")
    except Exception as e:
        logger.error(f"Tail Risk (Parametric VaR - {distribution}): Error calculating: {e}", exc_info=True)
        return np.nan

def calculate_parametric_cvar(returns: pd.Series, confidence_level: float = 0.95, distribution: str = 'normal') -> float:
    """Calculates parametric CVaR using normal or t-distribution."""
    if returns is None or returns.empty or len(returns.dropna()) < 2:
        logger.warning(f"Tail Risk (Parametric CVaR - {distribution}): Insufficient valid data.")
        return np.nan
        
    valid_returns = returns.dropna()
    mean = valid_returns.mean()
    std = valid_returns.std()
    if std == 0 or pd.isna(std):
         logger.warning(f"Tail Risk (Parametric CVaR - {distribution}): Standard deviation is zero or NaN. Cannot calculate.")
         return np.nan

    try:
        if distribution == 'normal':
            z_score = stats.norm.ppf(1 - confidence_level)
            cvar = mean - std * stats.norm.pdf(z_score) / (1 - confidence_level)
            # logger.debug(f"Tail Risk (Parametric CVaR - Normal): Calculated {cvar:.4f}")
            return cvar
        elif distribution == 't':
            # Fit t-distribution, handle potential errors
            try:
                 df, loc, scale = stats.t.fit(valid_returns)
                 if pd.isna(df) or df <= 0:
                     logger.warning(f"Tail Risk (Parametric CVaR - t): Invalid degrees of freedom ({df}) from t-fit. Falling back to Normal.")
                     return calculate_parametric_cvar(returns, confidence_level, 'normal') # Fallback
                 # logger.debug(f"Tail Risk (Parametric CVaR - t): Fit df={df:.2f}, loc={loc:.4f}, scale={scale:.4f}")
                 t_score_var = stats.t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
                 # Expected value below VaR for t-distribution
                 # E[X | X <= VaR] = loc - scale * (df + ( (VaR-loc)/scale )^2) / (df - 1) * pdf( (VaR-loc)/scale ) / CDF( (VaR-loc)/scale )
                 var_standardized = (t_score_var - loc) / scale
                 pdf_val = stats.t.pdf(var_standardized, df)
                 cdf_val = 1 - confidence_level # CDF at VaR is alpha
                 if df > 1:
                      factor = scale * (df + var_standardized**2) / (df - 1)
                      cvar = loc - factor * (pdf_val / cdf_val)
                      # logger.debug(f"Tail Risk (Parametric CVaR - t): Calculated {cvar:.4f}")
                 else:
                      logger.warning(f"Tail Risk (Parametric CVaR - t): Cannot calculate CVaR for df={df} <= 1. Mean is undefined.")
                      cvar = np.nan # CVaR undefined if df <= 1
                 return cvar
            except Exception as fit_err:
                 logger.warning(f"Tail Risk (Parametric CVaR - t): Failed to fit t-distribution ({fit_err}). Falling back to Normal.", exc_info=False)
                 return calculate_parametric_cvar(returns, confidence_level, 'normal') # Fallback
        else:
            logger.error(f"Tail Risk (Parametric CVaR): Invalid distribution specified: {distribution}")
            raise ValueError("Distribution must be 'normal' or 't'")
    except Exception as e:
         logger.error(f"Tail Risk (Parametric CVaR - {distribution}): Error calculating: {e}", exc_info=True)
         return np.nan


def plot_var_comparison(returns: pd.Series, confidence_level: float = 0.95) -> go.Figure:
    """Compares different VaR methodologies."""
    fig = go.Figure()
    title = f'Value at Risk ({confidence_level*100:.0f}%) Comparison'
    
    if returns is None or returns.empty:
        logger.warning("Tail Risk (VaR Plot): Input returns are empty. Returning empty plot.")
        fig.update_layout(title=f'{title} (No Data)')
        return fig
        
    methods = ['Historical', 'Parametric (Normal)', 'Parametric (t)']
    vars_ = [
        calculate_historical_var(returns, confidence_level),
        calculate_parametric_var(returns, confidence_level, 'normal'),
        calculate_parametric_var(returns, confidence_level, 't')
    ]
    
    # Filter out NaN values for plotting
    valid_methods = [m for m, v in zip(methods, vars_) if pd.notnull(v)]
    valid_vars = [v for v in vars_ if pd.notnull(v)]
    
    logger.info(f"Tail Risk (VaR Plot): VaR values calculated: Historical={vars_[0]:.4f}, Normal={vars_[1]:.4f}, t={vars_[2]:.4f}")

    if not valid_vars:
        logger.warning("Tail Risk (VaR Plot): All VaR calculation methods resulted in NaN. Returning empty plot.")
        fig.update_layout(title=f'{title} (Calculation Error)')
        return fig
        
    fig.add_trace(go.Bar(
        x=valid_methods,
        y=[-v * 100 for v in valid_vars], # Display as positive percentage loss
        text=[f'{-v*100:.2f}%' for v in valid_vars], # Add text labels
        textposition='auto',
        marker_color=px.colors.qualitative.Plotly[:len(valid_methods)] # Use Plotly colors dynamically
    ))
    fig.update_layout(
        title=title,
        yaxis_title='Potential Loss (%)',
        xaxis_title='Methodology',
        bargap=0.2,
        yaxis_range=[0, max([-v * 100 for v in valid_vars] + [0]) * 1.1] # Adjust y-axis range, handle if max loss is positive
    )
    return fig

def plot_rolling_var(returns: pd.Series, window: int = 63, confidence_level: float = 0.95) -> go.Figure:
    """Plots rolling VaR and CVaR."""
    fig = go.Figure()
    title = f'Rolling VaR & CVaR ({window}-Day Window, {confidence_level*100:.0f}% Confidence)'

    if returns is None or returns.empty or len(returns.dropna()) < window:
        logger.warning(f"Tail Risk (Rolling VaR Plot): Insufficient data (need > {window} valid points) for rolling window. Returning empty plot.")
        fig.update_layout(title=f'{title} (Insufficient Data)')
        return fig

    try:
        # Use .apply for potentially complex calculations, raw=False needed for quantile etc.
        rolling_hist_var = returns.rolling(window).apply(lambda x: calculate_historical_var(pd.Series(x), confidence_level), raw=False) * 100
        rolling_hist_cvar = returns.rolling(window).apply(lambda x: calculate_historical_cvar(pd.Series(x), confidence_level), raw=False) * 100

        # Drop initial NaNs from rolling calculation for cleaner plotting
        rolling_hist_var = rolling_hist_var.dropna()
        rolling_hist_cvar = rolling_hist_cvar.dropna()

        logger.info(f"Tail Risk (Rolling VaR Plot): Calculated {len(rolling_hist_var)} rolling VaR points and {len(rolling_hist_cvar)} rolling CVaR points.")

        plot_added = False
        if not rolling_hist_var.empty:
            fig.add_trace(go.Scatter(x=rolling_hist_var.index, y=-rolling_hist_var, mode='lines', name=f'Rolling VaR ({confidence_level*100:.0f}%)', line=dict(color='#1f77b4')))
            plot_added = True
        else:
             logger.warning("Tail Risk (Rolling VaR Plot): Rolling VaR calculation resulted in all NaNs.")

        if not rolling_hist_cvar.empty:
            fig.add_trace(go.Scatter(x=rolling_hist_cvar.index, y=-rolling_hist_cvar, mode='lines', name=f'Rolling CVaR ({confidence_level*100:.0f}%)', line=dict(dash='dash', color='#ff7f0e')))
            plot_added = True
        else:
            logger.warning("Tail Risk (Rolling VaR Plot): Rolling CVaR calculation resulted in all NaNs.")

        if not plot_added:
            logger.warning("Tail Risk (Rolling VaR Plot): No valid rolling VaR or CVaR data to plot.")
            fig.update_layout(title=f'{title} (Calculation Error)')
            return fig

        fig.update_layout(
            title=title,
            yaxis_title='Potential Loss (%)',
            xaxis_title='Date',
            hovermode='x unified'
        )
        return fig
    except Exception as e:
        logger.error(f"Tail Risk (Rolling VaR Plot): Error during calculation or plotting: {e}", exc_info=True)
        fig.update_layout(title=f'{title} (Error)')
        return fig

def fit_gpd(returns: pd.Series, threshold_quantile: float = 0.95):
    """Fits Generalized Pareto Distribution to exceedances over a threshold."""
    if returns is None or returns.empty:
        logger.warning("Tail Risk (GPD Fit): Input returns series is empty.")
        return None, None, None, None
        
    # Work with positive losses
    losses = -returns.dropna()
    if losses.empty:
         logger.warning("Tail Risk (GPD Fit): Returns series is empty after dropping NaNs.")
         return None, None, None, None
         
    if not (0 < threshold_quantile < 1):
         logger.error(f"Tail Risk (GPD Fit): Invalid threshold_quantile ({threshold_quantile}). Must be between 0 and 1.")
         return None, None, None, None

    try:
        threshold = losses.quantile(threshold_quantile)
        exceedances = losses[losses > threshold] - threshold
        logger.info(f"Tail Risk (GPD Fit): Found {len(exceedances)} exceedances above threshold {threshold:.4f} (quantile={threshold_quantile}) for {len(losses)} losses.")
        
        # Check for sufficient unique exceedances
        if len(exceedances) < 10 or exceedances.nunique() < 3: 
            logger.warning(f"Tail Risk (GPD Fit): Insufficient or non-unique exceedances ({len(exceedances)} total, {exceedances.nunique()} unique) to reliably fit GPD model. Need >= 10 total and >= 3 unique.")
            return None, None, threshold, exceedances # Return threshold and exceedances even if fit fails
            
        # Fit GPD (Shape parameter 'c', Location 'loc', Scale 'scale')
        # Use floc=0 as we shifted exceedances by the threshold
        # Handle potential optimization errors
        c, loc, scale = genpareto.fit(exceedances, floc=0)
        # Check if fit parameters are reasonable
        if pd.isna(c) or pd.isna(scale) or scale <= 0:
             logger.warning(f"Tail Risk (GPD Fit): Fit resulted in invalid parameters (c={c}, scale={scale}).")
             return None, None, threshold, exceedances
             
        logger.info(f"Tail Risk (GPD Fit): Successful: c={c:.4f}, scale={scale:.4f}, loc={loc:.4f} (fixed)")
        return c, scale, threshold, exceedances
    except Exception as e: # Catch potential fitting errors (e.g., LinAlgError, ValueError)
        logger.error(f"Tail Risk (GPD Fit): Failed to fit GPD model: {e}", exc_info=True)
        # Attempt to return threshold and exceedances if calculated before error
        try:
            threshold_val = threshold if 'threshold' in locals() else np.nan
            exceedances_val = exceedances if 'exceedances' in locals() else None
            return None, None, threshold_val, exceedances_val
        except: return None, None, None, None

def plot_return_level(returns: pd.Series, threshold_quantile: float = 0.95, return_periods = [2, 5, 10, 20, 50, 100, 252]) -> go.Figure:
    """Plots the EVT Return Level plot using GPD."""
    fig = go.Figure()
    title = f'EVT Return Level Plot (GPD, Threshold Quantile={threshold_quantile*100:.0f}%)'
    
    if returns is None or returns.empty:
        logger.warning("Tail Risk (Return Level Plot): Input returns are empty.")
        fig.update_layout(title=f'{title} (No Data)')
        return fig
        
    # Fit GPD to losses (negative returns)
    neg_returns = -returns.dropna() # Ensure no NaNs and work with positive losses
    if neg_returns.empty:
         logger.warning("Tail Risk (Return Level Plot): No valid negative returns to analyze.")
         fig.update_layout(title=f'{title} (No Data)')
         return fig
         
    logger.info(f"Tail Risk (Return Level Plot): Calculating based on {len(neg_returns)} negative returns.")
    c, scale, threshold, exceedances = fit_gpd(neg_returns, threshold_quantile)
    
    if c is None or scale is None or threshold is None:
         logger.warning("Tail Risk (Return Level Plot): Cannot plot return levels due to missing/invalid GPD fit parameters.")
         fig.update_layout(title=f'{title} (Fit Error or Insufficient Data)')
         # Optionally add annotation about the failure
         fig.add_annotation(text="GPD fit failed or insufficient exceedances", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
         return fig # Return empty figure with title

    n = len(neg_returns)
    n_u = len(exceedances) if exceedances is not None else 0
    if n_u == 0:
         logger.warning("Tail Risk (Return Level Plot): Number of exceedances is zero. Cannot calculate return levels.")
         fig.update_layout(title=f'{title} (No Exceedances)')
         return fig
         
    lambda_u = n_u / n # Probability of exceeding threshold
    logger.info(f"Tail Risk (Return Level Plot): GPD Params: c={c:.4f}, scale={scale:.4f}, threshold={threshold:.4f}, n={n}, n_u={n_u}, lambda_u={lambda_u:.4f}")

    levels = []
    valid_return_periods = []
    for T in return_periods:
        try:
            # GPD return level formula
            if abs(c) < 1e-6: # Handle c close to 0 (Exponential distribution case)
                level = threshold + scale * np.log(lambda_u * T)
            else:
                # Ensure term inside power is positive if c is fractional
                term = lambda_u * T
                if term <= 0 and c > 0 and c < 1:
                    logger.warning(f"Tail Risk (Return Level): Skipping T={T} due to non-positive base for fractional exponent c={c:.4f}")
                    continue 
                # Avoid potential overflow/invalid ops
                power_term = term**c 
                level = threshold + (scale / c) * (power_term - 1)
                
            if pd.notna(level):
                levels.append(level * 100) # As percentage loss
                valid_return_periods.append(T)
            else:
                 logger.warning(f"Tail Risk (Return Level): Calculated level is NaN for T={T}")
                 
        except (ValueError, OverflowError) as calc_err:
            logger.warning(f"Tail Risk (Return Level): Calculation error for T={T}: {calc_err}")
            continue # Skip this return period

    if not levels:
         logger.warning("Tail Risk (Return Level Plot): No valid return levels could be calculated.")
         fig.update_layout(title=f'{title} (Calculation Error)')
         return fig
         
    fig.add_trace(go.Scatter(
        x=valid_return_periods,
        y=levels,
        mode='lines+markers',
        name='Estimated Return Level'
    ))
    fig.update_layout(
        title=title,
        yaxis_title='Expected Loss (%)',
        xaxis_title='Return Period (Days)',
        xaxis_type='log' # Often plotted on log scale
    )
    return fig

def calculate_hill_estimator(returns: pd.Series):
    """Calculates the Hill estimator for various numbers of order statistics."""
    if returns is None or returns.empty:
        logger.warning("Tail Risk (Hill Estimator): Input returns series is empty.")
        return [], []
        
    # Drop NaNs/Infs and ensure numeric type before abs/sort
    valid_returns = returns.dropna()
    valid_returns = valid_returns[np.isfinite(valid_returns)]
    if valid_returns.empty:
        logger.warning("Tail Risk (Hill Estimator): No valid (non-NaN, finite) returns found.")
        return [], []
        
    # Use absolute returns for tail index estimation
    abs_returns_sorted = np.sort(np.abs(valid_returns.astype(float)))[::-1] # Sort absolute returns descending
    n = len(abs_returns_sorted)
    logger.info(f"Tail Risk (Hill Estimator): Calculating for {n} valid absolute returns.")
    hill_estimates = []
    
    # Determine k range dynamically, e.g., from k=10 up to n//10 or a max cap
    k_start = 10
    # Cap k at a fraction of n (e.g., 10%) or an absolute max like 500
    k_end = min(n - 2, max(k_start, n // 10), 500) 
    
    if k_end < k_start:
        logger.warning(f"Tail Risk (Hill Estimator): Insufficient data points ({n}) to generate Hill k_values range ({k_start} to {k_end}). Returning empty lists.")
        return [], []
        
    k_values = range(k_start, k_end + 1)
    logger.info(f"Tail Risk (Hill Estimator): k range: {k_start} to {k_end}")
    calculated_k_values = [] # Keep track of k for which estimate was calculated

    for k in k_values:
        try:
            # Get the top k+1 absolute returns
            top_k_plus_1_returns = abs_returns_sorted[:k+1]
            
            # Check for non-positive values before log (should be rare after abs)
            if np.any(top_k_plus_1_returns <= 1e-12): # Use a small threshold instead of 0
                # logger.debug(f"Skipping k={k} in Hill calculation due to non-positive returns <= 1e-12.")
                continue
                
            log_returns = np.log(top_k_plus_1_returns)
            
            # Ensure denominator is not zero
            # Formula: alpha_hat = 1 / ( (1/k * sum(log(X_i) for i=1 to k)) - log(X_{k+1}) )
            mean_log_top_k = np.mean(log_returns[:-1]) # Mean of log of top k
            log_k_plus_1 = log_returns[-1] # Log of (k+1)th value
            denominator = mean_log_top_k - log_k_plus_1
            
            if abs(denominator) < 1e-9:
                 # logger.debug(f"Skipping k={k} in Hill calculation due to near-zero denominator ({denominator:.2e}).")
                 continue
                 
            hill_alpha = 1.0 / denominator
            # Tail index xi = 1 / alpha
            hill_xi = 1.0 / hill_alpha 
            
            if pd.notna(hill_xi):
                hill_estimates.append(hill_xi)
                calculated_k_values.append(k)
            # else: logger.debug(f"Hill calculation resulted in NaN for k={k}")
                
        except Exception as e:
            logger.warning(f"Tail Risk (Hill Estimator): Error calculating for k={k}: {e}", exc_info=False)
            continue # Skip to next k if error occurs
            
    logger.info(f"Tail Risk (Hill Estimator): Calculated {len(hill_estimates)} estimates for k values {calculated_k_values[0]} to {calculated_k_values[-1] if calculated_k_values else 'N/A'}.")
    return calculated_k_values, hill_estimates # Return xi (tail index)

def plot_hill(returns: pd.Series) -> go.Figure:
    """Plots the Hill estimator against the number of order statistics (k)."""
    fig = go.Figure()
    title = 'Hill Plot (Tail Index Estimator ξ)'

    if returns is None or returns.empty:
        logger.warning("Tail Risk (Hill Plot): Input returns are empty.")
        fig.update_layout(title=f'{title} (No Data)')
        return fig

    k_values, hill_estimates = calculate_hill_estimator(returns)

    if not k_values or not hill_estimates:
        logger.warning("Tail Risk (Hill Plot): Hill estimator calculation yielded no results.")
        fig.update_layout(title=f'{title} (Calculation Error or Insufficient Data)')
        # Optionally add annotation
        fig.add_annotation(text="Could not calculate Hill estimates", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    fig.add_trace(go.Scatter(
        x=k_values,
        y=hill_estimates,
        mode='lines+markers',
        name='Hill Estimator (ξ = 1/α)'
    ))
    fig.update_layout(
        title=title,
        yaxis_title='Tail Index Estimate (ξ)',
        xaxis_title='Number of Order Statistics (k)',
        hovermode='x unified'
    )
    # Add horizontal line at 0 for reference (often indicates transition)
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    return fig

def calculate_tail_risk_plots(data: pd.DataFrame) -> dict:
    """Calculates all tail risk metrics and generates plots."""
    logger.info("Starting tail risk analysis...")
    results = {}
    
    if data is None or data.empty:
        logger.warning("Tail Risk Analysis: Input DataFrame is empty. Skipping all calculations.")
        # Return dictionary with None values to match expected structure
        return {
            'var_comparison_plot': None,
            'rolling_var_cvar_plot': None,
            'evt_return_level_plot': None,
            'hill_plot': None
        }
        
    # 1. Calculate Returns
    returns = calculate_returns(data)
    
    # Check if returns calculation was successful
    if returns is None or returns.empty:
        logger.warning("Tail Risk Analysis: Return calculation failed or resulted in empty series. Skipping plots.")
        # Generate empty plots with informative titles
        empty_results = {}
        try:
             # Create empty figures with informative titles
             var_comp_fig = go.Figure().update_layout(title="VaR Comparison (No Return Data)")
             roll_var_fig = go.Figure().update_layout(title="Rolling VaR/CVaR (No Return Data)")
             evt_ret_fig = go.Figure().update_layout(title="EVT Return Level (No Return Data)")
             hill_fig = go.Figure().update_layout(title="Hill Plot (No Return Data)")
             
             empty_results['var_comparison_plot'] = json.loads(json.dumps(var_comp_fig, cls=PlotlyJSONEncoder))
             empty_results['rolling_var_cvar_plot'] = json.loads(json.dumps(roll_var_fig, cls=PlotlyJSONEncoder))
             empty_results['evt_return_level_plot'] = json.loads(json.dumps(evt_ret_fig, cls=PlotlyJSONEncoder))
             empty_results['hill_plot'] = json.loads(json.dumps(hill_fig, cls=PlotlyJSONEncoder))
        except Exception as e:
            logger.error(f"Tail Risk Analysis: Error serializing empty plots: {e}", exc_info=True)
            # Fallback to returning None values if serialization of empty plots fails
            return {'var_comparison_plot': None, 'rolling_var_cvar_plot': None, 'evt_return_level_plot': None, 'hill_plot': None}
        return {k: v for k, v in empty_results.items() if v} # Filter None just in case serialization fails

    # 2. Generate Plots (plots internally handle NaN/errors)
    logger.info("Generating VaR comparison plot...")
    try:
        var_comp_fig = plot_var_comparison(returns)
        results['var_comparison_plot'] = json.loads(json.dumps(var_comp_fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error generating/serializing VaR comparison plot: {e}", exc_info=True)
        results['var_comparison_plot'] = None

    logger.info("Generating rolling VaR/CVaR plot...")
    try:
        roll_var_fig = plot_rolling_var(returns)
        results['rolling_var_cvar_plot'] = json.loads(json.dumps(roll_var_fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error generating/serializing rolling VaR/CVaR plot: {e}", exc_info=True)
        results['rolling_var_cvar_plot'] = None
        
    logger.info("Generating EVT return level plot...")
    try:
        evt_ret_fig = plot_return_level(returns)
        results['evt_return_level_plot'] = json.loads(json.dumps(evt_ret_fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error generating/serializing EVT return level plot: {e}", exc_info=True)
        results['evt_return_level_plot'] = None

    logger.info("Generating Hill plot...")
    try:
        hill_fig = plot_hill(returns)
        results['hill_plot'] = json.loads(json.dumps(hill_fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error generating/serializing Hill plot: {e}", exc_info=True)
        results['hill_plot'] = None

    # Filter out None results before returning
    final_results = {k: v for k, v in results.items() if v is not None}
    logger.info(f"Tail risk analysis complete. Generated {len(final_results)} plots.")
    return final_results
