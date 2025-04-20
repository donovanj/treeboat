import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
import ta
from hurst import compute_Hc # Assuming 'hurst' library is installed
import plotly.graph_objs as go
import json
from plotly.utils import PlotlyJSONEncoder
from typing import Tuple, Dict, Any, Optional, List, Union
import plotly.subplots as sp

# FastAPI related imports (should already be present lower down, but ensure they are available)
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

# Constants
DEFAULT_LAGS = 30
ROLLING_WINDOWS = [5, 10, 20, 60, 120]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_WINDOW = 20
BB_STD = 2
HURST_WINDOW = 100 # Window for rolling Hurst exponent
ROLLING_WINDOW_ANALYSIS = 120 # Window size for rolling ACF/ADF etc.
DEFAULT_WINDOW_PHASE_PLOT = 20 # Default window for indicators in phase plot
DEFAULT_WINDOW_MOMENTUM = 10

# --- Helper Functions (Assuming these were previously restored or are present) ---

def _calculate_acf_pacf(series: pd.Series, nlags: int, title_suffix: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Calculates ACF and PACF plots data."""
    if series.empty or len(series) < nlags + 2: # Need enough data points
        print(f"Warning: Insufficient data for ACF/PACF calculation ({title_suffix}).")
        return None, None
        
    try:
        # Calculate ACF and PACF
        acf_values, confint_acf = acf(series, nlags=nlags, alpha=0.05, fft=False) # fft=False can be more stable
        pacf_values, confint_pacf = pacf(series, nlags=nlags, alpha=0.05, method='ols')

        # Confidence intervals (assuming symmetry for plotting)
        # Recalculate CI bounds relative to zero
        ci_upper_acf = confint_acf[:, 1] - acf_values 
        ci_lower_acf = confint_acf[:, 0] - acf_values
        ci_upper_pacf = confint_pacf[:, 1] - pacf_values
        ci_lower_pacf = confint_pacf[:, 0] - pacf_values

        lags = list(range(len(acf_values)))

        acf_plot = {
            "data": [
                {"type": "bar", "x": lags, "y": acf_values.tolist(), "name": "ACF"},
                 # Confidence Interval area - Plotting bounds directly now
                {"type": "scatter", "x": lags + lags[::-1], # x values for polygon
                 "y": (confint_acf[:, 1]).tolist() + (confint_acf[:, 0][::-1]).tolist(), # y values for polygon
                 "fill": "toself", "fillcolor": "rgba(255,0,0,0.1)", "line": {"color": "transparent"}, 
                 "name": "95% Confidence Interval", "showlegend": False}
            ],
            "layout": {"title": f"Autocorrelation Function (ACF) - {title_suffix}", "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF"}}
        }
        pacf_plot = {
             "data": [
                {"type": "bar", "x": lags[1:], "y": pacf_values[1:].tolist(), "name": "PACF"}, # Skip lag 0 for PACF
                # Confidence Interval area
                {"type": "scatter", "x": lags[1:] + lags[1:][::-1], # x values for polygon
                 "y": (confint_pacf[1:, 1]).tolist() + (confint_pacf[1:, 0][::-1]).tolist(), # y values for polygon
                 "fill": "toself", "fillcolor": "rgba(255,0,0,0.1)", "line": {"color": "transparent"}, 
                 "name": "95% Confidence Interval", "showlegend": False}
            ],
            "layout": {"title": f"Partial Autocorrelation Function (PACF) - {title_suffix}", "xaxis": {"title": "Lag"}, "yaxis": {"title": "PACF"}}
        }
        return json.loads(json.dumps(acf_plot, cls=PlotlyJSONEncoder)), json.loads(json.dumps(pacf_plot, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating ACF/PACF for {title_suffix}: {e}")
        return None, None

def _create_heatmap(z_data: np.ndarray, x_labels: List[str], y_labels: List[int], title: str, colorscale: str = "Viridis", zmid: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Helper function to create Plotly heatmap data."""
    if z_data is None or z_data.size == 0 or not x_labels or not y_labels:
        print(f"Warning: Insufficient data for heatmap: {title}")
        return None
        
    heatmap_layout = {
        "title": title,
        "xaxis": {"title": "Date (End of Window)"},
        "yaxis": {"title": "Lag", "dtick": 5} # Adjust tick frequency if needed
    }
    
    heatmap_data = {
        "type": "heatmap", 
        "z": z_data.tolist(), 
        "x": x_labels, 
        "y": y_labels, 
        "colorscale": colorscale
    }
    
    if zmid is not None:
        heatmap_data["zmid"] = zmid
        
    plot_data = {
        "data": [heatmap_data],
        "layout": heatmap_layout
    }
    return json.loads(json.dumps(plot_data, cls=PlotlyJSONEncoder))

# --- Autocorrelation Analysis ---

def calculate_return_autocorrelation(df: pd.DataFrame, nlags: int = DEFAULT_LAGS) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Calculates ACF and PACF for daily price returns."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < nlags + 3:
         print(f"Warning: Insufficient data for return autocorrelation.")
         return None, None
         
    returns = df['close'].pct_change().dropna()
    return _calculate_acf_pacf(returns, nlags, "Daily Returns")

def calculate_volatility_autocorrelation(df: pd.DataFrame, nlags: int = DEFAULT_LAGS) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Calculates ACF and PACF for absolute daily returns (volatility proxy)."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < nlags + 3:
        print(f"Warning: Insufficient data for volatility autocorrelation.")
        return None, None
        
    returns = df['close'].pct_change().dropna()
    # Using absolute returns as a proxy for volatility
    abs_returns = returns.abs()
    return _calculate_acf_pacf(abs_returns, nlags, "Absolute Daily Returns")

def calculate_rolling_acf_pacf_heatmap(series: pd.Series, window: int, nlags: int, title_suffix: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Calculates rolling ACF and PACF and returns heatmap data."""
    if series.empty or len(series) < window + nlags + 1:
        print(f"Warning: Insufficient data for rolling ACF/PACF ({title_suffix}). Need {window + nlags + 1}, have {len(series)}.")
        return None, None

    all_acf_values = []
    all_pacf_values = []
    valid_indices = []

    for i in range(window, len(series)):
        segment = series.iloc[i-window:i]
        if segment.var() < 1e-10: # Skip if variance is near zero
             continue
             
        try:
            # Calculate ACF/PACF for the segment (excluding lag 0)
            acf_vals, _ = acf(segment, nlags=nlags, alpha=0.05, fft=False)
            pacf_vals, _ = pacf(segment, nlags=nlags, alpha=0.05, method='ols')
            
            all_acf_values.append(acf_vals[1:]) # Exclude lag 0
            all_pacf_values.append(pacf_vals[1:]) # Exclude lag 0
            # Ensure index elements are Timestamps before adding
            valid_indices.append(pd.Timestamp(series.index[i])) # Convert potential int64 to Timestamp
        except Exception as e:
            # print(f"Skipping segment ending at {series.index[i]} due to ACF/PACF error: {e}")
            continue # Skip segment if calculation fails

    if not all_acf_values or not valid_indices:
        print(f"Warning: No valid segments found for rolling ACF/PACF ({title_suffix}).")
        return None, None

    # Transpose for heatmap: rows=lags, cols=time
    acf_matrix = np.array(all_acf_values).T
    pacf_matrix = np.array(all_pacf_values).T
    
    # Now valid_indices definitely contains Timestamps
    dates_str = [d.strftime('%Y-%m-%d') for d in valid_indices]
    lags_list = list(range(1, nlags + 1)) # Lags from 1 to nlags

    acf_heatmap = _create_heatmap(acf_matrix, dates_str, lags_list, f"Rolling ACF Heatmap - {title_suffix}", colorscale="RdBu", zmid=0)
    pacf_heatmap = _create_heatmap(pacf_matrix, dates_str, lags_list, f"Rolling PACF Heatmap - {title_suffix}", colorscale="RdBu", zmid=0)

    return acf_heatmap, pacf_heatmap

def calculate_rolling_return_acf_heatmap(df: pd.DataFrame, window: int = ROLLING_WINDOW_ANALYSIS, nlags: int = DEFAULT_LAGS) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
     """Wrapper to calculate rolling ACF/PACF heatmap for returns."""
     if df is None or df.empty or 'close' not in df.columns or len(df) < window + nlags + 2:
         print(f"Warning: Insufficient data for rolling return ACF heatmap.")
         return None, None
     returns = df['close'].pct_change().dropna()
     return calculate_rolling_acf_pacf_heatmap(returns, window, nlags, "Daily Returns")

def calculate_rolling_vol_acf_heatmap(df: pd.DataFrame, window: int = ROLLING_WINDOW_ANALYSIS, nlags: int = DEFAULT_LAGS) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
     """Wrapper to calculate rolling ACF/PACF heatmap for absolute returns."""
     if df is None or df.empty or 'close' not in df.columns or len(df) < window + nlags + 2:
          print(f"Warning: Insufficient data for rolling volatility ACF heatmap.")
          return None, None
     returns = df['close'].pct_change().dropna()
     abs_returns = returns.abs()
     return calculate_rolling_acf_pacf_heatmap(abs_returns, window, nlags, "Absolute Daily Returns")

# --- Momentum/Reversion Interaction Analysis ---

def calculate_momentum_reversion_phase_plot(df: pd.DataFrame,
                                             mom_indicator: str = 'roc', mom_window: int = DEFAULT_WINDOW_PHASE_PLOT,
                                             rev_indicator: str = 'rsi', rev_window: int = RSI_WINDOW) -> Optional[Dict[str, Any]]:
    """Creates a phase plot comparing a momentum and a mean reversion indicator."""
    if df is None or df.empty or 'close' not in df.columns:
         print("Warning: Insufficient data for phase plot.")
         return None

    df_analysis = df.copy()
    mom_col = f"{mom_indicator.upper()}_{mom_window}d"
    rev_col = f"{rev_indicator.upper()}_{rev_window}d"

    # --- Calculate Momentum Indicator ---
    if mom_indicator == 'roc':
        if len(df_analysis) > mom_window:
             roc_indicator = ta.momentum.ROCIndicator(close=df_analysis['close'], window=mom_window)
             df_analysis[mom_col] = roc_indicator.roc()
        else:
             print(f"Warning: Insufficient data for ROC({mom_window}) in phase plot.")
             return None # Cannot proceed without this indicator
    elif mom_indicator == 'macd':
        # Use the standard MACD parameters here for the phase plot
        if len(df_analysis) > MACD_SLOW + MACD_SIGNAL:
             macd_indicator = ta.trend.MACD(close=df_analysis['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
             df_analysis[mom_col] = macd_indicator.macd() # Using the MACD line itself
             mom_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}" # Update col name to reflect params
        else:
             print(f"Warning: Insufficient data for MACD in phase plot.")
             return None
    else:
         print(f"Warning: Unsupported momentum indicator '{mom_indicator}' for phase plot.")
         return None

    # --- Calculate Mean Reversion Indicator ---
    if rev_indicator == 'hurst':
        if len(df_analysis) > rev_window + 1:
            price_series = df_analysis['close'].values
            hurst_values = []
            valid_hurst_indices = []
            for i in range(rev_window, len(price_series)):
                segment = price_series[i-rev_window:i]
                if np.std(segment) < 1e-9: continue
                try:
                    H, _, _ = compute_Hc(segment, kind='price', simplified=True)
                    hurst_values.append(H)
                    valid_hurst_indices.append(pd.Timestamp(df_analysis.index[i]))
                except Exception:
                    continue
            if valid_hurst_indices:
                 df_analysis[rev_col] = pd.Series(hurst_values, index=valid_hurst_indices)
                 rev_col = f"Hurst_{rev_window}d" # Ensure col name reflects window
            else:
                 print(f"Warning: Hurst calculation failed for phase plot.")
                 return None
        else:
             print(f"Warning: Insufficient data for Hurst({rev_window}) in phase plot.")
             return None
    elif rev_indicator == 'rsi':
         # Use the provided rev_window for RSI
         if len(df_analysis) > rev_window + 1:
             rsi_indicator = ta.momentum.RSIIndicator(close=df_analysis['close'], window=rev_window)
             df_analysis[rev_col] = rsi_indicator.rsi()
             rev_col = f"RSI_{rev_window}d" # Update col name
         else:
              print(f"Warning: Insufficient data for RSI({rev_window}) in phase plot.")
              return None
    else:
         print(f"Warning: Unsupported mean reversion indicator '{rev_indicator}' for phase plot.")
         return None

    # --- Combine and Prepare Plot Data ---
    df_analysis = df_analysis[[mom_col, rev_col]].dropna()

    if df_analysis.empty:
        print("Warning: No overlapping data found for phase plot indicators.")
        return None
        
    # Ensure index is datetime before formatting for hover text
    datetime_index = pd.to_datetime(df_analysis.index)
    hover_text = datetime_index.strftime('%Y-%m-%d').tolist() # Use list of strings

    phase_plot = {
        "data": [
            {"type": "scatter",
             "x": df_analysis[mom_col].tolist(),
             "y": df_analysis[rev_col].tolist(),
             "mode": "markers",
             "marker": {
                 # Use DatetimeIndex directly if Plotly handles it, otherwise convert to something numeric
                 # Converting index to numerical representation for coloring (e.g., days since start)
                 "color": (datetime_index - datetime_index.min()).days,
                 "colorscale": "Viridis",
                 "showscale": True,
                 "colorbar": {"title": "Time"}
             },
             "text": hover_text, # Use the formatted date strings
             "name": f"{mom_col} vs {rev_col}"
            }
        ],
        "layout": {
            "title": f"Phase Plot: {mom_col} vs {rev_col}",
            "xaxis": {"title": f"Momentum ({mom_col})"},
            "yaxis": {"title": f"Mean Reversion ({rev_col})"}
        }
    }
    return json.loads(json.dumps(phase_plot, cls=PlotlyJSONEncoder))

# --- Aggregation Function ---
def run_all_momentum_mean_reversion_analyses(df: pd.DataFrame, params: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Runs all momentum and mean reversion analyses and collects the results."""
    
    results = {}
    
    # Parameter Extraction with Defaults
    lags = params.get('lags', DEFAULT_LAGS)
    rolling_window = params.get('rolling_window', ROLLING_WINDOW_ANALYSIS)
    hurst_window = params.get('hurst_window', HURST_WINDOW)
    rsi_window = params.get('rsi_window', RSI_WINDOW)
    macd_fast = params.get('macd_fast', MACD_FAST)
    macd_slow = params.get('macd_slow', MACD_SLOW)
    macd_signal = params.get('macd_signal', MACD_SIGNAL)
    bb_window = params.get('bb_window', BB_WINDOW)
    bb_std = params.get('bb_std', BB_STD)
    roc_windows = params.get('roc_windows', ROLLING_WINDOWS) # Expect list or int
    corr_lag = params.get('correlation_lag', 1)
    phase_mom_ind = params.get('phase_mom_indicator', 'roc')
    phase_mom_win = params.get('phase_mom_window', DEFAULT_WINDOW_PHASE_PLOT)
    phase_rev_ind = params.get('phase_rev_indicator', 'rsi')
    phase_rev_win = params.get('phase_rev_window', RSI_WINDOW)
    adf_window = params.get('adf_window', ROLLING_WINDOW_ANALYSIS)
    
    # --- Autocorrelation ---
    results['return_acf_plot'], results['return_pacf_plot'] = calculate_return_autocorrelation(df, nlags=lags)
    results['volatility_acf_plot'], results['volatility_pacf_plot'] = calculate_volatility_autocorrelation(df, nlags=lags)
    results['rolling_return_acf_heatmap'], results['rolling_return_pacf_heatmap'] = calculate_rolling_return_acf_heatmap(df, window=rolling_window, nlags=lags)
    results['rolling_vol_acf_heatmap'], results['rolling_vol_pacf_heatmap'] = calculate_rolling_vol_acf_heatmap(df, window=rolling_window, nlags=lags)
    
    # --- Momentum ---
    results['rolling_return_correlation_heatmap'] = calculate_rolling_return_correlation(df, window=rolling_window, lag=corr_lag)
    results['roc_plots'] = calculate_roc(df, windows=roc_windows)
    results['macd_plot'] = calculate_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    
    # --- Mean Reversion ---
    results['hurst_plot'] = calculate_rolling_hurst(df, window=hurst_window)
    results['rsi_plot'] = calculate_rsi(df, window=rsi_window)
    results['bollinger_bands_plots'] = calculate_bollinger_bands(df, window=bb_window, std_dev=bb_std)
    results['rolling_adf_plots'] = calculate_rolling_adf_test(df, window=adf_window)

    # --- Interaction --- 
    # Renaming keys slightly for clarity
    results['phase_plot_roc_rsi'] = calculate_momentum_reversion_phase_plot(df, 
                                                                    mom_indicator=phase_mom_ind, mom_window=phase_mom_win,
                                                                    rev_indicator=phase_rev_ind, rev_window=phase_rev_win)
    # Example of another phase plot combination if needed
    # results['phase_plot_macd_hurst'] = calculate_momentum_reversion_phase_plot(df, mom_indicator='macd', rev_indicator='hurst', mom_window=MACD_SLOW, rev_window=hurst_window)
    
    # Filter out None results before returning
    return {k: v for k, v in results.items() if v is not None}

# --- Momentum Indicators & Analysis ---

def calculate_rolling_return_correlation(df: pd.DataFrame, window: int = ROLLING_WINDOW_ANALYSIS, lag: int = 1) -> Optional[Dict[str, Any]]:
    """Calculates the rolling correlation between returns and lagged returns."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < window + lag + 2:
        print(f"Warning: Insufficient data for rolling return correlation.")
        return None
    returns = df['close'].pct_change()
    lagged_returns = returns.shift(lag)
    valid_data = pd.concat([returns, lagged_returns], axis=1).dropna()
    valid_data.columns = ['returns', 'lagged_returns']
    if len(valid_data) < window:
         print(f"Warning: Insufficient non-NaN data points ({len(valid_data)}) for rolling correlation with window {window}.")
         return None
    rolling_corr = valid_data['returns'].rolling(window=window).corr(valid_data['lagged_returns'])
    rolling_corr = rolling_corr.dropna()
    if rolling_corr.empty:
        print("Warning: Rolling correlation calculation resulted in empty series.")
        return None
    datetime_index = pd.to_datetime(rolling_corr.index)
    x = datetime_index.strftime('%Y-%m-%d').tolist()
    y = rolling_corr.values.tolist()
    plot_data = {
        "data": [
            {"type": "scatter", "mode": "lines", "x": x, "y": y, "name": f"Rolling {window}-Day Corr (Lag {lag})"}
        ],
        "layout": {
            "title": f"Rolling {window}-Day Return Correlation (Lag {lag})",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Correlation"},
            "yaxis_range": [-1, 1]
        }
    }
    return json.loads(json.dumps(plot_data, cls=PlotlyJSONEncoder))

def calculate_roc(df: pd.DataFrame, windows: Union[int, List[int]] = DEFAULT_WINDOW_MOMENTUM) -> Optional[Dict[str, Any]]:
    """Calculates the Rate of Change (ROC) indicator for one or multiple windows."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for ROC({windows}).")
        return None
        
    if isinstance(windows, int):
        windows = [windows]
        
    roc_values = {}
    for window in windows:
        roc_values[f"ROC_{window}d"] = df['close'].pct_change(window).dropna()
        
    if not roc_values:
        print("Warning: ROC calculation resulted in empty series.")
        return None
        
    # Ensure index is datetime for proper plotting
    datetime_index = pd.to_datetime(list(roc_values[list(roc_values.keys())[0]].index))
    x = datetime_index.strftime('%Y-%m-%d').tolist() # Dates
    
    roc_plot = {
        "data": [],
        "layout": {
            "title": f"Rate of Change (ROC) - Multiple Windows",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "ROC"}
        }
    }
    
    for window, values in roc_values.items():
        y = values.values.tolist()
        roc_plot["data"].append({"type": "scatter", "mode": "lines", "x": x, "y": y, "name": window})
        
    return json.loads(json.dumps(roc_plot, cls=PlotlyJSONEncoder))

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Optional[Dict[str, Any]]:
    """Calculates the Moving Average Convergence Divergence (MACD) indicator."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for MACD({fast}, {slow}, {signal}).")
        return None
    
    if len(df) < slow + signal: # Need enough data for calculation
        print(f"Warning: Not enough data points ({len(df)}) for MACD calculation (requires at least {slow + signal}).")
        return None

    try:
        macd_indicator = ta.trend.MACD(close=df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        
        macd_line = macd_indicator.macd()
        signal_line = macd_indicator.macd_signal()
        histogram = macd_indicator.macd_diff()
        
        # Create Plotly figure
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.7, 0.3])

        # Price Plot (optional, but good for context)
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'), row=1, col=1)

        # MACD Plot
        fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode='lines', name='MACD Line', line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode='lines', name='Signal Line', line=dict(color='orange')), row=2, col=1)
        
        # Histogram Colors
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(go.Bar(x=df.index, y=histogram.values, name='Histogram', marker_color=colors), row=2, col=1)

        fig.update_layout(
            title_text=f'MACD ({fast}, {slow}, {signal})',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis2_title='MACD',
            legend_title='Legend',
            xaxis_rangeslider_visible=False, # Typically off for indicator subplots
            height=500 # Adjust height as needed
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating MACD: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_rolling_hurst(df: pd.DataFrame, window: int = HURST_WINDOW) -> Optional[Dict[str, Any]]:
    """Calculates the Hurst Exponent using R/S analysis."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for Hurst({window}).")
        return None
        
    try:
        # Calculate Hurst exponent on the last 'window' data points
        series = df['close'].iloc[-window:].values
        H, c, data = compute_Hc(series, kind='price', simplified=True)
        return {'hurst': H}
    except Exception as e:
        print(f"Error calculating Hurst exponent: {e}")
        return None
        
def calculate_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> Optional[Dict[str, Any]]:
    """Calculates the Relative Strength Index (RSI) indicator."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for RSI({window}).")
        return None
        
    if len(df) < window + 1: # Need enough data for RSI calculation
        print(f"Warning: Not enough data points ({len(df)}) for RSI calculation (requires at least {window + 1}).")
        return None

    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=window)
        rsi_series = rsi_indicator.rsi()

        if rsi_series.isnull().all():
             print(f"Warning: RSI calculation resulted in all NaNs for window {window}.")
             return None

        # Create Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode='lines', name='RSI'))

        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)", annotation_position="bottom right")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)", annotation_position="bottom right")

        fig.update_layout(
            title=f'Relative Strength Index (RSI - {window})',
            xaxis_title='Date',
            yaxis_title='RSI',
            yaxis_range=[0, 100] # RSI is bounded between 0 and 100
        )

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None
        
def calculate_bollinger_bands(df: pd.DataFrame, window: int = BB_WINDOW, std_dev: int = BB_STD) -> Optional[Dict[str, Any]]:
    """Calculates the Bollinger Bands indicator."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for Bollinger Bands({window}, {std_dev}).")
        return None
        
    if len(df) < window: # Need enough data for the moving average
        print(f"Warning: Not enough data points ({len(df)}) for Bollinger Bands calculation (requires at least {window}).")
        return None

    try:
        bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=window, window_dev=std_dev)
        bb_high = bb_indicator.bollinger_hband()
        bb_low = bb_indicator.bollinger_lband()
        bb_mid = bb_indicator.bollinger_mavg()

        if bb_high.isnull().all() or bb_low.isnull().all() or bb_mid.isnull().all():
             print(f"Warning: Bollinger Bands calculation resulted in all NaNs for window {window}, std_dev {std_dev}.")
             return None

        # Create Plotly figure
        fig = go.Figure()

        # Add Bands
        fig.add_trace(go.Scatter(x=df.index, y=bb_high, mode='lines', line=dict(width=0), showlegend=False, name='High Band')) # High band fill area
        fig.add_trace(go.Scatter(x=df.index, y=bb_low, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,80,0.2)', showlegend=False, name='Low Band')) # Low band fill area
        fig.add_trace(go.Scatter(x=df.index, y=bb_mid, mode='lines', line=dict(color='rgba(0,100,80,0.8)'), name='Middle Band'))
        fig.add_trace(go.Scatter(x=df.index, y=bb_high, mode='lines', line=dict(color='rgba(0,100,80,0.5)'), name='Upper Band')) # Upper band line
        fig.add_trace(go.Scatter(x=df.index, y=bb_low, mode='lines', line=dict(color='rgba(0,100,80,0.5)'), name='Lower Band')) # Lower band line

        # Add Price
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', line=dict(color='rgba(0,0,200,0.8)'), name='Close Price'))

        fig.update_layout(
            title=f'Bollinger Bands ({window}, {std_dev})',
            xaxis_title='Date',
            yaxis_title='Price',
            legend_title='Legend'
        )

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return None
        
def calculate_rolling_adf_test(df: pd.DataFrame, window: int = ROLLING_WINDOW_ANALYSIS) -> Optional[Dict[str, Any]]:
    """Calculates the Augmented Dickey-Fuller (ADF) test for stationarity on the latest window."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for ADF test({window}).")
        return None
        
    try:
        # Perform ADF test on the last 'window' data points
        series = df['close'].iloc[-window:].values
        adf_result = adfuller(series)
        return {'adf_stat': adf_result[0], 'adf_p_value': adf_result[1]}
    except Exception as e:
        # statsmodels can raise errors like LinAlgError for certain inputs
        print(f"Error calculating ADF test: {e}")
        return None