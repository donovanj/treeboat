import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
import ta
from hurst import compute_Hc # Assuming 'hurst' library is installed
import plotly.graph_objs as go
import json
from plotly.utils import PlotlyJSONEncoder
from typing import Tuple, Dict, Any, Optional, List

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

# --- Helper Functions ---

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
        ci_upper_acf = confint_acf[:, 1] - acf_values
        ci_lower_acf = acf_values - confint_acf[:, 0]
        ci_upper_pacf = confint_pacf[:, 1] - pacf_values
        ci_lower_pacf = pacf_values - confint_pacf[:, 0]

        lags = list(range(len(acf_values)))

        acf_plot = {
            "data": [
                {"type": "bar", "x": lags, "y": acf_values.tolist(), "name": "ACF"},
                # Confidence Interval Lines (upper and lower bounds)
                {"type": "scatter", "x": lags, "y": ci_upper_acf.tolist(), "mode": "lines", "line": {"dash": "dash", "color": "rgba(0,0,0,0)"}, "name": "Upper CI", "showlegend": False},
                {"type": "scatter", "x": lags, "y": ci_lower_acf.tolist(), "mode": "lines", "line": {"dash": "dash", "color": "rgba(0,0,0,0)"}, "fill": "tonexty", "fillcolor": "rgba(255, 0, 0, 0.2)", "name": "Lower CI", "showlegend": False}
            ],
            "layout": {"title": f"Autocorrelation Function (ACF) - {title_suffix}", "xaxis": {"title": "Lag"}, "yaxis": {"title": "ACF"}}
        }
        pacf_plot = {
             "data": [
                {"type": "bar", "x": lags[1:], "y": pacf_values[1:].tolist(), "name": "PACF"}, # Skip lag 0 for PACF
                # Confidence Interval Lines
                {"type": "scatter", "x": lags[1:], "y": ci_upper_pacf[1:].tolist(), "mode": "lines", "line": {"dash": "dash", "color": "rgba(0,0,0,0)"}, "name": "Upper CI", "showlegend": False},
                {"type": "scatter", "x": lags[1:], "y": ci_lower_pacf[1:].tolist(), "mode": "lines", "line": {"dash": "dash", "color": "rgba(0,0,0,0)"}, "fill": "tonexty", "fillcolor": "rgba(255, 0, 0, 0.2)", "name": "Lower CI", "showlegend": False}
            ],
            "layout": {"title": f"Partial Autocorrelation Function (PACF) - {title_suffix}", "xaxis": {"title": "Lag"}, "yaxis": {"title": "PACF"}}
        }
        return json.loads(json.dumps(acf_plot, cls=PlotlyJSONEncoder)), json.loads(json.dumps(pacf_plot, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating ACF/PACF for {title_suffix}: {e}")
        return None, None

def _create_heatmap(z_data: np.ndarray, x_labels: List[str], y_labels: List[str], title: str, colorscale: str = "Viridis", zmid: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Helper function to create Plotly heatmap data."""
    if z_data is None or not x_labels or not y_labels:
        print(f"Warning: Insufficient data for heatmap: {title}")
        return None
        
    heatmap_layout = {
        "title": title,
        "xaxis": {"title": "Date (End of Window)"},
        "yaxis": {"title": "Lag"}
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
    """Calculates ACF and PACF for absolute/squared daily returns (volatility proxy)."""
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
            # Calculate ACF/PACF for the segment
            acf_vals, _ = acf(segment, nlags=nlags, alpha=0.05, fft=False)
            pacf_vals, _ = pacf(segment, nlags=nlags, alpha=0.05, method='ols')
            
            all_acf_values.append(acf_vals)
            all_pacf_values.append(pacf_vals)
            valid_indices.append(series.index[i]) # Index at the end of the window
        except Exception as e:
            # print(f"Skipping segment ending at {series.index[i]} due to ACF/PACF error: {e}")
            continue # Skip segment if calculation fails

    if not all_acf_values or not valid_indices:
        print(f"Warning: No valid segments found for rolling ACF/PACF ({title_suffix}).")
        return None, None

    # Transpose for heatmap: rows=lags, cols=time
    acf_matrix = np.array(all_acf_values).T
    pacf_matrix = np.array(all_pacf_values).T
    
    dates_str = [d.strftime('%Y-%m-%d') for d in valid_indices]
    lags_list = list(range(nlags + 1))

    acf_heatmap = _create_heatmap(acf_matrix, dates_str, lags_list, f"Rolling ACF Heatmap - {title_suffix}", colorscale="RdBu", zmid=0)
    pacf_heatmap = _create_heatmap(pacf_matrix[1:,:], dates_str, lags_list[1:], f"Rolling PACF Heatmap - {title_suffix}", colorscale="RdBu", zmid=0) # Skip lag 0 for PACF

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

# --- Momentum Indicators & Analysis ---

def calculate_rolling_return_correlation(df: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS) -> Optional[Dict[str, Any]]:
    """Calculates rolling correlation between past and future returns."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < max(windows) * 2 + 1: # Need enough data
        print(f"Warning: Insufficient data for rolling return correlation.")
        return None

    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df.dropna(subset=['returns'], inplace=True)
    if df.empty:
        print(f"Warning: No returns calculable for rolling correlation.")
        return None

    correlation_data = pd.DataFrame(index=df.index)
    
    min_required_len = 0
    for lookback in windows:
        for lookforward in windows:
            # Find the minimum length needed for this specific calculation
            required_len = lookback + lookforward + 1 # +1 for the current point
            min_required_len = max(min_required_len, required_len)
            
            if len(df) < required_len:
                print(f"Skipping {lookback}d past vs {lookforward}d future due to insufficient data.")
                continue
            
            past_ret_col = f'past_{lookback}d_ret'
            future_ret_col = f'future_{lookforward}d_ret'
            
            # Calculate cumulative return over the past 'lookback' days (excluding current day)
            df[past_ret_col] = (1 + df['returns'].shift(1)).rolling(window=lookback).apply(np.prod, raw=True) - 1
            
            # Calculate cumulative return over the next 'lookforward' days (excluding current day)
            # Shift returns forward, then apply rolling prod looking back
            df[future_ret_col] = (1 + df['returns'].shift(-lookforward)).rolling(window=lookforward).apply(np.prod, raw=True) - 1
            
            # Calculate rolling correlation between the past and future returns
            # The correlation window should align points in time correctly
            # We need to align the past return ending at t-1 with the future return starting at t+1
            # This requires careful shifting. Let's align based on the *end* of the future window.
            col_name = f'{lookback}d_vs_{lookforward}d'
            
            # Correlation window size: Using 'lookback' seems reasonable to capture the relationship's stability
            # The rolling correlation takes two Series. We need to align them.
            # Past return (ending t-1) vs Future return (starting t+1, ending t+lookforward)
            # Aligning based on time 't'
            past_shifted = df[past_ret_col] # Represents return up to t-1
            future_shifted = df[future_ret_col] # Represents return from t+1 to t+lookforward

            # The rolling window should apply to the *pair* of observations
            # Use a reasonable window for the correlation itself, e.g., 60 days
            corr_window = 60 
            if len(df) >= corr_window + max(lookback, lookforward): # Check sufficient data for rolling corr
                 correlation_data[col_name] = past_shifted.rolling(corr_window).corr(future_shifted)
            else:
                 print(f"Skipping rolling correlation for {col_name} due to insufficient data for correlation window.")
                 correlation_data[col_name] = np.nan # Fill with NaN if not enough data


    correlation_data.dropna(how='all', axis=1, inplace=True) # Drop columns that are all NaN
    correlation_data.dropna(how='all', axis=0, inplace=True) # Drop rows that are all NaN

    if correlation_data.empty:
        print(f"Warning: Rolling correlation calculation resulted in empty data.")
        return None

    # Prepare for heatmap
    z = correlation_data.values.T # Transpose for heatmap (windows on axes)
    x = correlation_data.index.strftime('%Y-%m-%d').tolist() # Dates
    y = correlation_data.columns.tolist() # Window pairs

    heatmap = {
        "data": [
            {"type": "heatmap", "z": z.tolist(), "x": x, "y": y, "colorscale": "RdBu", "zmid": 0}
        ],
        "layout": {"title": "Rolling Correlation: Past vs. Future Returns", "xaxis": {"title": "Date"}, "yaxis": {"title": "Past Lookback vs Future Lookforward"}}
    }
    return json.loads(json.dumps(heatmap, cls=PlotlyJSONEncoder))
    
def calculate_roc(df: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS) -> Optional[Dict[str, Any]]:
    """Calculates Rate of Change (ROC) for different lookback periods."""
    if df is None or df.empty or 'close' not in df.columns:
        print(f"Warning: Insufficient data for ROC calculation.")
        return None
        
    roc_data = {}
    df_roc = pd.DataFrame(index=df.index)
    
    for window in windows:
        if len(df) > window:
            roc_indicator = ta.momentum.ROCIndicator(close=df['close'], window=window)
            col_name = f'ROC_{window}d'
            df_roc[col_name] = roc_indicator.roc()
        else:
            print(f"Warning: Insufficient data for ROC window {window}.")

    df_roc.dropna(how='all', inplace=True)
    if df_roc.empty:
        print(f"Warning: ROC calculation resulted in empty data.")
        return None

    roc_plots = {}
    for col in df_roc.columns:
        # Histogram for distribution
        hist = {
            "data": [{"type": "histogram", "x": df_roc[col].dropna().tolist(), "name": col}],
            "layout": {"title": f"Distribution of {col}", "xaxis": {"title": "ROC Value"}, "yaxis": {"title": "Frequency"}}
        }
        roc_plots[f'{col}_hist'] = json.loads(json.dumps(hist, cls=PlotlyJSONEncoder))
        
        # Line plot over time
        line = {
            "data": [{"type": "scatter", "x": df_roc.index.strftime('%Y-%m-%d').tolist(), "y": df_roc[col].tolist(), "mode": "lines", "name": col}],
            "layout": {"title": f"{col} Over Time", "xaxis": {"title": "Date"}, "yaxis": {"title": "ROC Value"}}
        }
        roc_plots[f'{col}_line'] = json.loads(json.dumps(line, cls=PlotlyJSONEncoder))

    return roc_plots # Return dict of plot data

def calculate_macd(df: pd.DataFrame, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Optional[Dict[str, Any]]:
    """Calculates MACD indicator."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < slow + signal:
        print(f"Warning: Insufficient data for MACD calculation.")
        return None

    try:
        macd_indicator = ta.trend.MACD(close=df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        df_macd = pd.DataFrame(index=df.index)
        df_macd['MACD'] = macd_indicator.macd()
        df_macd['Signal'] = macd_indicator.macd_signal()
        df_macd['Histogram'] = macd_indicator.macd_diff() # Histogram = MACD - Signal
        df_macd.dropna(inplace=True)
        
        if df_macd.empty:
            print(f"Warning: MACD calculation resulted in empty data.")
            return None

        plot = {
            "data": [
                {"type": "scatter", "x": df_macd.index.strftime('%Y-%m-%d').tolist(), "y": df_macd['MACD'].tolist(), "mode": "lines", "name": "MACD", "line": {"color": "blue"}},
                {"type": "scatter", "x": df_macd.index.strftime('%Y-%m-%d').tolist(), "y": df_macd['Signal'].tolist(), "mode": "lines", "name": "Signal Line", "line": {"color": "red"}},
                {"type": "bar", "x": df_macd.index.strftime('%Y-%m-%d').tolist(), "y": df_macd['Histogram'].tolist(), "name": "Histogram", "marker": {"color": df_macd['Histogram'].apply(lambda x: 'green' if x >= 0 else 'orange')}} # Color bars based on sign
            ],
            "layout": {"title": "MACD", "xaxis": {"title": "Date"}, "yaxis": {"title": "Value"}}
        }
        return json.loads(json.dumps(plot, cls=PlotlyJSONEncoder))
        
    except Exception as e:
        print(f"Error calculating MACD: {e}")
        return None

# --- Mean Reversion Indicators & Analysis ---

def calculate_rolling_hurst(df: pd.DataFrame, window: int = HURST_WINDOW) -> Optional[Dict[str, Any]]:
    """Calculates the rolling Hurst exponent."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < window + 1:
        print(f"Warning: Insufficient data for rolling Hurst calculation (need {window+1}, have {len(df)}).")
        return None

    try:
        # The compute_Hc function expects a numpy array or list
        price_series = df['close'].values
        
        # Calculate rolling Hurst Exponent
        hurst_values = []
        indices = []
        for i in range(window, len(price_series)):
             segment = price_series[i-window:i]
             # compute_Hc returns H, c, data - we only need H
             H, _, _ = compute_Hc(segment, kind='price', simplified=True)
             hurst_values.append(H)
             indices.append(df.index[i]) # Store the index corresponding to the end of the window

        if not hurst_values:
             print(f"Warning: Rolling Hurst calculation produced no values.")
             return None

        df_hurst = pd.Series(hurst_values, index=indices)

        plot = {
            "data": [
                {"type": "scatter", "x": df_hurst.index.strftime('%Y-%m-%d').tolist(), "y": df_hurst.tolist(), "mode": "lines", "name": "Hurst Exponent"},
                # Add line at 0.5 for reference
                {"type": "scatter", "x": df_hurst.index.strftime('%Y-%m-%d').tolist(), "y": [0.5]*len(df_hurst), "mode": "lines", "name": "0.5 (Random Walk)", "line": {"dash": "dash", "color": "red"}}
            ],
            "layout": {"title": f"Rolling Hurst Exponent ({window}-day window)", "xaxis": {"title": "Date"}, "yaxis": {"title": "Hurst Exponent (H)"}}
        }
        return json.loads(json.dumps(plot, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating rolling Hurst exponent: {e}")
        # Consider more specific error handling if needed
        if "Input array must be longer than subsequence length" in str(e):
             print(f"Hurst Error Detail: Input segment likely too short or lacks variance.")
        return None

def calculate_rsi(df: pd.DataFrame, window: int = RSI_WINDOW) -> Optional[Dict[str, Any]]:
    """Calculates the Relative Strength Index (RSI)."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < window + 1:
        print(f"Warning: Insufficient data for RSI calculation.")
        return None

    try:
        rsi_indicator = ta.momentum.RSIIndicator(close=df['close'], window=window)
        df_rsi = pd.DataFrame(index=df.index)
        df_rsi['RSI'] = rsi_indicator.rsi()
        df_rsi.dropna(inplace=True)

        if df_rsi.empty:
            print(f"Warning: RSI calculation resulted in empty data.")
            return None

        plot = {
            "data": [
                {"type": "scatter", "x": df_rsi.index.strftime('%Y-%m-%d').tolist(), "y": df_rsi['RSI'].tolist(), "mode": "lines", "name": f"RSI({window})"},
                # Overbought/Oversold lines
                {"type": "scatter", "x": df_rsi.index.strftime('%Y-%m-%d').tolist(), "y": [70]*len(df_rsi), "mode": "lines", "name": "Overbought (70)", "line": {"dash": "dash", "color": "red"}, "showlegend": False},
                 {"type": "scatter", "x": df_rsi.index.strftime('%Y-%m-%d').tolist(), "y": [30]*len(df_rsi), "mode": "lines", "name": "Oversold (30)", "line": {"dash": "dash", "color": "green"}, "showlegend": False}
            ],
            "layout": {"title": f"Relative Strength Index (RSI - {window} days)", "xaxis": {"title": "Date"}, "yaxis": {"title": "RSI", "range": [0, 100]}}
        }
        return json.loads(json.dumps(plot, cls=PlotlyJSONEncoder))

    except Exception as e:
        print(f"Error calculating RSI: {e}")
        return None

def calculate_bollinger_bands(df: pd.DataFrame, window: int = BB_WINDOW, std_dev: int = BB_STD) -> Optional[Dict[str, Any]]:
    """Calculates Bollinger Bands."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < window:
        print(f"Warning: Insufficient data for Bollinger Bands calculation.")
        return None
        
    try:
        bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=window, window_dev=std_dev)
        df_bb = pd.DataFrame(index=df.index)
        df_bb['Middle'] = bb_indicator.bollinger_mavg()
        df_bb['Upper'] = bb_indicator.bollinger_hband()
        df_bb['Lower'] = bb_indicator.bollinger_lband()
        df_bb['%B'] = bb_indicator.bollinger_pband() # %B indicator
        df_bb['Width'] = bb_indicator.bollinger_wband() # Bandwidth indicator
        
        # Include close price for context
        df_bb['Close'] = df['close']
        
        df_bb.dropna(inplace=True)

        if df_bb.empty:
             print(f"Warning: Bollinger Bands calculation resulted in empty data.")
             return None

        # Candlestick with Bands Plot
        bb_plot = {
            "data": [
                 # Candlestick (using the reduced date range from df_bb)
                 {"type": "candlestick", 
                  "x": df_bb.index.strftime('%Y-%m-%d').tolist(), 
                  "open": df.loc[df_bb.index, 'open'].tolist(), # Align original OHLC data
                  "high": df.loc[df_bb.index, 'high'].tolist(),
                  "low": df.loc[df_bb.index, 'low'].tolist(),
                  "close": df_bb['Close'].tolist(), 
                  "name": "Price", 
                  "showlegend": False},
                # Bollinger Bands
                {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": df_bb['Upper'].tolist(), "mode": "lines", "name": "Upper Band", "line": {"color": "rgba(255, 165, 0, 0.7)"}}, # Orange
                {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": df_bb['Middle'].tolist(), "mode": "lines", "name": "Middle Band (SMA)", "line": {"color": "rgba(0, 0, 255, 0.7)", "dash": "dash"}}, # Blue dashed
                {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": df_bb['Lower'].tolist(), "mode": "lines", "name": "Lower Band", "line": {"color": "rgba(255, 165, 0, 0.7)"}, "fill": "tonexty", "fillcolor": "rgba(255, 165, 0, 0.1)"}, # Fill between lower and middle (or upper?) check plotly docs
            ],
            "layout": {"title": f"Bollinger Bands ({window}, {std_dev})", "xaxis": {"title": "Date", "rangeslider": {"visible": False}}, "yaxis": {"title": "Price"}}
        }
        
        # %B Indicator Plot
        percent_b_plot = {
            "data": [
                {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": df_bb['%B'].tolist(), "mode": "lines", "name": "%B"},
                # Lines at 0 and 1
                 {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": [1]*len(df_bb), "mode": "lines", "name": "Upper Band (1)", "line": {"dash": "dash", "color": "red"}, "showlegend": False},
                 {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": [0]*len(df_bb), "mode": "lines", "name": "Lower Band (0)", "line": {"dash": "dash", "color": "green"}, "showlegend": False}
            ],
             "layout": {"title": f"Bollinger Bands %B ({window}, {std_dev})", "xaxis": {"title": "Date"}, "yaxis": {"title": "%B", "range": [df_bb['%B'].min() - 0.1, df_bb['%B'].max() + 0.1]}} # Dynamic range plus padding
        }
        
        # Bandwidth Plot
        bandwidth_plot = {
             "data": [
                {"type": "scatter", "x": df_bb.index.strftime('%Y-%m-%d').tolist(), "y": df_bb['Width'].tolist(), "mode": "lines", "name": "Bandwidth"}
            ],
             "layout": {"title": f"Bollinger Bands Width ({window}, {std_dev})", "xaxis": {"title": "Date"}, "yaxis": {"title": "Bandwidth"}}
        }

        return {
            "bollinger_bands_plot": json.loads(json.dumps(bb_plot, cls=PlotlyJSONEncoder)),
            "percent_b_plot": json.loads(json.dumps(percent_b_plot, cls=PlotlyJSONEncoder)),
            "bandwidth_plot": json.loads(json.dumps(bandwidth_plot, cls=PlotlyJSONEncoder)),
        }

    except Exception as e:
        print(f"Error calculating Bollinger Bands: {e}")
        return None
        
def calculate_rolling_adf_test(df: pd.DataFrame, window: int = ROLLING_WINDOW_ANALYSIS) -> Optional[Dict[str, Any]]:
    """Calculates rolling Augmented Dickey-Fuller test statistics and p-values."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < window + 1:
        print(f"Warning: Insufficient data for rolling ADF test (need {window+1}, have {len(df)})." )
        return None

    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("Error: statsmodels.tsa.stattools.adfuller not found. Please ensure statsmodels is installed correctly.")
        return None
        
    adf_stats = []
    p_values = []
    valid_indices = []

    price_series = df['close']

    for i in range(window, len(price_series)):
        segment = price_series.iloc[i-window:i]
        if segment.var() < 1e-10: # Check for variance
             continue
             
        try:
            # Perform ADF test on the price segment
            # Using 'AIC' for automatic lag selection
            result = adfuller(segment, autolag='AIC', regression='c') # 'c' includes constant (trend)
            adf_stats.append(result[0]) # ADF statistic
            p_values.append(result[1]) # p-value
            valid_indices.append(price_series.index[i])
        except Exception as e:
             # print(f"Skipping segment ending at {price_series.index[i]} due to ADF error: {e}")
             continue # Skip segment if ADF test fails

    if not valid_indices:
        print("Warning: Rolling ADF calculation produced no valid results.")
        return None

    df_adf = pd.DataFrame({'ADF_Statistic': adf_stats, 'P_Value': p_values}, index=valid_indices)

    # Plot ADF Statistic
    adf_stat_plot = {
        "data": [
            {"type": "scatter", "x": df_adf.index.strftime('%Y-%m-%d').tolist(), "y": df_adf['ADF_Statistic'].tolist(), "mode": "lines", "name": "ADF Statistic"}
            # Add critical values if needed (e.g., result[4]['5%']) - requires storing them
        ],
        "layout": {"title": f"Rolling ADF Statistic ({window}-day window)", "xaxis": {"title": "Date (End of Window)"}, "yaxis": {"title": "ADF Statistic"}}
    }
    
    # Plot P-Value
    p_value_plot = {
        "data": [
             {"type": "scatter", "x": df_adf.index.strftime('%Y-%m-%d').tolist(), "y": df_adf['P_Value'].tolist(), "mode": "lines", "name": "P-Value"},
             # Add significance level line
             {"type": "scatter", "x": df_adf.index.strftime('%Y-%m-%d').tolist(), "y": [0.05]*len(df_adf), "mode": "lines", "name": "5% Significance", "line": {"dash": "dash", "color": "red"}, "showlegend": False}
        ],
        "layout": {"title": f"Rolling ADF P-Value ({window}-day window)", "xaxis": {"title": "Date (End of Window)"}, "yaxis": {"title": "P-Value", "range": [0, max(1.0, df_adf['P_Value'].max() * 1.1)]}} # Adjust range for visibility
    }
    
    try:
        return {
            "adf_statistic_plot": json.loads(json.dumps(adf_stat_plot, cls=PlotlyJSONEncoder)),
            "adf_p_value_plot": json.loads(json.dumps(p_value_plot, cls=PlotlyJSONEncoder))
        }
    except Exception as e:
        print(f"Error during rolling ADF calculation or JSON serialization: {e}")
        return None

# --- Momentum/Reversion Interaction Analysis ---

def calculate_momentum_reversion_phase_plot(df: pd.DataFrame, 
                                             mom_indicator: str = 'roc', mom_window: int = 20, 
                                             rev_indicator: str = 'hurst', rev_window: int = HURST_WINDOW) -> Optional[Dict[str, Any]]:
    """Creates a phase plot comparing a momentum and a mean reversion indicator."""
    if df is None or df.empty or 'close' not in df.columns:
         print("Warning: Insufficient data for phase plot.")
         return None

    df_analysis = df.copy()
    mom_col = f"{mom_indicator.upper()}_{mom_window}d"
    rev_col = f"{rev_indicator.upper()}_{rev_window}d"

    # Calculate Momentum Indicator
    if mom_indicator == 'roc':
        if len(df_analysis) > mom_window:
             roc = ta.momentum.ROCIndicator(close=df_analysis['close'], window=mom_window).roc()
             df_analysis[mom_col] = roc
        else:
             print(f"Warning: Insufficient data for ROC({mom_window}) in phase plot.")
             return None
    elif mom_indicator == 'macd':
        if len(df_analysis) > MACD_SLOW + MACD_SIGNAL: # Use default MACD params for simplicity here
             macd = ta.trend.MACD(close=df_analysis['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL).macd()
             df_analysis[mom_col] = macd
             mom_col = f"MACD_{MACD_FAST}_{MACD_SLOW}" # Update col name
        else:
             print(f"Warning: Insufficient data for MACD in phase plot.")
             return None
    else:
         print(f"Warning: Unsupported momentum indicator '{mom_indicator}' for phase plot.")
         return None

    # Calculate Mean Reversion Indicator
    if rev_indicator == 'hurst':
        if len(df_analysis) > rev_window + 1:
             price_series = df_analysis['close'].values
             hurst_values = []
             valid_hurst_indices = []
             for i in range(rev_window, len(price_series)):
                 segment = price_series[i-rev_window:i]
                 try:
                     H, _, _ = compute_Hc(segment, kind='price', simplified=True)
                     hurst_values.append(H)
                     valid_hurst_indices.append(df_analysis.index[i])
                 except Exception:
                     continue # Skip if Hurst calc fails
             if valid_hurst_indices:
                  df_analysis[rev_col] = pd.Series(hurst_values, index=valid_hurst_indices)
             else:
                  print(f"Warning: Hurst calculation failed for phase plot.")
                  return None
        else:
             print(f"Warning: Insufficient data for Hurst({rev_window}) in phase plot.")
             return None
    elif rev_indicator == 'rsi':
         if len(df_analysis) > RSI_WINDOW + 1: # Use default RSI window
             rsi = ta.momentum.RSIIndicator(close=df_analysis['close'], window=RSI_WINDOW).rsi()
             df_analysis[rev_col] = rsi
             rev_col = f"RSI_{RSI_WINDOW}" # Update col name
         else:
              print(f"Warning: Insufficient data for RSI in phase plot.")
              return None
    else:
         print(f"Warning: Unsupported mean reversion indicator '{rev_indicator}' for phase plot.")
         return None

    # Combine and drop NaNs
    df_analysis = df_analysis[[mom_col, rev_col]].dropna()

    if df_analysis.empty:
        print("Warning: No overlapping data found for phase plot indicators.")
        return None

    # Create Scatter Plot (Phase Diagram)
    plot = {
        "data": [
            {
                "type": "scatter",
                "x": df_analysis[mom_col].tolist(),
                "y": df_analysis[rev_col].tolist(),
                "mode": "markers", # Could use 'lines+markers' but might be messy
                "marker": {
                    "size": 5,
                    "color": np.arange(len(df_analysis)), # Color by time sequence
                    "colorscale": "Viridis",
                    "colorbar": {"title": "Time Progression"}
                },
                "text": df_analysis.index.strftime('%Y-%m-%d'), # Show date on hover
                "name": "State"
            }
        ],
        "layout": {
            "title": f"Momentum ({mom_col}) vs. Mean Reversion ({rev_col}) Phase Plot",
            "xaxis": {"title": f"Momentum Indicator ({mom_col})"},
            "yaxis": {"title": f"Mean Reversion Indicator ({rev_col})"},
            "hovermode": "closest"
        }
    }
    
    return json.loads(json.dumps(plot, cls=PlotlyJSONEncoder))

# --- Aggregation Function ---

def run_all_momentum_mean_reversion_analyses(df: pd.DataFrame, symbol: str) -> dict:
    """Runs all momentum and mean reversion analyses and returns results."""
    results = {}

    # Autocorrelation
    acf_ret, pacf_ret = calculate_return_autocorrelation(df)
    results['return_acf_plot'] = acf_ret
    results['return_pacf_plot'] = pacf_ret

    acf_vol, pacf_vol = calculate_volatility_autocorrelation(df)
    results['volatility_acf_plot'] = acf_vol
    results['volatility_pacf_plot'] = pacf_vol
    
    # Rolling ACF Heatmaps
    acf_ret_hm, pacf_ret_hm = calculate_rolling_return_acf_heatmap(df)
    results['rolling_return_acf_heatmap'] = acf_ret_hm
    results['rolling_return_pacf_heatmap'] = pacf_ret_hm
    
    acf_vol_hm, pacf_vol_hm = calculate_rolling_vol_acf_heatmap(df)
    results['rolling_volatility_acf_heatmap'] = acf_vol_hm
    results['rolling_volatility_pacf_heatmap'] = pacf_vol_hm

    # Momentum
    results['rolling_return_correlation_heatmap'] = calculate_rolling_return_correlation(df)
    results['roc_plots'] = calculate_roc(df) # Returns a dict of plots
    results['macd_plot'] = calculate_macd(df)

    # Mean Reversion
    results['hurst_plot'] = calculate_rolling_hurst(df)
    results['rsi_plot'] = calculate_rsi(df)
    results['bollinger_bands_plots'] = calculate_bollinger_bands(df) # Returns a dict of plots
    results['rolling_adf_plots'] = calculate_rolling_adf_test(df) # Returns a dict of plots

    # Interaction Analysis
    results['phase_plot_roc_hurst'] = calculate_momentum_reversion_phase_plot(df, mom_indicator='roc', rev_indicator='hurst')
    results['phase_plot_macd_rsi'] = calculate_momentum_reversion_phase_plot(df, mom_indicator='macd', rev_indicator='rsi')

    # --- Placeholder for more advanced analyses ---
    # Timeframe Transition Analysis (Requires more complex logic, potentially combining indicators)
    # Regime-Based Signal Strength (Requires defining regimes first, e.g., based on volatility)
    
    print(f"Completed Momentum/Mean Reversion Analysis for {symbol}")
    return results

# --- FastAPI Router ---
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ...dependencies import get_db # Adjusted path: up from eda_calculations -> routes -> api
from .base_data import prepare_data_for_analysis

router = APIRouter()

@router.post("/momentum_mean_reversion/", 
             tags=["EDA Calculations"], 
             summary="Calculate Momentum and Mean Reversion metrics",
             response_description="Dictionary of Plotly JSON plots and analysis results")
async def get_momentum_mean_reversion_analysis(
    symbol: str = Query(default="AAPL", description="Stock symbol (e.g., AAPL)"),
    start_date: Optional[str] = Query(default=None, description="Start date in YYYY-MM-DD format. Defaults to 1 year ago."),
    end_date: Optional[str] = Query(default=None, description="End date in YYYY-MM-DD format. Defaults to today."),
    db: Session = Depends(get_db)
):
    """
    Perform Momentum and Mean Reversion analysis on historical stock data.

    Calculates:
    - **Autocorrelation:** ACF/PACF plots for returns and absolute returns.
    - **Rolling Autocorrelation:** Heatmaps of rolling ACF/PACF for returns and absolute returns.
    - **Momentum Indicators:** Rolling return correlation heatmap, ROC plots, MACD plot.
    - **Mean Reversion Indicators:** Rolling Hurst exponent plot, RSI plot, Bollinger Bands plots.
    - **Statistical Tests:** Rolling Augmented Dickey-Fuller (ADF) test plots (statistic and p-value).
    - **Interaction:** Phase plots (ROC vs. Hurst, MACD vs. RSI).
    """
    symbol, start, end, stock_df, _, _ = prepare_data_for_analysis(db, symbol, start_date, end_date)

    if stock_df is None or stock_df.empty:
        raise HTTPException(status_code=404, detail=f"Could not fetch or prepare sufficient data for symbol '{symbol}' between {start} and {end}.")

    # Pass the dataframe with OHLCV data (stock_df)
    analysis_results = run_all_momentum_mean_reversion_analyses(stock_df, symbol)

    # Filter out None results before returning to avoid clutter/errors in frontend
    filtered_results = {k: v for k, v in analysis_results.items() if v is not None}
    
    # Further filter sub-dictionaries (like roc_plots, bollinger_bands_plots)
    for key in list(filtered_results.keys()):
        if isinstance(filtered_results[key], dict):
            filtered_results[key] = {sub_k: sub_v for sub_k, sub_v in filtered_results[key].items() if sub_v is not None}
            if not filtered_results[key]: # Remove empty dicts after filtering
                 del filtered_results[key]

    if not filtered_results:
         # Use 500 as it indicates an analysis failure despite having initial data
         raise HTTPException(status_code=500, detail=f"Momentum/Mean Reversion analysis failed to produce any results for {symbol} (potentially due to insufficient data length for rolling windows). Check logs.")

    return filtered_results

# Note: The Timeframe Transition and Regime-Based analyses require more complex logic
# and potentially inputs from other analysis modules (like volatility regimes).
# These are left as placeholders for future implementation.
# Also, the momentum portfolio sorting is more involved and might be better suited
# for a dedicated backtesting or strategy evaluation module rather than basic EDA plotting.
# Advanced techniques like Wavelets, DTW, Variance Ratio, CUSUM, Markov Models etc. are deferred.
