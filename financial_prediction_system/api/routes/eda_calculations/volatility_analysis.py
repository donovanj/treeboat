# Placeholder for volatility analysis calculations 

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
from plotly.utils import PlotlyJSONEncoder
import logging # Import logging

# Get the logger instance
logger = logging.getLogger("financial_prediction_system")

# Check if statsmodels is available for optional features
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    STATSMODELS_LOWESS_AVAILABLE = True
except ImportError:
    STATSMODELS_LOWESS_AVAILABLE = False
    # logger.warning("statsmodels not found. LOWESS smoothing will be unavailable.") # Logged inside function now
    def sm_lowess(*args, **kwargs): # Dummy function if statsmodels is not installed
        raise NotImplementedError("LOWESS requires statsmodels to be installed.")

# --- LOWESS Helper Function (Revised V2) ---
def _add_lowess_trace(fig, x_data, y_data, name_suffix=" LOWESS", color=None, frac=0.25, row=None, col=None, **kwargs):
    """Calculates LOWESS and adds it as a trace to the figure."""
    if not STATSMODELS_LOWESS_AVAILABLE:
        # logger.debug("Statsmodels not available. Skipping LOWESS.") # Optional: log skipping
        return

    # Ensure y_data is a Series
    if not isinstance(y_data, pd.Series):
        try:
            y_series = pd.Series(y_data)
        except Exception as e:
             logger.warning(f"Could not convert y_data to Series for LOWESS {name_suffix}: {e}. Skipping.")
             return
    else:
        y_series = y_data

    # --- Handle x_data ---
    x_plot = None
    x_calc_numeric = None
    original_x_index = None

    try:
        if isinstance(x_data, pd.Series):
            original_x_index = x_data.index
            if isinstance(x_data.index, pd.DatetimeIndex):
                 x_plot = x_data.index
                 x_calc_numeric = x_data.index.astype(np.int64)
            elif pd.api.types.is_numeric_dtype(x_data):
                x_plot = x_data.values
                x_calc_numeric = x_data.values
            else:
                # logger.debug(f"Non-datetime index, non-numeric values for LOWESS x_data Series {name_suffix}. Using index for plot, positional for calc.")
                x_plot = x_data.index
                x_calc_numeric = np.arange(len(x_data))

        elif isinstance(x_data, pd.DatetimeIndex):
             x_plot = x_data
             x_calc_numeric = x_data.astype(np.int64)
             original_x_index = x_data

        elif isinstance(x_data, (list, np.ndarray)):
             if len(x_data) == len(y_series):
                 try:
                     potential_dt_index = pd.to_datetime(x_data)
                     x_plot = potential_dt_index
                     x_calc_numeric = potential_dt_index.astype(np.int64)
                 except (ValueError, TypeError):
                      if all(isinstance(i, (int, float, np.number)) and np.isfinite(i) for i in x_data):
                         x_plot = np.array(x_data)
                         x_calc_numeric = np.array(x_data)
                      else:
                         # logger.debug(f"Non-numeric, non-datetime list/array for LOWESS x_data {name_suffix}. Using original for plot, positional for calc.")
                         x_plot = np.array(x_data)
                         x_calc_numeric = np.arange(len(x_data))
                 original_x_index = y_series.index
             else:
                logger.warning(f"x_data (list/array) length ({len(x_data)}) mismatch with y_series length ({len(y_series)}) for LOWESS {name_suffix}. Skipping.")
                return
        else:
            logger.warning(f"Unsupported x_data type ({type(x_data)}) for LOWESS {name_suffix}. Skipping.")
            return
    except Exception as e:
         logger.warning(f"Error processing x_data for LOWESS {name_suffix}: {e}. Skipping.")
         return

    # --- Align y_data with the chosen x representation ---
    alignment_index = original_x_index if original_x_index is not None else y_series.index

    if not y_series.index.equals(alignment_index):
         try:
            y_series_aligned = y_series.reindex(alignment_index)
         except Exception as align_err:
             logger.warning(f"Failed to align y_data index ({y_series.index}) with x_data index ({alignment_index}) for LOWESS {name_suffix}. Error: {align_err}. Skipping.")
             return
    else:
        y_series_aligned = y_series

    # --- Drop NaNs simultaneously --- 
    try:
        # Create a temporary DataFrame for easier simultaneous dropping
        temp_df = pd.DataFrame({'y': y_series_aligned, 'x_calc': x_calc_numeric, 'x_plot': x_plot})
        temp_df.dropna(subset=['y', 'x_calc'], inplace=True) # Drop based on calculation inputs
        
        if temp_df.empty:
            # logger.debug(f"No valid non-NaN data points remain for LOWESS {name_suffix}.")
            return
            
        y_calc_final = temp_df['y'].values
        x_calc_final = temp_df['x_calc'].values
        x_plot_final = temp_df['x_plot'] # Keep as Series/Index for Plotly

    except Exception as e:
        logger.warning(f"Error during NaN handling/alignment for LOWESS {name_suffix}: {e}. Skipping.")
        return

    if len(y_calc_final) < 10:
        # logger.debug(f"Not enough valid points ({len(y_calc_final)}) after alignment/NaN removal for LOWESS {name_suffix}")
        return

    # --- Perform LOWESS and Plot ---
    try:
        # Ensure inputs are 1D numpy arrays
        if not (isinstance(y_calc_final, np.ndarray) and y_calc_final.ndim == 1):
             y_calc_final = np.asarray(y_calc_final)
        if not (isinstance(x_calc_final, np.ndarray) and x_calc_final.ndim == 1):
             x_calc_final = np.asarray(x_calc_final)

        # Check for NaN/inf again after all processing, should not happen but safeguard
        if np.any(np.isnan(y_calc_final)) or np.any(np.isinf(y_calc_final)) or \
           np.any(np.isnan(x_calc_final)) or np.any(np.isinf(x_calc_final)):
             logger.warning(f"NaN or inf detected in final LOWESS input for {name_suffix}. Skipping.")
             return

        smoothed = sm_lowess(y_calc_final, x_calc_final, frac=frac, return_sorted=False)

        # Ensure smoothed output has same length as x_plot_final
        if len(smoothed) != len(x_plot_final):
             logger.warning(f"LOWESS output length ({len(smoothed)}) mismatch with plot x-axis length ({len(x_plot_final)}) for {name_suffix}. Skipping trace.")
             return

        trace_kwargs = dict(
            x=x_plot_final,
            y=smoothed,
            mode='lines',
            name=f"{kwargs.get('name', '')}{name_suffix}",
            line=dict(dash='dash', width=1.5)
        )
        if color:
            trace_kwargs['line']['color'] = color

        if row is not None and col is not None:
            fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=col)
        else:
            fig.add_trace(go.Scatter(**trace_kwargs))
    except Exception as e:
        logger.error(f"Error in LOWESS calculation or plotting for {name_suffix}. Lengths x={len(x_calc_final)}, y={len(y_calc_final)}. Error: {e}", exc_info=True)

# --- Constants ---
VOLATILITY_PERIODS = [5, 10, 21, 42, 63, 126, 252] # Days for cone/structure
ANNUALIZATION_FACTOR = np.sqrt(252)
REGIME_PERIOD = 21 # Use 21d vol for regime classification
REGIME_QUANTILES = [0.0, 0.25, 0.75, 0.95, 1.0] # Quantiles for Low, Normal, High, Extreme
REGIME_LABELS = ['Low', 'Normal', 'High', 'Extreme']

# --- Volatility Calculation Functions ---

def calculate_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates log returns and adds them to the DataFrame."""
    if df is not None and 'close' in df.columns and len(df) > 1:
        # Ensure numeric type before calculation
        close_prices = pd.to_numeric(df['close'], errors='coerce')
        shifted_close = close_prices.shift(1)
        # Avoid division by zero or log(negative)
        valid_mask = (close_prices > 0) & (shifted_close > 0)
        df['log_returns'] = np.nan # Initialize
        df.loc[valid_mask, 'log_returns'] = np.log(close_prices[valid_mask] / shifted_close[valid_mask])
    elif df is not None:
        df['log_returns'] = np.nan # Assign NaN if calculation not possible
    # If df is None, do nothing and return None implicitly or handle upstream
    return df

def calculate_rolling_volatility(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Calculates rolling volatility for standard periods."""
    vol_data = {}
    if df is not None and 'log_returns' in df.columns and not df['log_returns'].isnull().all():
        log_returns = df['log_returns'].dropna()
        for period in VOLATILITY_PERIODS:
            col_name = f'vol_{period}d'
            if len(log_returns) >= period:
                rolling_std = log_returns.rolling(window=period).std()
                df[col_name] = (rolling_std * ANNUALIZATION_FACTOR).reindex(df.index) # Reindex to original df
                vol_data[period] = df[col_name].dropna()
            else:
                df[col_name] = np.nan
                vol_data[period] = pd.Series(dtype=float)
    elif df is not None:
         # Create empty columns if log_returns doesn't exist or is all NaN
         for period in VOLATILITY_PERIODS:
             df[f'vol_{period}d'] = np.nan
             vol_data[period] = pd.Series(dtype=float)
    # If df is None, return None, {} or handle upstream
    return df, vol_data

def calculate_parkinson_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Parkinson volatility (using High/Low range)."""
    window = 21 # Standard Parkinson window
    if df is not None and all(c in df.columns for c in ['high', 'low']) and len(df) >= window:
         try:
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')
            # Add epsilon only where low is zero or less
            low_safe = np.where(low <= 0, 1e-10, low)
            high_safe = np.where(high <= 0, low_safe, high) # Ensure high >= low_safe
            
            # Calculate only where high > low_safe
            valid_mask = high_safe > low_safe
            log_hl_ratio_sq = pd.Series(np.nan, index=df.index)
            log_hl_ratio_sq.loc[valid_mask] = (np.log(high_safe[valid_mask] / low_safe[valid_mask])) ** 2
            
            parkinson_sum = log_hl_ratio_sq.rolling(window=window).sum()
            df['parkinson_vol'] = (np.sqrt(parkinson_sum / (4 * window * np.log(2))) * ANNUALIZATION_FACTOR).reindex(df.index)
         except Exception as e:
             logger.error(f"Error calculating Parkinson volatility: {e}", exc_info=True)
             df['parkinson_vol'] = np.nan
    elif df is not None:
        df['parkinson_vol'] = np.nan
    # If df is None, return None or handle upstream
    return df

def calculate_atr(df: pd.DataFrame, window: int = 14, smooth_window: int = 5) -> pd.DataFrame:
    """Calculates Average True Range (ATR) and related metrics."""
    required_cols = ['high', 'low', 'close']
    if df is None or not all(c in df.columns for c in required_cols) or len(df) < 2:
        # Assign NaNs to all potential output columns if calculation impossible
        for col in [f'atr_{window}d', f'atr_{window}d_smoothed', 'atr_normalized', 'atr_normalized_smoothed']:
             if df is not None and col not in df.columns: df[col] = np.nan
        return df

    try:
        high = pd.to_numeric(df['high'], errors='coerce')
        low = pd.to_numeric(df['low'], errors='coerce')
        close = pd.to_numeric(df['close'], errors='coerce')
        close_prev = close.shift(1)

        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)

        tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
        # Calculate True Range, fill initial NaN if any
        tr = tr_df.max(axis=1).fillna(0)
        
        atr_col = f'atr_{window}d'
        atr_smooth_col = f'atr_{window}d_smoothed'
        atr_norm_col = 'atr_normalized'
        atr_norm_smooth_col = 'atr_normalized_smoothed'

        if len(df.dropna(subset=required_cols)) >= window:
            df[atr_col] = tr.rolling(window=window).mean()
            # Ensure enough data for centered smoothing window
            if len(df.dropna(subset=required_cols)) >= window + smooth_window -1: 
                 df[atr_smooth_col] = df[atr_col].rolling(window=smooth_window, center=True).mean()
            else:
                 df[atr_smooth_col] = np.nan
                 
            # Normalize ATR - avoid division by zero/NaN
            close_safe = np.where(close > 0, close, np.nan) # Use NaN for non-positive close
            df[atr_norm_col] = (df[atr_col] / close_safe) * 100 
            df[atr_norm_smooth_col] = df[atr_norm_col].rolling(window=smooth_window, center=True).mean()
        else:
            df[atr_col] = np.nan
            df[atr_smooth_col] = np.nan
            df[atr_norm_col] = np.nan
            df[atr_norm_smooth_col] = np.nan
            
        # Reindex final columns to match original DataFrame index
        for col in [atr_col, atr_smooth_col, atr_norm_col, atr_norm_smooth_col]:
             if col in df.columns: df[col] = df[col].reindex(df.index)
             
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}", exc_info=True)
        # Assign NaNs if error occurs
        for col in [f'atr_{window}d', f'atr_{window}d_smoothed', 'atr_normalized', 'atr_normalized_smoothed']:
             if col not in df.columns: df[col] = np.nan
             else: df[col].values[:] = np.nan
             
    return df

def calculate_volatility_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classifies periods based on volatility levels."""
    vol_col = f'vol_{REGIME_PERIOD}d'
    if df is not None and vol_col in df.columns:
        try:
            vol_for_regime = df[vol_col].dropna()
            if not vol_for_regime.empty and len(vol_for_regime) > len(REGIME_QUANTILES):
                quantile_thresholds = vol_for_regime.quantile(REGIME_QUANTILES).tolist()
                unique_thresholds = sorted(list(set(quantile_thresholds)))
                
                if len(unique_thresholds) > 1:
                    num_bins = len(unique_thresholds) - 1
                    # Adjust labels based on the actual number of bins created
                    current_labels = REGIME_LABELS[:num_bins] if num_bins <= len(REGIME_LABELS) else REGIME_LABELS[-num_bins:]
                    # Ensure length of labels matches num_bins
                    if len(current_labels) != num_bins:
                        logger.warning(f"Mismatch between number of unique quantile bins ({num_bins}) and labels ({len(current_labels)}). Adjusting labels.")
                        # Fallback: generate generic labels if needed
                        current_labels = [f"Q{i+1}" for i in range(num_bins)]
                        
                    df['vol_regime'] = pd.cut(
                        df[vol_col],
                        bins=unique_thresholds,
                        labels=current_labels,
                        include_lowest=True,
                        duplicates='drop'
                    )
                    # Fill NaNs robustly
                    fill_value = df['vol_regime'].mode()[0] if not df['vol_regime'].mode().empty else 'Normal'
                    # First bfill then ffill to handle edges
                    df['vol_regime'] = df['vol_regime'].bfill().ffill()
                    # Fill any remaining with mode
                    df['vol_regime'].fillna(fill_value, inplace=True)
                    df['vol_regime'] = df['vol_regime'].reindex(df.index) # Ensure alignment
                else:
                     logger.warning(f"Only one unique quantile threshold found for {vol_col}. Assigning default regime 'Normal'.")
                     df['vol_regime'] = 'Normal'
            else:
                logger.info(f"Not enough data points ({len(vol_for_regime)}) or vol column {vol_col} empty/all NaN for regime classification.")
                df['vol_regime'] = 'Unknown'
        except Exception as e:
            logger.error(f"Error calculating volatility regime: {e}", exc_info=True)
            df['vol_regime'] = 'Unknown'
    elif df is not None:
        df['vol_regime'] = 'Unknown' # Column doesn't exist
    # If df is None, return None or handle upstream
    return df

# --- Volatility Plotting Functions ---

def plot_rolling_volatility(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates the rolling volatility plot with LOWESS trendlines."""
    if df is None or df.empty or 'date' not in df.columns:
        logger.warning(f"Insufficient data for rolling volatility plot for {symbol}.")
        return None
    
    fig = go.Figure()
    colors = {
        5: 'blue',
        21: 'orange',
        63: 'green'
    }
    periods_to_plot = [5, 21, 63]
    plot_added = False

    df_plot = df.set_index('date') if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df_plot.index, pd.DatetimeIndex):
         logger.warning(f"Could not ensure DatetimeIndex for rolling volatility plot of {symbol}. Skipping.")
         return None
         
    dates = df_plot.index

    for period in periods_to_plot:
        col_name = f'vol_{period}d'
        if col_name in df_plot.columns and not df_plot[col_name].isnull().all():
            series = df_plot[col_name].dropna()
            if series.empty:
                continue # Skip if all values were NaN
                
            trace_name = f"{period}-Day Rolling Vol"
            trace_color = colors.get(period, 'black')
            
            fig.add_trace(go.Scatter(
                x=series.index, y=series.values, mode='lines',
                name=trace_name, line=dict(color=trace_color)
            ))
            # Add LOWESS trace (pass original series with NaNs)
            _add_lowess_trace(fig, x_data=df_plot.index, y_data=df_plot[col_name], name_suffix=" LOWESS", color=trace_color, name=trace_name)
            plot_added = True

    if not plot_added:
        logger.warning(f"No valid rolling volatility data found to plot for periods {periods_to_plot} for {symbol}.")
        return None

    fig.update_layout(
        title=f"{symbol} Rolling Volatility (Annualized)",
        yaxis_title="Annualized Volatility",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize rolling volatility plot for {symbol}: {e}", exc_info=True)
        return None

def plot_parkinson_volatility(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates the Parkinson volatility plot with LOWESS trendline."""
    col_name = 'parkinson_vol'
    if df is None or df.empty or 'date' not in df.columns or col_name not in df.columns or df[col_name].isnull().all():
        logger.warning(f"Insufficient data for Parkinson volatility plot for {symbol}.")
        return None

    df_plot = df.set_index('date') if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df_plot.index, pd.DatetimeIndex):
         logger.warning(f"Could not ensure DatetimeIndex for Parkinson volatility plot of {symbol}. Skipping.")
         return None
         
    series = df_plot[col_name].dropna()
    if series.empty:
        logger.warning(f"Parkinson volatility data is all NaN for {symbol}. Skipping plot.")
        return None
        
    fig = go.Figure()
    trace_name = "Parkinson Volatility (21d)"
    trace_color = "purple"

    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode='lines',
        name=trace_name, line=dict(color=trace_color)
    ))
    # Add LOWESS trace (pass original series with NaNs)
    _add_lowess_trace(fig, x_data=df_plot.index, y_data=df_plot[col_name], name_suffix=" LOWESS", color=trace_color, name=trace_name)

    fig.update_layout(
        title=f"{symbol} Parkinson High-Low Volatility (21d, Annualized)",
        yaxis_title="Annualized Volatility",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize Parkinson volatility plot for {symbol}: {e}", exc_info=True)
        return None

def plot_atr(df: pd.DataFrame, symbol: str, window: int = 14) -> dict | None:
    """Creates the ATR plot with Price and Normalized ATR (with LOWESS)."""
    atr_norm_col = 'atr_normalized'
    required_cols = ['date', 'close', atr_norm_col]
    if df is None or df.empty or not all(c in df.columns for c in required_cols) or df[atr_norm_col].isnull().all():
        logger.warning(f"Insufficient data for ATR plot for {symbol} (requires: {required_cols}).")
        return None
        
    df_plot = df.set_index('date') if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df_plot.index, pd.DatetimeIndex):
         logger.warning(f"Could not ensure DatetimeIndex for ATR plot of {symbol}. Skipping.")
         return None
         
    dates = df_plot.index
    close_series = df_plot['close'].dropna()
    atr_series = df_plot[atr_norm_col].dropna()

    if close_series.empty or atr_series.empty:
         logger.warning(f"Close or Normalized ATR data is all NaN for {symbol}. Skipping ATR plot.")
         return None
         
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, 
                        row_heights=[0.7, 0.3])

    # Price Line (Top Row)
    fig.add_trace(
        go.Scatter(x=close_series.index, y=close_series.values, name='Close Price', 
                   line=dict(color='black', width=1)),
        row=1, col=1
    )

    # Normalized ATR Line (Bottom Row)
    trace_name_atr = f'Normalized ATR ({window}d %)'
    trace_color_atr = 'magenta'
    fig.add_trace(
        go.Scatter(x=atr_series.index, y=atr_series.values,
                   name=trace_name_atr, line=dict(color=trace_color_atr)),
        row=2, col=1
    )
    # Add LOWESS for Normalized ATR (pass original series with NaNs)
    _add_lowess_trace(fig, x_data=df_plot.index, y_data=df_plot[atr_norm_col], name_suffix=" LOWESS", color=trace_color_atr, name=trace_name_atr, row=2, col=1)

    fig.update_layout(
        title_text=f"{symbol} Price and Normalized ATR ({window}d %)",
        xaxis_title=None,
        xaxis2_title="Date",
        yaxis_title="Price", yaxis_type="log",
        yaxis2_title="Norm ATR (%)",
        legend_title="Metrics",
        xaxis_rangeslider_visible=False
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize ATR plot for {symbol}: {e}", exc_info=True)
        return None

def plot_volatility_comparison(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates the volatility comparison plot (5d vs 21d) with price and LOWESS."""
    vol5_col = 'vol_5d'
    vol21_col = 'vol_21d'
    required_cols = ['date', 'open', 'high', 'low', 'close', vol5_col, vol21_col]
    if df is None or df.empty or not all(c in df.columns for c in required_cols):
        logger.warning(f"Insufficient data for volatility comparison plot for {symbol} (requires: {required_cols}).")
        return None

    df_plot = df.set_index('date') if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df_plot.index, pd.DatetimeIndex):
         logger.warning(f"Could not ensure DatetimeIndex for volatility comparison plot of {symbol}. Skipping.")
         return None
         
    dates = df_plot.index
    # Check if OHLC data exists and is not all NaN
    ohlc_cols = ['open', 'high', 'low', 'close']
    if df_plot[ohlc_cols].isnull().all().all():
        logger.warning(f"OHLC data is all NaN for {symbol}. Cannot plot candlestick.")
        # Optionally, could plot close price as a line instead
        # return None # Or modify to plot line
        plot_candlestick = False
    else:
        plot_candlestick = True
        
    vol5_series = df_plot[vol5_col].dropna()
    vol21_series = df_plot[vol21_col].dropna()

    if vol5_series.empty and vol21_series.empty:
        logger.warning(f"5d and 21d volatility data are all NaN for {symbol}. Skipping comparison plot.")
        return None
        
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.03, 
                         row_heights=[0.7, 0.3])

    # Price Candlestick or Line (Top Row)
    if plot_candlestick:
        fig.add_trace(go.Candlestick(x=dates,
                                   open=df_plot['open'], high=df_plot['high'],
                                   low=df_plot['low'], close=df_plot['close'],
                                   name='Price'), row=1, col=1)
    elif not df_plot['close'].dropna().empty:
        close_valid = df_plot['close'].dropna()
        fig.add_trace(go.Scatter(x=close_valid.index, y=close_valid.values, name='Close Price', 
                   line=dict(color='black', width=1)), row=1, col=1)
    else:
        logger.warning(f"No valid close price data to plot for {symbol} in top panel.")
        # Top panel will be empty
        
    plot_added_bottom = False
    # 5d Volatility (Bottom Row)
    if not vol5_series.empty:
        trace_name_5d = '5d Vol'
        trace_color_5d = 'blue'
        fig.add_trace(go.Scatter(x=vol5_series.index, y=vol5_series.values,
                                    mode='lines', name=trace_name_5d,
                                    line=dict(color=trace_color_5d)), row=2, col=1)
        # Add 5d Vol LOWESS (pass original series with NaNs)
        _add_lowess_trace(fig, x_data=dates, y_data=df_plot[vol5_col], name_suffix=" LOWESS", color=trace_color_5d, name=trace_name_5d, row=2, col=1)
        plot_added_bottom = True

    # 21d Volatility (Bottom Row)
    if not vol21_series.empty:
        trace_name_21d = '21d Vol'
        trace_color_21d = 'orange'
        fig.add_trace(go.Scatter(x=vol21_series.index, y=vol21_series.values,
                                    mode='lines', name=trace_name_21d,
                                    line=dict(color=trace_color_21d)), row=2, col=1)
        # Add 21d Vol LOWESS (pass original series with NaNs)
        _add_lowess_trace(fig, x_data=dates, y_data=df_plot[vol21_col], name_suffix=" LOWESS", color=trace_color_21d, name=trace_name_21d, row=2, col=1)
        plot_added_bottom = True
    
    if not plot_added_bottom:
        logger.warning(f"No valid 5d or 21d volatility data to plot for {symbol}. Returning plot without bottom panel.")
        # Proceed without bottom panel if top panel has data, otherwise return None?
        # If top panel is also empty (no OHLC/close), return None.
        if not plot_candlestick and df_plot['close'].dropna().empty:
             return None
             
    fig.update_layout(
        title_text=f"{symbol} Price and Volatility Comparison (5d vs 21d)",
        xaxis_rangeslider_visible=False,
        xaxis_title=None,
        xaxis2_title="Date",
        yaxis_title="Price", yaxis_type="log", # Keep price log scaled
        yaxis2_title="Ann. Volatility",
        legend_title="Metrics"
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize volatility comparison plot for {symbol}: {e}", exc_info=True)
        return None

def plot_volatility_term_structure(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates the volatility term structure heatmap."""
    if df is None or df.empty:
        logger.warning(f"Insufficient data for volatility term structure heatmap for {symbol}.")
        return None
        
    vol_cols = [f'vol_{p}d' for p in VOLATILITY_PERIODS if f'vol_{p}d' in df.columns]
    if not vol_cols:
        logger.warning(f"No volatility columns found for term structure heatmap for {symbol}.")
        return None
        
    df_plot = df.set_index('date') if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df_plot.index, pd.DatetimeIndex):
         logger.warning(f"Could not ensure DatetimeIndex for volatility term structure plot of {symbol}. Skipping.")
         return None
         
    vol_term_df = df_plot[vol_cols].copy()
    vol_term_df.dropna(how='all', inplace=True) # Drop rows where ALL periods are NaN

    if vol_term_df.empty:
        logger.warning(f"Volatility term structure data is empty after dropping NaNs for {symbol}.")
        return None

    # Extract periods that actually have data *after* dropping rows
    valid_cols = [c for c in vol_term_df.columns if not vol_term_df[c].isnull().all()]
    if not valid_cols:
         logger.warning(f"No valid volatility columns remain after dropping NaNs for term structure heatmap of {symbol}.")
         return None
         
    vol_term_df = vol_term_df[valid_cols]
    valid_periods = [int(c.split('_')[1][:-1]) for c in vol_term_df.columns]
    
    # Replace NaN with None for Plotly heatmap (though it should handle NaNs)
    z_values = vol_term_df.T.fillna(np.nan).values # Transpose so periods are on y-axis
    if np.all(np.isnan(z_values)):
         logger.warning(f"All volatility values are NaN in the final term structure matrix for {symbol}. Skipping heatmap.")
         return None

    try:
        fig = px.imshow(
            z_values,
            labels=dict(x="Date", y="Volatility Period (Days)", color="Ann. Volatility"),
            x=vol_term_df.index, # Use DatetimeIndex for x-axis
            y=valid_periods,
            aspect="auto",
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            title=f"{symbol} Volatility Term Structure Heatmap",
            xaxis_tickformat='%b %Y'
        )
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to create or serialize volatility term structure heatmap for {symbol}: {e}", exc_info=True)
        return None

def plot_volatility_cone(df: pd.DataFrame, vol_data: dict, symbol: str) -> dict | None:
    """Creates the volatility cone plot."""
    if df is None or df.empty or not vol_data:
        logger.warning(f"Insufficient data for volatility cone plot for {symbol}.")
        return None

    # Use periods for which we actually calculated volatility and have data
    valid_periods = [p for p in VOLATILITY_PERIODS if f'vol_{p}d' in df.columns and p in vol_data and not vol_data[p].empty]
    if not valid_periods:
        logger.warning(f"No valid volatility periods with data found for cone plot for {symbol}.")
        return None
        
    cone_stats = pd.DataFrame(index=valid_periods)
    try:
        cone_stats['min'] = [vol_data[p].min() for p in valid_periods]
        cone_stats['q25'] = [vol_data[p].quantile(0.25) for p in valid_periods]
        cone_stats['median'] = [vol_data[p].median() for p in valid_periods]
        cone_stats['q75'] = [vol_data[p].quantile(0.75) for p in valid_periods]
        cone_stats['max'] = [vol_data[p].max() for p in valid_periods]
        # Get current value robustly
        cone_stats['current'] = [vol_data[p].dropna().iloc[-1] if not vol_data[p].dropna().empty else np.nan for p in valid_periods]
        # Drop periods where any stat calculation failed (e.g., current)
        cone_stats.dropna(inplace=True)
    except (IndexError, KeyError, ValueError) as e:
        logger.error(f"Error calculating cone statistics for {symbol}: {e}", exc_info=True)
        return None
        
    if cone_stats.empty:
        logger.warning(f"Volatility cone statistics are empty after calculation/NaN drop for {symbol}.")
        return None

    fig = go.Figure()
    x_axis = cone_stats.index.tolist()

    try:
        # Max-Min Range
        fig.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=cone_stats['max'].tolist() + cone_stats['min'].tolist()[::-1],
            fill='toself', fillcolor='rgba(0,100,80,0.1)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='Max-Min Range'
        ))
        # IQR Range
        fig.add_trace(go.Scatter(
            x=x_axis + x_axis[::-1],
            y=cone_stats['q75'].tolist() + cone_stats['q25'].tolist()[::-1],
            fill='toself', fillcolor='rgba(0,176,246,0.2)', line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip", showlegend=True, name='IQR Range'
        ))
        # Median
        fig.add_trace(go.Scatter(
            x=x_axis, y=cone_stats['median'],
            line=dict(color='rgba(0,176,246,0.5)', dash='dash'), name='Historical Median'
        ))
        # Current
        fig.add_trace(go.Scatter(
            x=x_axis, y=cone_stats['current'],
            line=dict(color='red', width=3), mode='lines+markers', name='Current Volatility'
        ))

        fig.update_layout(
            title=f"{symbol} Volatility Cone (Current vs Historical Distribution)",
            xaxis_title="Volatility Period (Days)",
            yaxis_title="Annualized Volatility",
            xaxis_tickvals=x_axis, xaxis_ticktext=[str(p) for p in x_axis],
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error creating or serializing volatility cone plot for {symbol}: {e}", exc_info=True)
        return None

def plot_volatility_regime(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates the volatility regime plot."""
    regime_col = 'vol_regime'
    if df is None or df.empty or 'date' not in df.columns or regime_col not in df.columns or df[regime_col].isnull().all():
        logger.warning(f"Insufficient data for volatility regime plot for {symbol}.")
        return None
        
    df_plot = df.set_index('date') if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) else df
    if not isinstance(df_plot.index, pd.DatetimeIndex):
         logger.warning(f"Could not ensure DatetimeIndex for volatility regime plot of {symbol}. Skipping.")
         return None
         
    # Drop rows where regime is NaN for plotting
    df_plot = df_plot.dropna(subset=[regime_col])
    if df_plot.empty:
        logger.warning(f"Volatility regime data is all NaN after dropping NaNs for {symbol}. Skipping plot.")
        return None

    # Use the actual labels present in the data for plotting & define colors
    present_labels = df_plot[regime_col].unique().tolist()
    all_regime_colors = {'Low': 'blue', 'Normal': 'green', 'High': 'orange', 'Extreme': 'red', 'Unknown': 'grey'}
    # Filter colors for labels actually present
    plot_colors = {label: all_regime_colors.get(label, 'grey') for label in present_labels if label in all_regime_colors}
    
    if not plot_colors:
        logger.warning(f"No known regimes found in the data for {symbol}. Skipping regime plot.")
        return None
        
    fig = go.Figure()
    # Plot in specific order if possible, ensuring labels exist in plot_colors
    ordered_labels = [label for label in REGIME_LABELS if label in plot_colors]
    
    try:
        for label in ordered_labels:
            df_regime = df_plot[df_plot[regime_col] == label]
            if not df_regime.empty:
                # Use y=1 for bar height, hover text shows regime name
                fig.add_trace(go.Bar(
                    x=df_regime.index,
                    y=[1] * len(df_regime),
                    name=label,
                    marker_color=plot_colors[label],
                    hoverinfo='x+name',
                    showlegend=True # Ensure legend items appear
                ))

        fig.update_layout(
            title=f"{symbol} Volatility Regimes based on {REGIME_PERIOD}d Volatility Quantiles",
            xaxis_title="Date",
            yaxis_title=None, # No meaningful y-axis title
            xaxis_tickformat='%b %Y',
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1.1]), # Hide y-axis, set range
            barmode='stack',
            showlegend=True,
            legend=dict(traceorder='reversed', # Match REGIME_LABELS order if possible
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error creating or serializing volatility regime plot for {symbol}: {e}", exc_info=True)
        return None

def run_all_volatility_analyses(df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Runs all volatility calculations and plotting, returns updated df and results dict."""
    logger.info(f"Starting volatility analysis for {symbol}...")
    if df is None:
        logger.warning(f"Input DataFrame is None for volatility analysis of {symbol}. Returning empty results.")
        return None, {}
        
    # Work on a copy to avoid side effects
    df_analysis = df.copy()
    
    df_analysis = calculate_log_returns(df_analysis)
    df_analysis, vol_data = calculate_rolling_volatility(df_analysis)
    df_analysis = calculate_parkinson_volatility(df_analysis)
    df_analysis = calculate_atr(df_analysis) # Uses default window=14
    df_analysis = calculate_volatility_regime(df_analysis)
    
    results = {}
    logger.info(f"Generating volatility plots for {symbol}...")
    results["rolling_volatility"] = plot_rolling_volatility(df_analysis, symbol)
    results["parkinson_volatility"] = plot_parkinson_volatility(df_analysis, symbol)
    results["atr"] = plot_atr(df_analysis, symbol) # Uses default window=14
    results["volatility_comparison"] = plot_volatility_comparison(df_analysis, symbol)
    results["volatility_term_structure"] = plot_volatility_term_structure(df_analysis, symbol)
    results["volatility_cone"] = plot_volatility_cone(df_analysis, vol_data, symbol)
    results["volatility_regime"] = plot_volatility_regime(df_analysis, symbol)
    
    # Filter out None results
    filtered_results = {k: v for k, v in results.items() if v is not None}
    logger.info(f"Volatility analysis complete for {symbol}. Generated {len(filtered_results)} plots.")
    return df_analysis, filtered_results 