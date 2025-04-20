# Placeholder for correlation analysis calculations 

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import json
from plotly.utils import PlotlyJSONEncoder
import logging # Import logging

# Get the logger instance
logger = logging.getLogger("financial_prediction_system")

# Add lowess import
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    STATSMODELS_LOWESS_AVAILABLE = True
except ImportError:
    STATSMODELS_LOWESS_AVAILABLE = False
    # logger.warning("statsmodels not installed or lowess not available. LOWESS smoothing will be skipped.") # Logged inside function

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


# --- Correlation Analysis Functions ---

def calculate_rolling_correlations(returns_df: pd.DataFrame, symbol: str, rolling_window: int = 63) -> dict | None:
    """Calculates and plots rolling correlations with LOWESS trendlines."""
    if returns_df is None or returns_df.empty or symbol not in returns_df.columns:
        logger.warning(f"Insufficient data for rolling correlations for {symbol}.")
        return None

    other_assets = [col for col in returns_df.columns if col != symbol]
    if not other_assets:
        logger.warning(f"No other assets found for rolling correlation calculation against {symbol}.")
        return None

    fig = make_subplots(specs=[[{"secondary_y": False}]])
    colors = px.colors.qualitative.Plotly # Use Plotly's qualitative colors
    plot_added = False
    
    # Ensure index is DatetimeIndex for plotting
    if not isinstance(returns_df.index, pd.DatetimeIndex):
        try:
             returns_df.index = pd.to_datetime(returns_df.index)
        except Exception as e:
            logger.warning(f"Could not convert index to DatetimeIndex for rolling correlations plot of {symbol}: {e}. Skipping.")
            return None

    for i, asset in enumerate(other_assets):
        try:
            # Calculate rolling correlation, ensuring alignment
            s1 = returns_df[symbol]
            s2 = returns_df[asset]
            # Align series before rolling calculation to handle potential index differences
            s1_aligned, s2_aligned = s1.align(s2, join='inner')
            
            if s1_aligned.empty or len(s1_aligned) < rolling_window:
                 logger.info(f"Not enough aligned data points between {symbol} and {asset} for rolling window {rolling_window}. Skipping.")
                 continue
                 
            rolling_corr = s1_aligned.rolling(rolling_window).corr(s2_aligned).dropna()
            
            if not rolling_corr.empty:
                 trace_name = f'{symbol} vs {asset} ({rolling_window}d Corr)'
                 trace_color = colors[i % len(colors)]
                 
                 # Plot valid (non-NaN) rolling correlation values
                 fig.add_trace(
                     go.Scatter(
                         x=rolling_corr.index,
                         y=rolling_corr.values,
                         name=trace_name,
                         line=dict(color=trace_color),
                         mode='lines'
                     )
                 )
                 # Add LOWESS trace (pass original rolling_corr with potential NaNs handled inside)
                 _add_lowess_trace(fig, x_data=rolling_corr.index, y_data=rolling_corr, name_suffix=" LOWESS", color=trace_color, name=trace_name)
                 plot_added = True
            else:
                logger.info(f"Rolling correlation between {symbol} and {asset} resulted in empty series after dropna.")
        except KeyError:
            logger.warning(f"Column {asset} not found in returns data for rolling correlation with {symbol}.")
        except Exception as e:
            logger.error(f"Error calculating rolling correlation between {symbol} and {asset}: {e}", exc_info=True)
            
    if not plot_added:
        logger.warning(f"No valid rolling correlation traces could be added for {symbol}.")
        return None

    fig.update_layout(
        title_text=f"{symbol} Rolling Correlations ({rolling_window}d Window)",
        xaxis_title="Date",
        yaxis_title="Correlation Coefficient",
        yaxis_range=[-1.05, 1.05], # Add slight buffer
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize rolling correlation plot for {symbol}: {e}", exc_info=True)
        return None

def calculate_correlation_matrix(returns_df: pd.DataFrame, symbol: str, start: str, end: str) -> dict | None:
    """Calculates and plots the static correlation matrix heatmap."""
    if returns_df is None or returns_df.empty or len(returns_df.columns) < 2:
        logger.warning(f"Insufficient data for correlation matrix for {symbol} (requires >= 2 columns).")
        return None

    try:
        # Calculate correlation on non-NaN aligned data
        corr_matrix = returns_df.dropna().corr()
        if corr_matrix.empty:
             logger.warning(f"Correlation matrix is empty after dropna() for {symbol}.")
             return None
             
    except Exception as e:
        logger.error(f"Error calculating correlation matrix for {symbol}: {e}", exc_info=True)
        return None

    # Prepare labels and values
    labels = corr_matrix.columns.tolist()
    z_values = corr_matrix.values
    # Format annotations, handling potential NaNs in the matrix itself if dropna didn't catch all cases
    annotation_text = np.vectorize(lambda x: f'{x:.2f}' if pd.notnull(x) else 'NaN')(z_values)

    # Create annotated heatmap
    try:
        fig = ff.create_annotated_heatmap(
            z=z_values,
            x=labels,
            y=labels,
            annotation_text=annotation_text,
            colorscale='Viridis',
            showscale=True,
            zmin=-1, zmax=1 # Fix color scale range
        )
        fig.update_layout(
            title_text=f"Correlation Matrix ({symbol} & Others) | {start} to {end}",
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_side='bottom',
            yaxis_autorange='reversed' # Often makes heatmaps easier to read
        )
        # Rotate x-axis labels if many assets
        if len(labels) > 10:
            fig.update_layout(xaxis_tickangle=-45)
            
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error creating or serializing annotated heatmap for {symbol}: {e}", exc_info=True)
        return None

def run_all_correlation_analyses(returns_df: pd.DataFrame, symbol: str, start: str, end: str) -> dict:
    """Runs all correlation analysis calculations and returns results."""
    logger.info(f"Starting correlation analysis for {symbol}...")
    results = {}
    if returns_df is not None and not returns_df.empty:
        results["rolling_correlation"] = calculate_rolling_correlations(returns_df, symbol)
        results["correlation_matrix"] = calculate_correlation_matrix(returns_df, symbol, start, end)
    else:
        logger.warning(f"Returns DataFrame is None or empty for correlation analysis of {symbol}. Skipping.")
        results["rolling_correlation"] = None
        results["correlation_matrix"] = None
        
    filtered_results = {k: v for k, v in results.items() if v is not None}
    logger.info(f"Correlation analysis complete for {symbol}. Generated {len(filtered_results)} plots/results.")
    return filtered_results 