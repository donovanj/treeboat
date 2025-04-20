# Placeholder for anomaly detection calculation logic 

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px # Import px consistently
import json
from plotly.utils import PlotlyJSONEncoder
from scipy import stats # For z-score and median absolute deviation
import logging

# Optional QuantLib for business day frequency
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

# Optional Sklearn for clustering/anomaly detection models
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Dummy classes/functions if sklearn not available
    class IsolationForest: pass
    class StandardScaler: pass
    class DBSCAN: pass
    class PCA: pass

# Optional Statsmodels for LOWESS/STL
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm_lowess = None
    STL = None
    # print("Warning: statsmodels not installed. LOWESS/STL will be skipped.")

# Get the logger instance
logger = logging.getLogger("financial_prediction_system")

# --- Constants ---
Z_SCORE_THRESHOLD = 3.0
MOD_Z_SCORE_THRESHOLD = 3.5 # Common threshold for modified z-score
IFOREST_CONTAMINATION = 0.05 # Expected proportion of outliers for Isolation Forest
DBSCAN_EPS = 0.5 # DBSCAN epsilon parameter (highly sensitive, needs tuning)
DBSCAN_MIN_SAMPLES = 5 # DBSCAN min_samples parameter (sensitive)
STL_PERIOD = 21 # Default period for STL decomposition (e.g., business days in a month)

# --- Anomaly Calculation Functions ---

def calculate_z_scores(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Calculates Z-scores and Modified Z-scores for a given series."""
    if series is None or series.isnull().all():
        return pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        # Z-score calculation
        z_scores = pd.Series(stats.zscore(series.dropna()), index=series.dropna().index)

        # Modified Z-score calculation
        median = series.median()
        # Use nan_policy='omit' which is default in recent scipy versions but explicit is safer
        mad = stats.median_abs_deviation(series.dropna(), scale='normal', nan_policy='omit')

        if mad == 0 or np.isnan(mad):
            logger.warning("Median Absolute Deviation is 0 or NaN. Cannot calculate Modified Z-scores.")
            mod_z_scores = pd.Series(np.nan, index=series.dropna().index)
        else:
            mod_z_scores = 0.6745 * (series.dropna() - median) / mad # 0.6745 is approx. factor for normal dist

        # Reindex to the original series index, filling NaNs where calculation wasn't possible
        z_scores = z_scores.reindex(series.index)
        mod_z_scores = mod_z_scores.reindex(series.index)

        return z_scores, mod_z_scores
    except Exception as e:
        logger.error(f"Error calculating Z-scores: {e}", exc_info=True)
        return pd.Series(dtype=float), pd.Series(dtype=float)

def apply_isolation_forest(df: pd.DataFrame, features: list[str]) -> pd.Series | None:
    """Applies Isolation Forest for anomaly detection."""
    if not SKLEARN_AVAILABLE:
        logger.warning("Scikit-learn not available. Skipping Isolation Forest.")
        return None
    if df is None or df.empty or not all(f in df.columns for f in features):
        logger.warning(f"Insufficient data or missing features {features} for Isolation Forest.")
        return None

    try:
        data_subset = df[features].dropna()
        if data_subset.empty or len(data_subset) < 2:
            logger.warning("Not enough valid data points for Isolation Forest after dropping NaNs.")
            return None
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)

        model = IsolationForest(contamination=IFOREST_CONTAMINATION, random_state=42)
        predictions = model.fit_predict(scaled_data)
        # Anomalies are -1, normal are 1. Convert to boolean (True for anomaly).
        anomalies = pd.Series(predictions == -1, index=data_subset.index)
        return anomalies.reindex(df.index, fill_value=False) # Reindex to original df index
    except Exception as e:
        logger.error(f"Error applying Isolation Forest: {e}", exc_info=True)
        return None

def apply_dbscan(df: pd.DataFrame, features: list[str]) -> pd.Series | None:
    """Applies DBSCAN for anomaly detection."""
    if not SKLEARN_AVAILABLE:
        logger.warning("Scikit-learn not available. Skipping DBSCAN.")
        return None
    if df is None or df.empty or not all(f in df.columns for f in features):
         logger.warning(f"Insufficient data or missing features {features} for DBSCAN.")
         return None

    try:
        data_subset = df[features].dropna()
        if data_subset.empty or len(data_subset) < DBSCAN_MIN_SAMPLES:
             logger.warning(f"Not enough valid data points ({len(data_subset)}) for DBSCAN (min_samples={DBSCAN_MIN_SAMPLES}).")
             return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)

        # Note: DBSCAN's eps parameter is highly sensitive and often requires tuning.
        model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        predictions = model.fit_predict(scaled_data)
        # Anomalies are typically labeled -1 by DBSCAN.
        anomalies = pd.Series(predictions == -1, index=data_subset.index)
        return anomalies.reindex(df.index, fill_value=False) # Reindex to original df index
    except Exception as e:
        logger.error(f"Error applying DBSCAN: {e}", exc_info=True)
        return None

def perform_stl_decomposition(series: pd.Series, period: int = STL_PERIOD) -> pd.DataFrame | None:
    """Performs STL decomposition on the series."""
    if not STATSMODELS_AVAILABLE or STL is None:
        logger.warning("Statsmodels not available or STL could not be imported. Skipping STL decomposition.")
        return None

    if series is None or series.isnull().all() or len(series.dropna()) < 2 * period:
        logger.warning(f"Not enough data ({len(series.dropna())}) for STL decomposition with period {period}. Required: {2 * period}")
        return None

    # --- Prepare Series Index ---
    series_indexed = series.dropna().copy()
    original_index = series_indexed.index
    is_datetime_index = isinstance(series_indexed.index, pd.DatetimeIndex)
    freq = None

    if is_datetime_index:
        # Try QuantLib calendar approach first if available
        if QUANTLIB_AVAILABLE:
            cal = ql.UnitedStates(ql.UnitedStates.NYSE)
            start_date_ql = ql.Date(original_index.min().day, original_index.min().month, original_index.min().year)
            end_date_ql = ql.Date(original_index.max().day, original_index.max().month, original_index.max().year)
            
            business_days_ql = []
            current_date_ql = start_date_ql
            while current_date_ql <= end_date_ql:
                if cal.isBusinessDay(current_date_ql):
                    business_days_ql.append(pd.Timestamp(current_date_ql.year(), current_date_ql.month(), current_date_ql.dayOfMonth()))
                current_date_ql += ql.Period(1, ql.Days)
            
            if business_days_ql:
                business_days_index = pd.DatetimeIndex(business_days_ql)
                try:
                    series_indexed = series_indexed.reindex(business_days_index).interpolate(method='time')
                    # Check if reindexing + interpolation succeeded
                    if not series_indexed.isnull().all():
                       freq = 'B' # Assign business day frequency
                       logger.info("Successfully reindexed series to NYSE business days for STL.")
                    else:
                         logger.warning("Reindexing to NYSE business days resulted in all NaNs. Falling back to original index.")
                         series_indexed = series.dropna().copy() # Revert
                         freq = pd.infer_freq(series_indexed.index)
                except Exception as e:
                    logger.warning(f"Failed to reindex/interpolate with QuantLib calendar: {e}. Falling back to original index.")
                    series_indexed = series.dropna().copy() # Revert
                    freq = pd.infer_freq(series_indexed.index)
            else:
                logger.warning("QuantLib calendar generated no business days for the period. Using original index.")
                freq = pd.infer_freq(series_indexed.index)
        else:
             # Fallback if QuantLib not available: try pandas frequency inference
             freq = pd.infer_freq(series_indexed.index)
             if freq:
                 try:
                    series_indexed = series_indexed.asfreq(freq)
                 except ValueError as e: # Handle cases like non-monotonic index etc.
                    logger.warning(f"Could not use inferred frequency '{freq}' with asfreq: {e}. Proceeding without regular frequency.")
                    freq = None # Reset freq if asfreq fails
             else:
                 logger.warning("Could not infer frequency for STL using pandas. Results might be less reliable.")
    else:
        # Not a DatetimeIndex initially, try converting it
        try:
            series_indexed.index = pd.to_datetime(series_indexed.index)
            # Try frequency inference again after conversion
            freq = pd.infer_freq(series_indexed.index)
            if freq:
                 try:
                    series_indexed = series_indexed.asfreq(freq)
                 except ValueError as e:
                    logger.warning(f"Could not use inferred frequency '{freq}' after index conversion: {e}. Proceeding without regular frequency.")
                    freq = None
            else:
                 logger.warning("Index converted to DatetimeIndex, but could not infer frequency for STL.")
        except Exception as e:
            logger.warning(f"Could not convert index to DatetimeIndex for STL: {e}. Using integer index.")
            # Use integer index if conversion fails
            series_indexed = series_indexed.reset_index(drop=True)
            is_datetime_index = False # Mark as not datetime index

    # --- Interpolate remaining NaNs ---
    # Interpolate first (time method if datetime index, linear otherwise)
    interp_method = 'time' if is_datetime_index and freq else 'linear'
    series_indexed.interpolate(method=interp_method, limit_direction='both', inplace=True)
    # Fill any remaining start/end NaNs
    series_indexed.bfill(inplace=True)
    series_indexed.ffill(inplace=True)

    if series_indexed.isnull().any():
        logger.warning("Series still contains NaNs after interpolation/fill for STL. Decomposition might fail or be inaccurate.")
        # Optionally, drop remaining NaNs if STL requires it, but this might shorten the series
        # series_indexed.dropna(inplace=True)

    if len(series_indexed) < 2 * period:
        logger.warning(f"Not enough data points ({len(series_indexed)}) after processing for STL period {period}. Required: {2 * period}")
        return None

    # --- Perform STL ---
    try:
        logger.info(f"Performing STL decomposition with period={period}, frequency='{freq if freq else 'None'}', length={len(series_indexed)}");
        # Pass freq to STL if inferred, otherwise let statsmodels handle it
        stl_instance = STL(series_indexed, period=period, robust=True)
        result = stl_instance.fit()

        # Combine results into a DataFrame, reindexing to original dates where possible
        decomp_df = pd.DataFrame({
            'observed': result.observed,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'resid': result.resid
        })
        # Reindex back to the original pre-processed index if possible
        try:
             decomp_df = decomp_df.reindex(original_index)
        except ValueError:
            logger.warning("Could not reindex STL results back to original index.")
            # Keep the index used by STL if reindexing fails

        return decomp_df
    except Exception as e:
        logger.error(f"Error performing STL decomposition: {e}", exc_info=True)
        return None

# --- Anomaly Plotting Functions ---

# --- LOWESS Helper Function (Revised V2) ---
def _add_lowess_trace(fig, x_data, y_data, name_suffix=" LOWESS", color=None, frac=0.25, row=None, col=None, **kwargs):
    """Calculates LOWESS and adds it as a trace to the figure."""
    if not STATSMODELS_AVAILABLE or sm_lowess is None:
        # logger.debug("Statsmodels or LOWESS not available. Skipping LOWESS trace.") # Optional: log skipping
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
            # Align, ensuring we don't introduce rows where x has no corresponding value
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

        # Final check for NaN/inf (should not happen often after dropna)
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

def plot_z_scores(z_scores: pd.Series, mod_z_scores: pd.Series, symbol: str, series_name: str) -> dict | None:
    """Plots Z-scores and Modified Z-scores over time with LOWESS trendlines."""
    if (z_scores is None or z_scores.empty) and (mod_z_scores is None or mod_z_scores.empty):
        logger.info(f"No Z-score data provided for {symbol} {series_name}. Skipping Z-score plot.")
        return None

    fig = go.Figure()
    # Determine common index, preferring z_scores if available
    dates = z_scores.index if (z_scores is not None and not z_scores.empty) else (mod_z_scores.index if (mod_z_scores is not None and not mod_z_scores.empty) else None)
    if dates is None:
        logger.warning(f"Cannot determine date index for Z-score plot of {symbol} {series_name}. Skipping.")
        return None # Cannot plot if no dates

    trace_color_z = 'blue'
    trace_color_mod_z = 'orange'
    plot_added = False

    if z_scores is not None and not z_scores.dropna().empty:
        # Plot only non-NaN values
        z_scores_valid = z_scores.dropna()
        fig.add_trace(go.Scatter(x=z_scores_valid.index, y=z_scores_valid.values, mode='lines', name='Z-score', line=dict(color=trace_color_z)))
        # Add Z-score LOWESS (use original series with NaNs for LOWESS func to handle)
        _add_lowess_trace(fig, x_data=z_scores.index, y_data=z_scores, name_suffix=" LOWESS", color=trace_color_z, name="Z-score")
        plot_added = True

    if mod_z_scores is not None and not mod_z_scores.dropna().empty:
         mod_z_scores_valid = mod_z_scores.dropna()
         fig.add_trace(go.Scatter(x=mod_z_scores_valid.index, y=mod_z_scores_valid.values, mode='lines', name='Modified Z-score', line=dict(color=trace_color_mod_z)))
         # Add Mod Z-score LOWESS
         _add_lowess_trace(fig, x_data=mod_z_scores.index, y_data=mod_z_scores, name_suffix=" LOWESS", color=trace_color_mod_z, name="Modified Z-score")
         plot_added = True

    if not plot_added:
         logger.warning(f"No valid Z-score or Modified Z-score data to plot for {symbol} {series_name}.")
         return None

    # Add threshold lines
    fig.add_hline(y=Z_SCORE_THRESHOLD, line_dash="dash", line_color="red", annotation_text=f"Z={Z_SCORE_THRESHOLD}")
    fig.add_hline(y=-Z_SCORE_THRESHOLD, line_dash="dash", line_color="red")
    fig.add_hline(y=MOD_Z_SCORE_THRESHOLD, line_dash="dot", line_color="green", annotation_text=f"ModZ={MOD_Z_SCORE_THRESHOLD}")
    fig.add_hline(y=-MOD_Z_SCORE_THRESHOLD, line_dash="dot", line_color="green")

    fig.update_layout(
        title=f"{symbol} - {series_name} Z-Scores",
        xaxis_title="Date",
        yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize Z-score plot for {symbol} {series_name}: {e}", exc_info=True)
        return None

def plot_data_with_z_anomalies(series: pd.Series, z_scores: pd.Series, mod_z_scores: pd.Series, symbol: str, series_name: str) -> dict | None:
    """Plots the original series highlighting anomalies detected by Z-scores, with LOWESS trendline."""
    if series is None or series.dropna().empty:
        logger.warning(f"Original series {series_name} for {symbol} is empty or all NaN. Skipping anomaly highlight plot.")
        return None

    fig = go.Figure()
    series_valid = series.dropna()

    # Plot original series first
    fig.add_trace(go.Scatter(x=series_valid.index, y=series_valid.values, mode='lines', name=f'Original {series_name}', line=dict(color='grey', width=1)))

    # Add LOWESS for the original series
    _add_lowess_trace(fig, x_data=series.index, y_data=series, name_suffix=" LOWESS", color='black', name=f'Original {series_name}')

    # Check for anomalies if scores are available and not all NaN
    z_available = z_scores is not None and not z_scores.dropna().empty
    mod_z_available = mod_z_scores is not None and not mod_z_scores.dropna().empty

    if z_available or mod_z_available:
        # Identify anomalies (handle potential KeyErrors if scores index doesn't perfectly match series)
        z_anomalies_idx = pd.Index([])
        mod_z_anomalies_idx = pd.Index([])
        try:
            if z_available:
                z_anomalies_idx = z_scores.loc[abs(z_scores) > Z_SCORE_THRESHOLD].index
            if mod_z_available:
                mod_z_anomalies_idx = mod_z_scores.loc[abs(mod_z_scores) > MOD_Z_SCORE_THRESHOLD].index
        except KeyError as e:
            logger.warning(f"KeyError finding anomaly indices for {symbol} {series_name}, likely index mismatch: {e}")
            # Attempt intersection to be safe
            common_index = series.index.intersection(z_scores.index if z_available else pd.Index([])).intersection(mod_z_scores.index if mod_z_available else pd.Index([]))
            if z_available:
                 z_anomalies_idx = z_scores.loc[common_index][abs(z_scores.loc[common_index]) > Z_SCORE_THRESHOLD].index
            if mod_z_available:
                 mod_z_anomalies_idx = mod_z_scores.loc[common_index][abs(mod_z_scores.loc[common_index]) > MOD_Z_SCORE_THRESHOLD].index

        all_anomalies_idx = z_anomalies_idx.union(mod_z_anomalies_idx)
        # Ensure anomaly indices exist in the original series index
        valid_anomaly_idx = series.index.intersection(all_anomalies_idx)

        # Plot Anomalies
        if not valid_anomaly_idx.empty:
            anomaly_values = series.loc[valid_anomaly_idx]
            fig.add_trace(go.Scatter(x=valid_anomaly_idx, y=anomaly_values, mode='markers',
                                   name='Z/ModZ Anomaly', marker=dict(color='red', size=8, symbol='x')))
            title = f"{symbol} - {series_name} with Z-Score Anomalies Highlighted"
        else:
             title = f"{symbol} - {series_name} (No Z-Score Anomalies Detected)"
    else:
        # If no scores, just plot original series and LOWESS
         title = f"{symbol} - {series_name} (Z-Scores Not Calculated or Empty)"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=series_name,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    try:
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Failed to serialize anomaly highlight plot for {symbol} {series_name}: {e}", exc_info=True)
        return None

def plot_clustering_anomalies(df: pd.DataFrame, features: list[str], anomaly_series: pd.Series, method_name: str, symbol: str) -> dict | None:
    """Visualizes clustering anomalies using PCA."""
    if not SKLEARN_AVAILABLE or PCA is None:
        logger.warning("Scikit-learn or PCA not available. Skipping clustering anomaly plot.")
        return None
    if df is None or df.empty or anomaly_series is None or anomaly_series.empty or not all(f in df.columns for f in features):
        logger.warning(f"Insufficient data for plotting {method_name} anomalies for {symbol}.")
        return None

    try:
        data_subset = df[features].dropna()
        # Align anomaly series with the data subset used for clustering/plotting
        # Ensure indices match before aligning
        common_index = data_subset.index.intersection(anomaly_series.index)
        if common_index.empty:
            logger.warning(f"No common index between data subset and anomaly series for {method_name} plot ({symbol}). Skipping.")
            return None
        
        aligned_anomalies = anomaly_series.reindex(common_index)
        data_subset = data_subset.loc[common_index]
        
        # Drop NaNs introduced by reindexing anomaly series (if any)
        valid_mask = aligned_anomalies.notna()
        aligned_anomalies = aligned_anomalies[valid_mask]
        data_subset = data_subset[valid_mask]

        if data_subset.empty or len(data_subset) < 2:
            logger.warning(f"Not enough valid, aligned data points for {method_name} PCA plot ({symbol}).")
            return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(scaled_data)

        plot_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'], index=data_subset.index)
        plot_df['Anomaly'] = aligned_anomalies # Add the boolean anomaly flag

        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='PCA1',
            y='PCA2',
            color='Anomaly', # Color points based on the anomaly flag
            title=f"{symbol} - {method_name} Anomalies (PCA Visualization)",
            labels={'Anomaly': f'{method_name} Anomaly'},
            color_discrete_map={True: 'red', False: 'blue'}, # Explicit colors
            hover_data=[plot_df.index] # Show date on hover
        )
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(legend_title_text=f'{method_name} Anomaly')

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error creating {method_name} PCA plot for {symbol}: {e}", exc_info=True)
        return None

def plot_stl_decomposition(decomp_df: pd.DataFrame, symbol: str) -> dict | None:
    """Plots the STL decomposition components."""
    if decomp_df is None or decomp_df.empty:
        logger.warning(f"STL decomposition data is empty for {symbol}. Skipping plot.")
        return None

    try:
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
            vertical_spacing=0.05
        )
        
        # Use descriptive names for traces
        trace_names = {
            'observed': f'Observed {symbol}',
            'trend': 'Trend',
            'seasonal': 'Seasonal',
            'resid': 'Residual'
        }
        colors = {
            'observed': '#1f77b4', # muted blue
            'trend': '#ff7f0e', # safety orange
            'seasonal': '#2ca02c', # cooked asparagus green
            'resid': '#d62728' # brick red
        }
        row_map = {'observed': 1, 'trend': 2, 'seasonal': 3, 'resid': 4}

        for col in ['observed', 'trend', 'seasonal', 'resid']:
             if col in decomp_df.columns and not decomp_df[col].isnull().all():
                 valid_data = decomp_df[col].dropna()
                 fig.add_trace(go.Scatter(
                     x=valid_data.index,
                     y=valid_data.values,
                     name=trace_names[col],
                     mode='lines',
                     line=dict(color=colors[col])
                 ), row=row_map[col], col=1)
        
        # Try to determine the period used for title (if possible from data index or default)
        # This requires passing the period used or storing it somehow if variable
        title_period = STL_PERIOD # Assume default for now
        
        fig.update_layout(
            title=f"{symbol} - STL Decomposition (Period ~{title_period} days)",
            showlegend=False,
            height=700 # Make plot taller to accommodate subplots
        )
        # Update axis titles
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)

        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    except Exception as e:
        logger.error(f"Error plotting STL decomposition for {symbol}: {e}", exc_info=True)
        return None

# --- Main Runner Function --- #

def run_all_anomaly_analyses(df: pd.DataFrame, symbol: str) -> dict:
    """Runs all anomaly detection calculations and plotting."""
    results = {}
    if df is None or df.empty or 'close' not in df.columns:
        logger.warning(f"DataFrame empty or 'close' column missing for {symbol} anomaly analysis.")
        return {}
        
    df_indexed = df.set_index('date') if 'date' in df.columns else df
    if not isinstance(df_indexed.index, pd.DatetimeIndex):
        try:
            df_indexed.index = pd.to_datetime(df_indexed.index)
        except Exception:
            logger.warning(f"Could not convert index to DatetimeIndex for {symbol} anomaly analysis. Some features might fail.")
            # Proceed with original index if conversion fails
            
    # Use a copy to avoid modifying the original df passed to the function
    df_analysis = df_indexed.copy()
    
    # --- Calculate Base Series for Analysis ---
    base_series_name = 'close' # Default to close price
    base_series = df_analysis[base_series_name]

    # 1. Z-Score / Modified Z-Score Analysis
    logger.info(f"Calculating Z-scores for {symbol} ({base_series_name})...")
    z_scores, mod_z_scores = calculate_z_scores(base_series)
    results["z_score_plot"] = plot_z_scores(z_scores, mod_z_scores, symbol, base_series_name)
    results["data_with_z_anomalies_plot"] = plot_data_with_z_anomalies(base_series, z_scores, mod_z_scores, symbol, base_series_name)

    # 2. Clustering-Based Anomaly Detection (Isolation Forest, DBSCAN)
    # Features commonly used: log returns, volatility
    # Ensure log returns and volatility are calculated (e.g., reuse from volatility module if available)
    # For simplicity here, let's recalculate if needed, assuming 'close' exists
    if 'log_returns' not in df_analysis.columns:
         if len(df_analysis) > 1:
            df_analysis['log_returns'] = np.log(df_analysis['close'] / df_analysis['close'].shift(1))
         else:
             df_analysis['log_returns'] = np.nan
    
    vol_col = f'vol_{STL_PERIOD}d' # Use same period as STL for consistency
    if vol_col not in df_analysis.columns and 'log_returns' in df_analysis.columns:
         if len(df_analysis) >= STL_PERIOD:
             df_analysis[vol_col] = df_analysis['log_returns'].rolling(window=STL_PERIOD).std() * np.sqrt(252) # Annualized
         else:
             df_analysis[vol_col] = np.nan
             
    features_for_clustering = ['log_returns', vol_col]
    # Check if features exist and have data before running clustering
    if all(f in df_analysis.columns for f in features_for_clustering) and not df_analysis[features_for_clustering].isnull().all().all():
        logger.info(f"Running Isolation Forest for {symbol}...")
        iforest_anomalies = apply_isolation_forest(df_analysis, features_for_clustering)
        if iforest_anomalies is not None:
             results["isolation_forest_plot"] = plot_clustering_anomalies(df_analysis, features_for_clustering, iforest_anomalies, "Isolation Forest", symbol)

        logger.info(f"Running DBSCAN for {symbol}...")
        dbscan_anomalies = apply_dbscan(df_analysis, features_for_clustering)
        if dbscan_anomalies is not None:
             results["dbscan_plot"] = plot_clustering_anomalies(df_analysis, features_for_clustering, dbscan_anomalies, "DBSCAN", symbol)
    else:
        logger.warning(f"Skipping clustering analysis for {symbol} due to missing features or all NaN data in: {features_for_clustering}")
        results["isolation_forest_plot"] = None
        results["dbscan_plot"] = None

    # 3. STL Decomposition (using close price)
    logger.info(f"Performing STL decomposition for {symbol}...")
    decomp_df = perform_stl_decomposition(base_series, period=STL_PERIOD)
    if decomp_df is not None:
        results["stl_decomposition_plot"] = plot_stl_decomposition(decomp_df, symbol)
    else:
        results["stl_decomposition_plot"] = None

    # Filter out None results before returning
    return {k: v for k, v in results.items() if v is not None} 