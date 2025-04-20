# Placeholder for anomaly detection calculation logic 

import pandas as pd
import numpy as np
import json
import logging

# Plotting
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

# Stats / Math
from scipy import stats

# Optional: QuantLib
try:
    import QuantLib as ql
    QUANTLIB_AVAILABLE = True
except ImportError:
    QUANTLIB_AVAILABLE = False

# Optional: Sklearn
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Dummy classes for graceful failure if sklearn is not installed
    class IsolationForest: pass
    class StandardScaler: pass
    class DBSCAN: pass
    class PCA: pass
    logger.warning("scikit-learn not found. Clustering-based anomaly detection (IsolationForest, DBSCAN) will be skipped.")

# Optional: Statsmodels
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
    from statsmodels.tsa.seasonal import STL
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    sm_lowess = None
    STL = None
    logger.warning("statsmodels not found. LOWESS smoothing and STL decomposition will be skipped.")

# Get the logger instance
logger = logging.getLogger("financial_prediction_system")
# Ensure logger is configured (e.g., basicConfig) - this might be done elsewhere in app setup
# logging.basicConfig(level=logging.INFO) # Example basic config if needed here

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
        logger.debug(f"[IForest] Shape of data_subset after dropna: {data_subset.shape}")
        if data_subset.empty or len(data_subset) < 2:
            logger.warning("[IForest] Not enough valid data points for Isolation Forest after dropping NaNs.")
            return None
            
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)
        logger.debug(f"[IForest] Scaled data shape: {scaled_data.shape}")

        model = IsolationForest(contamination=IFOREST_CONTAMINATION, random_state=42)
        predictions = model.fit_predict(scaled_data)
        anomalies = pd.Series(predictions == -1, index=data_subset.index)
        num_anomalies = anomalies.sum()
        logger.info(f"[IForest] Detected {num_anomalies} anomalies out of {len(anomalies)} points.")
        return anomalies.reindex(df.index, fill_value=False)
    except Exception as e:
        logger.error(f"[IForest] Error applying Isolation Forest: {e}", exc_info=True)
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
        logger.debug(f"[DBSCAN] Shape of data_subset after dropna: {data_subset.shape}")
        if data_subset.empty or len(data_subset) < DBSCAN_MIN_SAMPLES:
             logger.warning(f"[DBSCAN] Not enough valid data points ({len(data_subset)}) for DBSCAN (min_samples={DBSCAN_MIN_SAMPLES}).")
             return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)
        logger.debug(f"[DBSCAN] Scaled data shape: {scaled_data.shape}")

        model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        predictions = model.fit_predict(scaled_data)
        anomalies = pd.Series(predictions == -1, index=data_subset.index)
        num_anomalies = anomalies.sum()
        logger.info(f"[DBSCAN] Detected {num_anomalies} anomalies out of {len(anomalies)} points.")
        return anomalies.reindex(df.index, fill_value=False)
    except Exception as e:
        logger.error(f"[DBSCAN] Error applying DBSCAN: {e}", exc_info=True)
        return None

def perform_stl_decomposition(series: pd.Series, period: int = STL_PERIOD) -> pd.DataFrame | None:
    """Performs STL decomposition on the series."""
    if not STATSMODELS_AVAILABLE or STL is None:
        logger.warning("Statsmodels not available or STL could not be imported. Skipping STL decomposition.")
        return None

    if series is None or series.isnull().all() or len(series.dropna()) < 2 * period:
        logger.warning(f"Not enough data ({len(series.dropna())}) for STL decomposition with period {period}. Required: {2 * period}")
        return None

    # --- Prepare Series Index (Revised Logic V2 - Incorporating QuantLib) ---
    series_clean = series.dropna().copy()
    if series_clean.empty or len(series_clean) < 2 * period:
        logger.warning(f"Not enough non-NaN data ({len(series_clean)}) for STL period {period}.")
        return None
        
    original_index = series_clean.index
    processed_series = series_clean
    freq = None

    if isinstance(series_clean.index, pd.DatetimeIndex):
        logger.debug("[STL Prep] Input has DatetimeIndex.")
        
        # === Try QuantLib NYSE Calendar First ===
        if QUANTLIB_AVAILABLE:
            logger.debug("[STL Prep] QuantLib available. Attempting NYSE calendar regularization.")
            try:
                cal = ql.UnitedStates(ql.UnitedStates.NYSE)
                start_date_ql = ql.Date(original_index.min().day, original_index.min().month, original_index.min().year)
                end_date_ql = ql.Date(original_index.max().day, original_index.max().month, original_index.max().year)
                
                business_days_ql = []
                current_date_ql = start_date_ql
                while current_date_ql <= end_date_ql:
                    if cal.isBusinessDay(current_date_ql):
                        # Use pd.Timestamp for easier conversion back to DatetimeIndex
                        business_days_ql.append(pd.Timestamp(current_date_ql.year(), current_date_ql.month(), current_date_ql.dayOfMonth()))
                    current_date_ql += ql.Period(1, ql.Days)
                
                if business_days_ql:
                    business_days_index = pd.DatetimeIndex(business_days_ql)
                    # Reindex the original cleaned series to the QL business days
                    reindexed_series = series_clean.reindex(business_days_index)
                    # Interpolate gaps created by reindexing
                    reindexed_series.interpolate(method='time', inplace=True)
                    # Check if it worked
                    if not reindexed_series.isnull().all() and len(reindexed_series.dropna()) >= 2 * period:
                         processed_series = reindexed_series
                         freq = 'B' # Assign business day frequency based on QL calendar
                         logger.info(f"[STL Prep] Successfully regularized index using QuantLib NYSE calendar. Freq='{freq}'. New length: {len(processed_series)}")
                    else:
                         logger.warning("[STL Prep] Reindexing/interpolating with QuantLib calendar resulted in all NaNs or insufficient length. Falling back.")
                         # Keep processed_series as series_clean, freq remains None
                else:
                    logger.warning("[STL Prep] QuantLib calendar generated no business days for the period. Falling back.")
            except Exception as e:
                logger.warning(f"[STL Prep] Error during QuantLib calendar processing: {e}. Falling back.")
                # Keep processed_series as series_clean, freq remains None
        else:
             logger.debug("[STL Prep] QuantLib not available.")
             
        # === Fallback to Pandas Frequency Inference (if QuantLib failed or not available) ===
        if freq is None: # Only if QuantLib didn't set the frequency
             logger.debug("[STL Prep] Attempting pandas frequency inference.")
             inferred_freq = pd.infer_freq(series_clean.index)
             logger.debug(f"[STL Prep] Inferred frequency: {inferred_freq}")
             if inferred_freq in ['B', 'D']:
                 try:
                     temp_series = series_clean.asfreq(inferred_freq)
                     temp_series.interpolate(method='time', inplace=True)
                     if not temp_series.isnull().all() and len(temp_series.dropna()) >= 2 * period:
                          processed_series = temp_series
                          freq = inferred_freq
                          logger.info(f"[STL Prep] Successfully regularized index using pandas infer/asfreq '{freq}'. New length: {len(processed_series)}")
                     else:
                          logger.warning(f"[STL Prep] Regularizing with pandas asfreq('{inferred_freq}') failed or insufficient length. Reverting.")
                          processed_series = series_clean # Revert to original cleaned data
                          freq = None
                 except Exception as e:
                     logger.warning(f"[STL Prep] Failed pandas asfreq('{inferred_freq}'): {e}. Reverting.")
                     processed_series = series_clean # Revert
                     freq = None
             else:
                 logger.warning(f"[STL Prep] Inferred frequency ('{inferred_freq}') is not B or D.")
                 # We might still have a valid DatetimeIndex, just not daily/business
                 processed_series = series_clean 
                 # Keep freq as None unless infer_freq returned something usable by STL? Maybe not.
                 freq = None # Safer to treat irregular freq as needing integer index later
            
    # === Handle Non-DatetimeIndex or Conversion ===
    else: # Input was not DatetimeIndex
        logger.debug("[STL Prep] Input index is not DatetimeIndex. Attempting conversion.")
        try:
            datetime_index = pd.to_datetime(series_clean.index)
            processed_series.index = datetime_index
            logger.info("[STL Prep] Successfully converted index to DatetimeIndex.")
            # Now, attempt QuantLib / Pandas infer/asfreq on the *converted* index
            
            # === Try QuantLib NYSE Calendar First (After Conversion) ===
            if QUANTLIB_AVAILABLE:
                logger.debug("[STL Prep] (Post-Convert) Attempting QL NYSE calendar regularization.")
                try:
                    cal = ql.UnitedStates(ql.UnitedStates.NYSE)
                    current_original_index = processed_series.index # Use the newly converted index
                    start_date_ql = ql.Date(current_original_index.min().day, current_original_index.min().month, current_original_index.min().year)
                    end_date_ql = ql.Date(current_original_index.max().day, current_original_index.max().month, current_original_index.max().year)
                    
                    business_days_ql = []
                    current_date_ql = start_date_ql
                    while current_date_ql <= end_date_ql:
                        if cal.isBusinessDay(current_date_ql):
                            business_days_ql.append(pd.Timestamp(current_date_ql.year(), current_date_ql.month(), current_date_ql.dayOfMonth()))
                        current_date_ql += ql.Period(1, ql.Days)
                    
                    if business_days_ql:
                        business_days_index = pd.DatetimeIndex(business_days_ql)
                        # Use the current processed_series for reindexing
                        reindexed_series = processed_series.reindex(business_days_index)
                        reindexed_series.interpolate(method='time', inplace=True)
                        if not reindexed_series.isnull().all() and len(reindexed_series.dropna()) >= 2 * period:
                             processed_series = reindexed_series
                             freq = 'B'
                             logger.info(f"[STL Prep] (Post-Convert) Successfully regularized using QL NYSE calendar. Freq='{freq}'. Length: {len(processed_series)}")
                        else:
                             logger.warning("[STL Prep] (Post-Convert) QL reindexing failed or insufficient length. Falling back.")
                             # Freq remains None
                    else:
                        logger.warning("[STL Prep] (Post-Convert) QL calendar generated no business days. Falling back.")
                except Exception as e:
                    logger.warning(f"[STL Prep] (Post-Convert) Error during QL processing: {e}. Falling back.")
                    # Freq remains None
            else:
                 logger.debug("[STL Prep] (Post-Convert) QuantLib not available.")
                 
            # === Fallback to Pandas Frequency Inference (After Conversion) ===
            if freq is None: 
                 logger.debug("[STL Prep] (Post-Convert) Attempting pandas frequency inference.")
                 inferred_freq = pd.infer_freq(processed_series.index) # Use the converted index
                 logger.debug(f"[STL Prep] (Post-Convert) Inferred frequency: {inferred_freq}")
                 if inferred_freq in ['B', 'D']:
                     try:
                         temp_series = processed_series.asfreq(inferred_freq)
                         temp_series.interpolate(method='time', inplace=True)
                         if not temp_series.isnull().all() and len(temp_series.dropna()) >= 2 * period:
                              processed_series = temp_series
                              freq = inferred_freq
                              logger.info(f"[STL Prep] (Post-Convert) Successfully regularized using pandas '{freq}'. Length: {len(processed_series)}")
                         else:
                              logger.warning(f"[STL Prep] (Post-Convert) Pandas asfreq('{inferred_freq}') failed or insufficient length. Reverting.")
                              processed_series = processed_series # Keep the converted index, but freq is None
                              freq = None
                     except Exception as e:
                         logger.warning(f"[STL Prep] (Post-Convert) Failed pandas asfreq('{inferred_freq}'): {e}. Reverting.")
                         processed_series = processed_series # Keep converted index
                         freq = None
                 else:
                     logger.warning(f"[STL Prep] (Post-Convert) Inferred frequency ('{inferred_freq}') is not B or D.")
                     freq = None # Treat irregular freq as needing integer index later
                 
        except Exception as e:
            logger.warning(f"[STL Prep] Could not convert index to DatetimeIndex: {e}. Will use integer index.")
            processed_series = series_clean.reset_index(drop=True)
            freq = None 

    # --- Final Checks and Fallback to Integer Index ---
    if freq is None or len(processed_series.dropna()) < 2 * period:
        if freq is None:
             logger.warning("[STL Prep] Could not establish a regular frequency or convert index. Falling back to integer index.")
        else:
             logger.warning(f"[STL Prep] Series too short ({len(processed_series.dropna())}) after frequency regularization. Falling back to integer index.")
             
        processed_series = series_clean.reset_index(drop=True)
        freq = None
        logger.info(f"[STL Prep] Using integer index. Length: {len(processed_series)}")
        if len(processed_series) < 2 * period:
             logger.error(f"[STL Prep] Data length ({len(processed_series)}) insufficient for STL period {period} even with integer index.")
             return None
             
    # --- Interpolate/Fill any remaining NaNs ---
    interp_method = 'time' if freq else 'linear' # Use time for DatetimeIndex, linear for Int index
    processed_series.interpolate(method=interp_method, limit_direction='both', inplace=True)
    processed_series.ffill(inplace=True)
    processed_series.bfill(inplace=True)

    if processed_series.isnull().any():
        logger.error("[STL Prep] Series still contains NaNs after final interpolation/fill. Cannot perform STL.")
        return None # STL requires complete data
        
    # --- Perform STL ---
    try:
        logger.info(f"Performing STL decomposition with period={period}, frequency='{freq if freq else 'Integer'}', length={len(processed_series)}");
        # Pass freq ONLY if it's a valid DatetimeIndex frequency string
        # If we fell back to integer index, STL doesn't need/use freq
        stl_kwargs = {'period': period, 'robust': True}
        # STL might handle integer index directly, or we let it use internal defaults without freq
        
        stl_instance = STL(processed_series, **stl_kwargs)
        result = stl_instance.fit()
        
        # === DEBUGGING: Check residuals immediately after fit ===
        residual_nan_count = result.resid.isnull().sum() if result and hasattr(result, 'resid') else 'N/A'
        logger.debug(f"[STL Debug] Residuals directly after STL fit - NaN count: {residual_nan_count}")
        if result and hasattr(result, 'resid') and not result.resid.isnull().all():
             logger.debug(f"[STL Debug] Residuals head:\n{result.resid.head().to_string()}")
        else:
             logger.debug("[STL Debug] Residuals are None, do not exist, or are all NaN immediately after fit.")
        # =======================================================

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
            elif isinstance(x_data.index, pd.RangeIndex):
                x_plot = x_data.index
                x_calc_numeric = x_data.index.values
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
        elif isinstance(x_data, pd.RangeIndex):
            x_plot = x_data
            x_calc_numeric = x_data.values
            original_x_index = x_data
        # Add handling for generic Index before list/array
        elif isinstance(x_data, pd.Index): # Check for generic pd.Index (that isn't DatetimeIndex/RangeIndex)
            logger.debug(f"Handling generic pd.Index for LOWESS x_data {name_suffix}. Using index for plot, positional for calc.")
            x_plot = x_data
            x_calc_numeric = np.arange(len(x_data))
            # Use y_series's index if x_data was just an index
            original_x_index = y_series.index if y_series.index.size == x_data.size else None
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
        valid_anomaly_idx = series.index.intersection(all_anomalies_idx)
        logger.debug(f"[Plot Z Anom] Found {len(valid_anomaly_idx)} valid Z/ModZ anomalies for {series_name}.")

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
        logger.debug(f"[Plot Cluster {method_name}] Shape of input df: {data_subset.shape}, shape of anomaly_series: {anomaly_series.shape}, common index size: {len(data_subset.index.intersection(anomaly_series.index))}")
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
        logger.debug(f"[Plot Cluster {method_name}] Shape after alignment/masking: {data_subset.shape}")

        if data_subset.empty or len(data_subset) < 2:
            logger.warning(f"Not enough valid, aligned data points for {method_name} PCA plot ({symbol}).")
            return None

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset)

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(scaled_data)

        plot_df = pd.DataFrame(pca_results, columns=['PCA1', 'PCA2'], index=data_subset.index)
        plot_df['Anomaly'] = aligned_anomalies # Add the boolean anomaly flag
        num_plot_anomalies = plot_df['Anomaly'].sum()
        logger.info(f"[Plot Cluster {method_name}] Plotting {len(plot_df)} points with {num_plot_anomalies} anomalies.")

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
        logger.error(f"[Plot Cluster {method_name}] Error creating PCA plot for {symbol}: {e}", exc_info=True)
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
    """Runs all anomaly detection analyses and returns results."""
    logger.info(f"Starting anomaly analysis for {symbol}...")
    if df is None or df.empty or 'close' not in df.columns:
        logger.warning(f"Insufficient data for {symbol} to run anomaly analyses. Input df shape: {df.shape if df is not None else 'None'}")
        return {}
        
    results = {}
    df_analysis = df.copy()
    logger.debug(f"Initial df_analysis shape: {df_analysis.shape}. Head:\n{df_analysis.head().to_string()}")

    # --- Calculate necessary base series (handle potential NaNs later) ---
    if 'log_return' not in df_analysis.columns:
        # Ensure close price is numeric and positive before taking log
        close_numeric = pd.to_numeric(df_analysis['close'], errors='coerce')
        close_positive = close_numeric[close_numeric > 0]
        if not close_positive.empty:
             df_analysis['log_return'] = np.log(close_positive / close_positive.shift(1))
        else:
             df_analysis['log_return'] = np.nan # Assign NaN if no valid close prices
        logger.debug(f"Calculated log_return. Shape: {df_analysis.shape}. NaN count: {df_analysis['log_return'].isnull().sum() if 'log_return' in df_analysis else 'N/A'}")
             
    if 'volatility' not in df_analysis.columns and 'log_return' in df_analysis.columns:
        df_analysis['volatility'] = df_analysis['log_return'].rolling(window=21).std() * np.sqrt(252)
        logger.debug(f"Calculated volatility. Shape: {df_analysis.shape}. NaN count: {df_analysis['volatility'].isnull().sum() if 'volatility' in df_analysis else 'N/A'}")
        
    if 'volume_pct_change' not in df_analysis.columns and 'volume' in df_analysis.columns:
         # Ensure volume is numeric before calculating pct_change
         volume_numeric = pd.to_numeric(df_analysis['volume'], errors='coerce')
         df_analysis['volume_pct_change'] = volume_numeric.pct_change()
         logger.debug(f"Calculated volume_pct_change. Shape: {df_analysis.shape}. NaN count: {df_analysis['volume_pct_change'].isnull().sum() if 'volume_pct_change' in df_analysis else 'N/A'}")
         
    # REMOVED global dropna call
    # df_analysis.dropna(subset=['log_return', 'volatility', 'volume_pct_change'], how='any', inplace=True)

    # --- 1. Z-Score Analysis (Functions handle internal NaNs) ---
    logger.info(f"Calculating Z-scores for {symbol}...")
    # Use the potentially NaN-containing series from df_analysis
    close_z, close_mod_z = calculate_z_scores(df_analysis['close'])
    vol_z, vol_mod_z = calculate_z_scores(df_analysis['volume']) if 'volume' in df_analysis else (pd.Series(dtype=float), pd.Series(dtype=float))
    logret_z, logret_mod_z = calculate_z_scores(df_analysis['log_return']) if 'log_return' in df_analysis and not df_analysis['log_return'].isnull().all() else (pd.Series(dtype=float), pd.Series(dtype=float)) # Added check for all NaN
    logger.debug(f"Z-Score Results - Close NaNs: {close_z.isnull().sum()}, Vol NaNs: {vol_z.isnull().sum()}, LogRet NaNs: {logret_z.isnull().sum()}")

    # Plotting functions also handle potential NaNs in scores
    results['zscore_plots'] = plot_z_scores(close_z, close_mod_z, symbol, 'Close Price')
    results['zscore_vol_plots'] = plot_z_scores(vol_z, vol_mod_z, symbol, 'Volume')
    results['log_returns_z_score_plot'] = plot_z_scores(logret_z, logret_mod_z, symbol, 'Log Returns') # Added check for all NaN

    results['close_price_with_z_anomalies'] = plot_data_with_z_anomalies(df_analysis['close'], close_z, close_mod_z, symbol, 'Close Price')
    results['volume_with_z_anomalies'] = plot_data_with_z_anomalies(df_analysis['volume'], vol_z, vol_mod_z, symbol, 'Volume') if 'volume' in df_analysis else None
    results['log_returns_z_anomalies_plot'] = plot_data_with_z_anomalies(df_analysis['log_return'], logret_z, logret_mod_z, symbol, 'Log Returns') if 'log_return' in df_analysis else None

    # --- 2. Model-Based Anomaly Detection (Isolation Forest, DBSCAN) ---
    if SKLEARN_AVAILABLE:
        features_for_pca = ['log_return', 'volatility', 'volume_pct_change']
        available_features = [f for f in features_for_pca if f in df_analysis.columns]
        logger.debug(f"Features available for clustering: {available_features}")
        
        if len(available_features) >= 2: # Need at least 2 features for clustering/PCA
            df_pca_subset = df_analysis[available_features].copy()
            logger.debug(f"Created df_pca_subset. Shape before dropna: {df_pca_subset.shape}")
            df_pca_subset.dropna(how='any', inplace=True)
            logger.info(f"df_pca_subset shape after dropna: {df_pca_subset.shape}")
            
            min_samples_needed = max(2, DBSCAN_MIN_SAMPLES)
            if not df_pca_subset.empty and len(df_pca_subset) >= min_samples_needed:
                 logger.info(f"Applying Isolation Forest for {symbol} using features: {available_features} on {len(df_pca_subset)} points...")
                 iforest_anomalies = apply_isolation_forest(df_pca_subset, available_features) # Pass the *cleaned* subset
                 if iforest_anomalies is not None:
                     # Align anomalies back to the pca subset index for plotting
                     iforest_anomalies_aligned = iforest_anomalies.reindex(df_pca_subset.index)
                     if not iforest_anomalies_aligned.isnull().all(): # Check alignment didn't fail
                         results['iforest_pca_plot'] = plot_clustering_anomalies(
                             df_pca_subset, available_features, iforest_anomalies_aligned, "Isolation Forest", symbol
                         )
                     else: logger.warning(f"Isolation Forest anomaly alignment failed for {symbol}")
                 else: logger.warning(f"Isolation Forest returned None for {symbol}.")

                 logger.info(f"Applying DBSCAN for {symbol} using features: {available_features} on {len(df_pca_subset)} points...")
                 dbscan_anomalies = apply_dbscan(df_pca_subset, available_features) # Pass the *cleaned* subset
                 if dbscan_anomalies is not None:
                     dbscan_anomalies_aligned = dbscan_anomalies.reindex(df_pca_subset.index)
                     if not dbscan_anomalies_aligned.isnull().all():
                         results['dbscan_pca_plot'] = plot_clustering_anomalies(
                             df_pca_subset, available_features, dbscan_anomalies_aligned, "DBSCAN", symbol
                         )
                     else: logger.warning(f"DBSCAN anomaly alignment failed for {symbol}")
                 else: logger.warning(f"DBSCAN returned None for {symbol}.")
            else:
                logger.warning(f"Skipping Isolation Forest/DBSCAN for {symbol} because not enough valid data points ({len(df_pca_subset)}) remained in subset after dropping NaNs for features: {available_features}. Need at least {min_samples_needed}.")
        else:
            logger.warning(f"Skipping Isolation Forest/DBSCAN for {symbol} due to insufficient available base features ({len(available_features)} < 2) from {features_for_pca}")

    # --- 3. STL Decomposition (Function handles internal NaNs) ---
    if STATSMODELS_AVAILABLE and STL is not None and 'close' in df_analysis and not df_analysis['close'].isnull().all(): # Added check for all NaN
        logger.info(f"Performing STL decomposition for {symbol} close price...")
        stl_period = STL_PERIOD # Could make this adaptive or configurable
        decomp_close = perform_stl_decomposition(df_analysis['close'], period=stl_period)
        if decomp_close is not None:
            logger.debug(f"STL decomposition successful. Result shape: {decomp_close.shape}")
            results['stl_decomposition_plot'] = plot_stl_decomposition(decomp_close, symbol)
            
            # Calculate Z-scores on the residual component
            stl_resid_z, stl_resid_mod_z = calculate_z_scores(decomp_close['resid'])
            results['stl_residual_with_z_anomalies'] = plot_data_with_z_anomalies(
                decomp_close['resid'], stl_resid_z, stl_resid_mod_z, symbol, 'STL Residuals'
            )
            
            # Plot Z-scores of STL residuals
            results['stl_residual_zscore_plots'] = plot_z_scores(
                stl_resid_z, stl_resid_mod_z, symbol, 'STL Residuals'
             )
        else:
            logger.warning(f"STL decomposition failed or returned None for {symbol}.")

    logger.info(f"Completed Anomaly Analysis for {symbol}. Generated plot keys: {list(results.keys())}")
    # Filter out None results before returning
    final_results = {k: v for k, v in results.items() if v is not None}
    logger.info(f"Returning {len(final_results)} non-None anomaly plots for {symbol}.")
    return final_results

# --- FastAPI Router ---
# ... existing code ... 