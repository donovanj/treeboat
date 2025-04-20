import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
from scipy import signal # For windowing and peak finding
import pywt # PyWavelets for DWT, CWT etc.
from astropy.timeseries import LombScargle # For unevenly spaced data

# --- Helper Functions ---

def _preprocess_data(series: pd.Series, detrend: str | None = None, normalize: bool = False) -> pd.Series:
    """Applies preprocessing steps like detrending and normalization."""
    processed = series.copy()
    if detrend:
        processed = signal.detrend(processed.values, type=detrend)
        processed = pd.Series(processed, index=series.index) # Restore index

    if normalize:
        processed = (processed - processed.mean()) / processed.std()

    # Add other preprocessing steps here if needed (e.g., differencing)
    return processed

def calculate_log_returns(data: pd.DataFrame, col: str = 'close') -> pd.Series:
    """Calculates log returns for a specific column, handling potential zeros."""
    prices = data[col]
    # Shift and calculate log returns, replace inf/-inf with NaN
    log_returns = np.log(prices / prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return log_returns.dropna()

# --- Spectral Analysis Functions ---

def plot_fft(
    data: pd.DataFrame,
    target_col: str = 'close',
    detrend: str | None = 'linear', # Options: 'linear', 'constant', None
    window: str = 'hann', # Options: 'hann', 'hamming', 'blackman', None
    peak_prominence: float = 0.1, # Adjust based on expected peak significance relative to neighbors
    peak_height: float | None = None, # Adjust based on minimum absolute power threshold
    n_peaks: int = 5,
    plot_title: str = "FFT Power Spectrum"
) -> go.Figure:
    """
    Calculates and plots the FFT Power Spectrum of log returns.
    Includes options for detrending, windowing, and robust peak detection.
    NOTE: Assumes data is reasonably evenly spaced (e.g., daily).
          Use Lomb-Scargle for significantly uneven data.
    """
    log_returns = calculate_log_returns(data, target_col)
    if log_returns.empty:
        return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Not enough data'})

    # --- Preprocessing ---
    processed_returns = _preprocess_data(log_returns, detrend=detrend)
    N = len(processed_returns)
    if N < 2: # Need at least 2 points for FFT
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Data too short after processing'})

    # --- Windowing ---
    if window:
        window_func = signal.get_window(window, N)
        processed_returns = processed_returns * window_func

    # --- FFT Calculation ---
    # Frequency axis for daily data (sampling_period=1 day)
    sampling_period = 1.0
    yf = fft(processed_returns.values)
    xf = fftfreq(N, sampling_period)[:N//2] # Frequencies (cycles/day)

    # Power Spectrum (amplitude squared, scaled)
    # Factor 2.0/N for single-sided spectrum amplitude, then square for power
    # For windowed data, need to scale by window energy: 1.0 / (fs * (window_func**2).sum())? Let's use simpler scaling for now.
    power = 2.0/N * np.abs(yf[0:N//2])
    # power = (1.0 / (sampling_period * N)) * np.abs(yf[0:N//2])**2 # Alternative power spectral density scaling


    # --- Peak Detection ---
    peak_indices, properties = signal.find_peaks(power, prominence=peak_prominence, height=peak_height)

    # Sort peaks by prominence (or height) and take top N
    if len(peak_indices) > 0:
        prominences = properties.get('prominences', np.zeros_like(peak_indices)) # Handle cases where prominence is None
        sorted_peak_indices = peak_indices[np.argsort(prominences)[::-1]]
        top_peak_indices = sorted_peak_indices[:min(n_peaks, len(sorted_peak_indices))]

        peak_freqs = xf[top_peak_indices]
        # Avoid division by zero for frequency 0
        peak_periods = np.full_like(peak_freqs, np.inf)
        non_zero_freq_mask = peak_freqs != 0
        peak_periods[non_zero_freq_mask] = 1.0 / peak_freqs[non_zero_freq_mask] # In days
        peak_power = power[top_peak_indices]
    else:
        top_peak_indices = []
        peak_freqs = []
        peak_periods = []
        peak_power = []

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=power, mode='lines', name='Power Spectrum'))
    fig.add_trace(go.Scatter(x=peak_freqs, y=peak_power, mode='markers', name='Detected Peaks',
                           marker=dict(color='red', size=8, symbol='x')))

    annotations = []
    for i in range(len(peak_freqs)):
         annotations.append(dict(
            x=peak_freqs[i],
            y=peak_power[i],
            xref="x", yref="y",
            text=f"Period: {peak_periods[i]:.1f}d",
            showarrow=True, arrowhead=1, ax=-20, ay=-40,
            bgcolor="rgba(255, 255, 255, 0.7)"
        ))

    # Construct title with parameters
    title_parts = [plot_title, f"({target_col.capitalize()})"]
    if detrend: title_parts.append(f"Detrend: {detrend}")
    if window: title_parts.append(f"Window: {window}")
    full_title = ", ".join(title_parts)

    fig.update_layout(
        title=full_title,
        xaxis_title="Frequency (cycles/day)",
        yaxis_title="Power (Amplitude Scaled)",
        yaxis_type="log",
        annotations=annotations,
        hovermode='x unified'
    )
    # Add comment about lack of statistical significance testing
    # TODO: Implement statistical significance testing for peaks (e.g., based on red noise model)
    return fig

def plot_dwt(
    data: pd.DataFrame,
    target_col: str = 'close',
    wavelet: str = 'db4',
    level: int | None = None,
    detrend: str | None = 'linear',
    scale_coeffs: bool = False, # Option to scale coefficients per level
    plot_title: str = "Discrete Wavelet Transform (DWT)"
) -> go.Figure:
    """
    Performs and plots the Discrete Wavelet Transform (DWT) coefficients.
    Calculates multi-level DWT up to the maximum possible level or specified level.
    Includes optional detrending and coefficient scaling.
    """
    log_returns = calculate_log_returns(data, target_col)
    if log_returns.empty:
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Not enough data'})

    # --- Preprocessing ---
    processed_returns = _preprocess_data(log_returns, detrend=detrend)
    N = len(processed_returns)

    # Determine max level or use specified level
    try:
        w = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(data_len=N, filter_len=w.dec_len)
    except ValueError:
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Invalid wavelet name: {wavelet}'})

    if max_level < 1:
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Data too short for level 1 DWT'})

    current_level = min(level, max_level) if level is not None and level >= 1 else max_level
    if current_level < 1: current_level = 1 # Ensure at least level 1 if possible

    # --- DWT Calculation ---
    coeffs = pywt.wavedec(processed_returns, wavelet, level=current_level)

    # --- Approximate Period Ranges ---
    # Based on dyadic scales: scale = 2^level. Period ~ scale.
    # These are rough guides, actual frequencies depend on wavelet shape.
    approx_periods = [f"~ {2**j}-{2**(j+1)} days" for j in range(current_level, 0, -1)]
    approx_title = f'Approximation (cA{current_level}, Period > ~{2**current_level} days)'
    detail_titles = [f'Detail (cD{current_level-i}, Period {approx_periods[i]})' for i in range(current_level)]

    # --- Plotting ---
    fig = make_subplots(rows=current_level + 1, cols=1, shared_xaxes=True,
                        subplot_titles=[approx_title] + detail_titles)

    # Plot Approximation Coefficients (cA)
    cA = coeffs[0]
    if scale_coeffs: cA = (cA - np.mean(cA)) / np.std(cA) if np.std(cA) > 1e-8 else cA
    fig.add_trace(go.Scatter(y=cA, mode='lines', name=f'cA{current_level}'), row=1, col=1)

    # Plot Detail Coefficients (cD)
    for i in range(current_level):
        cD = coeffs[i+1]
        # Scale detail coefficients for better visualization if requested
        if scale_coeffs: cD = (cD - np.mean(cD)) / np.std(cD) if np.std(cD) > 1e-8 else cD
        fig.add_trace(go.Scatter(y=cD, mode='lines', name=f'cD{current_level-i}'), row=i+2, col=1)

    # Construct title
    title_parts = [plot_title, f"({target_col.capitalize()}, {wavelet}, Level {current_level})"]
    if detrend: title_parts.append(f"Detrend: {detrend}")
    full_title = ", ".join(title_parts)

    fig.update_layout(
        title=full_title,
        showlegend=False,
        height=150 * (current_level + 1) # Adjust height based on levels
    )
    fig.update_xaxes(title_text="Sample Index", row=current_level + 1, col=1)

    return fig


def plot_cwt(
    data: pd.DataFrame,
    target_col: str = 'close',
    wavelet: str = 'morl', # Morlet is common for CWT
    detrend: str | None = 'linear',
    min_period: float = 2.0,
    max_period: float | None = None, # If None, set to N/4
    num_scales: int = 100, # Number of scales to evaluate
    plot_title: str = "Continuous Wavelet Transform (CWT) Scalogram"
) -> go.Figure:
    """
    Calculates and plots the Continuous Wavelet Transform (CWT) using a scalogram.
    Includes optional detrending and adaptive scale selection.
    """
    log_returns = calculate_log_returns(data, target_col)
    if log_returns.empty:
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Not enough data'})

    # --- Preprocessing ---
    processed_returns = _preprocess_data(log_returns, detrend=detrend)
    N = len(processed_returns)
    time_index = processed_returns.index # Keep original time index for plotting

    # --- Adaptive Scale Selection ---
    # sampling_period=1.0 for daily data
    sampling_period = 1.0
    if max_period is None:
        max_period = N / 4 # Heuristic: max period shouldn't exceed 1/4th data length

    # Define scales logarithmically for better coverage of periods
    # scale = period * fourier_factor / sampling_period
    # Need central frequency for the wavelet to map scales to periods accurately.
    # pywt.central_frequency(wavelet) gives this, but let's approximate for now
    # Using np.geomspace to get logarithmically spaced scales roughly corresponding to periods
    # This mapping isn't perfect and depends on the wavelet. Morlet factor is ~1.
    scales = np.geomspace(min_period, max_period, num=num_scales)

    if N < int(max(scales)): # Check if data is long enough for largest scale
         print(f"Warning: Max scale ({max(scales):.1f}) might be too large for data length ({N}). Adjusting max_period.")
         max_period = N / 2 # Reduce max_period slightly
         scales = np.geomspace(min_period, max_period, num=num_scales)
         if N < int(max(scales)): # Still too short
             return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Data too short for chosen scales'})

    # --- CWT Calculation ---
    try:
        coefficients, frequencies = pywt.cwt(processed_returns, scales, wavelet, sampling_period=sampling_period)
    except ValueError as e:
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - CWT Error: {e}'})

    # Power is magnitude squared
    power = (np.abs(coefficients)) ** 2

    # Periods corresponding to frequencies (more accurate than using scales directly)
    # Filter out potential zero frequencies if they occur
    valid_freq_mask = frequencies > 0
    periods = np.full_like(frequencies, np.nan)
    periods[valid_freq_mask] = 1. / frequencies[valid_freq_mask]

    # --- Plotting ---
    # Ensure power and periods match dimensions and filtering
    power = power[valid_freq_mask, :]
    periods = periods[valid_freq_mask]

    if len(periods) == 0:
         return go.Figure(layout={'title': f'{plot_title} ({target_col}) - No valid frequencies/periods found'})

    fig = go.Figure(data=go.Heatmap(
        z=power,
        x=time_index, # Use original time index for x-axis
        y=periods,
        colorscale='Viridis',
        colorbar=dict(title='Power'),
        hoverongaps=False
    ))

    # Construct title
    title_parts = [plot_title, f"({target_col.capitalize()}, {wavelet})"]
    if detrend: title_parts.append(f"Detrend: {detrend}")
    full_title = ", ".join(title_parts)

    fig.update_layout(
        title=full_title,
        xaxis_title='Date',
        yaxis_title='Period (days)',
        yaxis_type='log', # Log scale for periods is standard
        yaxis_autorange='reversed' # Show shorter periods at the top
    )

    return fig

def plot_lomb_scargle(
    data: pd.DataFrame,
    target_col: str = 'close',
    time_col: str | None = None, # Optional: specify time column if not index
    detrend: str | None = 'linear', # Linear detrending often useful here
    min_period: float = 2.0,
    max_period: float | None = None, # If None, defaults based on data span
    samples_per_peak: int = 10, # Controls frequency resolution
    fap_levels: list[float] = [0.1, 0.05, 0.01], # False Alarm Probability levels
    plot_title: str = "Lomb-Scargle Periodogram"
) -> go.Figure:
    """
    Calculates and plots the Lomb-Scargle Periodogram, suitable for unevenly spaced data.
    Includes detrending and False Alarm Probability levels.
    """
    if target_col not in data.columns:
        return go.Figure(layout={'title': f'{plot_title} - Target column "{target_col}" not found'})

    # --- Prepare Data ---
    values = data[target_col].dropna()
    if time_col:
        if time_col not in data.columns:
            return go.Figure(layout={'title': f'{plot_title} - Time column "{time_col}" not found'})
        # Convert time to numerical representation (e.g., days since start)
        time_stamps = pd.to_datetime(data.loc[values.index, time_col])
        times = (time_stamps - time_stamps.min()).dt.total_seconds() / (24 * 3600) # Days
    else:
        # Assume index is time-like, convert to numeric days
        time_stamps = pd.to_datetime(values.index)
        times = (time_stamps - time_stamps.min()).dt.total_seconds() / (24 * 3600) # Days

    values = values.values # Get numpy array
    if len(values) < 3:
        return go.Figure(layout={'title': f'{plot_title} ({target_col}) - Not enough data points'})

    # --- Preprocessing ---
    values = _preprocess_data(pd.Series(values), detrend=detrend).values # Detrend values only

    # --- Frequency Grid ---
    t_span = times.max() - times.min()
    if max_period is None:
        max_period = t_span / 2 # Nyquist-like limit for uneven data

    # Define frequency range based on period range
    min_freq = 1.0 / max_period if max_period > 0 else 1e-3 # Avoid division by zero
    max_freq = 1.0 / min_period if min_period > 0 else 1.0

    # --- Lomb-Scargle Calculation ---
    ls = LombScargle(times, values)
    # frequency, power = ls.autopower(minimum_frequency=min_freq, maximum_frequency=max_freq, samples_per_peak=samples_per_peak)
    # Use linspace for frequency to control range more directly
    frequency = np.linspace(min_freq, max_freq, int(samples_per_peak * t_span * (max_freq-min_freq))) # Heuristic for num points
    power = ls.power(frequency)

    # --- False Alarm Probability ---
    faps = {}
    if fap_levels:
        try:
            fap_powers = ls.false_alarm_level(fap_levels)
            faps = dict(zip(fap_levels, fap_powers))
        except Exception as e:
            print(f"Warning: Could not calculate FAP levels: {e}")


    periods = 1.0 / frequency

    # --- Plotting ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=periods, y=power, mode='lines', name='LS Power'))

    # Add FAP level lines
    colors = ['red', 'orange', 'green']
    for i, (fap, p_val) in enumerate(faps.items()):
        fig.add_hline(y=p_val, line_dash="dash", line_color=colors[i % len(colors)],
                      annotation_text=f"FAP {fap*100:.0f}%",
                      annotation_position="bottom right")

    # Identify significant peaks (above highest FAP threshold if available)
    significant_threshold = max(faps.values()) if faps else None
    if significant_threshold is not None:
        peak_indices, _ = signal.find_peaks(power, height=significant_threshold)
        if len(peak_indices) > 0:
             fig.add_trace(go.Scatter(x=periods[peak_indices], y=power[peak_indices], mode='markers',
                                    name=f'Peaks (>{list(faps.keys())[-1]*100:.0f}% FAP)',
                                    marker=dict(color='purple', size=8, symbol='x')))


    # Construct title
    title_parts = [plot_title, f"({target_col.capitalize()})"]
    if detrend: title_parts.append(f"Detrend: {detrend}")
    full_title = ", ".join(title_parts)

    fig.update_layout(
        title=full_title,
        xaxis_title="Period (days)",
        yaxis_title="Lomb-Scargle Power",
        xaxis_type="log", # Often view periods on log scale
        # yaxis_type="log", # Power can also be log-scaled if needed
        hovermode='x unified'
    )

    return fig


# --- Placeholder for other advanced spectral methods ---

# def plot_mtm(data: pd.DataFrame, ...):
#     # Multi-Taper Method - requires 'spectrum' library or similar
#     # Good for reducing variance in spectral estimates
#     pass

# def plot_emd(data: pd.DataFrame, ...):
#     # Empirical Mode Decomposition - requires 'PyEMD' library
#     # Good for non-stationary, non-linear data - decomposes into Intrinsic Mode Functions (IMFs)
#     pass

# def plot_wavelet_coherence(data1: pd.Series, data2: pd.Series, ...):
#     # Wavelet Coherence - requires 'pycwt' or similar
#     # Analyzes frequency-specific correlation between two time series over time
#     pass

# TODO: Add Cross-Spectrum analysis function (using FFT results of two series)
# TODO: Add Wigner-Ville Distribution (higher-order time-frequency analysis) 