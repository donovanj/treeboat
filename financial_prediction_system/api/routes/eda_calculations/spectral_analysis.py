import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
import pywt # PyWavelets for DWT, CWT etc.
# Consider adding: from astropy.timeseries import LombScargle for Lomb-Scargle

def calculate_log_returns(data: pd.DataFrame):
    """Calculates log returns, handling potential zeros."""
    close_prices = data['close']
    # Shift and calculate log returns, replace -inf with NaN
    log_returns = np.log(close_prices / close_prices.shift(1)).replace([np.inf, -np.inf], np.nan)
    return log_returns.dropna()

def plot_fft(data: pd.DataFrame, plot_title: str = "FFT Power Spectrum (Log Returns)") -> go.Figure:
    """
    Calculates and plots the FFT Power Spectrum of log returns.
    """
    log_returns = calculate_log_returns(data)
    if log_returns.empty:
        return go.Figure(layout={'title': f'{plot_title} - Not enough data'})

    N = len(log_returns)
    yf = fft(log_returns.values)
    xf = fftfreq(N, 1)[:N//2] # Assuming daily data (frequency unit = 1/day)

    power = 2.0/N * np.abs(yf[0:N//2])

    # Identify dominant frequencies (excluding DC component at index 0)
    # Find indices of peaks (simple approach, consider more robust peak finding)
    if len(power) > 1:
        peak_indices = np.argsort(power[1:])[-5:] + 1 # Top 5 peaks excluding DC
        peak_freqs = xf[peak_indices]
        peak_periods = 1.0 / peak_freqs # In days
        peak_power = power[peak_indices]
    else:
        peak_freqs = []
        peak_periods = []
        peak_power = []


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=power, mode='lines', name='Power Spectrum'))

    # Add annotations for dominant periods
    annotations = []
    for i in range(len(peak_freqs)):
         annotations.append(dict(
            x=peak_freqs[i],
            y=peak_power[i],
            xref="x",
            yref="y",
            text=f"Period: {peak_periods[i]:.1f} days",
            showarrow=True,
            arrowhead=1,
            ax=-20,
            ay=-40
        ))

    fig.update_layout(
        title=plot_title,
        xaxis_title="Frequency (cycles/day)",
        yaxis_title="Power",
        yaxis_type="log", # Often better to view power spectrum on log scale
        annotations=annotations
    )
    return fig

def plot_dwt(data: pd.DataFrame, wavelet='db4', level=None, plot_title: str = "Discrete Wavelet Transform (DWT)") -> go.Figure:
    """
    Performs and plots the Discrete Wavelet Transform (DWT) coefficients.
    Calculates multi-level DWT up to the maximum possible level or specified level.
    """
    log_returns = calculate_log_returns(data)
    if log_returns.empty:
         return go.Figure(layout={'title': f'{plot_title} - Not enough data'})

    # Determine max level or use specified level
    max_level = pywt.dwt_max_level(data_len=len(log_returns), filter_len=pywt.Wavelet(wavelet).dec_len)
    current_level = min(level, max_level) if level is not None else max_level
    if current_level < 1:
         return go.Figure(layout={'title': f'{plot_title} - Data too short for DWT'})

    coeffs = pywt.wavedec(log_returns, wavelet, level=current_level)

    # Create subplots: one for approximation (cA) and one for each detail level (cD)
    fig = make_subplots(rows=current_level + 1, cols=1, shared_xaxes=True,
                        subplot_titles=[f'Approximation (cA{current_level})'] + [f'Detail (cD{i})' for i in range(current_level, 0, -1)])

    # Plot Approximation Coefficients (cA) - Scale appropriately if needed for visualization
    fig.add_trace(go.Scatter(y=coeffs[0], mode='lines', name=f'cA{current_level}'), row=1, col=1)

    # Plot Detail Coefficients (cD)
    for i in range(current_level):
        fig.add_trace(go.Scatter(y=coeffs[i+1], mode='lines', name=f'cD{current_level-i}'), row=i+2, col=1)

    fig.update_layout(
        title=f'{plot_title} ({wavelet}, Level {current_level})',
        showlegend=False,
        height=200 * (current_level + 1) # Adjust height based on levels
    )
    # Update x-axis title for the last subplot (which is visible)
    fig.update_xaxes(title_text="Sample Index", row=current_level + 1, col=1)

    return fig


def plot_cwt(data: pd.DataFrame, scales=np.arange(1, 64), wavelet='morl', plot_title: str = "Continuous Wavelet Transform (CWT) Scalogram") -> go.Figure:
    """
    Calculates and plots the Continuous Wavelet Transform (CWT) using a scalogram.
    """
    log_returns = calculate_log_returns(data)
    if log_returns.empty or len(log_returns) < max(scales): # Basic check
         return go.Figure(layout={'title': f'{plot_title} - Not enough data or scales too large'})

    coefficients, frequencies = pywt.cwt(log_returns, scales, wavelet, sampling_period=1.0) # Assuming daily data

    # Power is magnitude squared
    power = (np.abs(coefficients)) ** 2

    # Periods corresponding to frequencies
    periods = 1. / frequencies

    fig = go.Figure(data=go.Heatmap(
        z=power,
        x=data.index[1:], # Match time axis (log returns start from index 1)
        y=periods,
        colorscale='Viridis', # Or choose another colorscale
        colorbar=dict(title='Power')
    ))

    fig.update_layout(
        title=f'{plot_title} ({wavelet})',
        xaxis_title='Date',
        yaxis_title='Period (days)',
        yaxis_type='log', # Often better to view periods on a log scale
        yaxis_autorange='reversed' # Show shorter periods at the top
    )

    return fig

# --- Placeholder for other spectral methods ---

# def plot_lomb_scargle(data: pd.DataFrame, ...):
#     # Implementation using astropy.timeseries.LombScargle
#     pass

# def plot_mtm(data: pd.DataFrame, ...):
#     # Implementation requires a library like 'spectrum' or custom code
#     pass

# def plot_emd(data: pd.DataFrame, ...):
#     # Implementation requires a library like 'PyEMD'
#     pass

# def plot_wavelet_coherence(data1: pd.Series, data2: pd.Series, ...):
#     # Implementation requires a library like 'pycwt'
#     pass 