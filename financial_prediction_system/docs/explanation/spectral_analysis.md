# Spectral Analysis Fundamentals

## Introduction

Spectral analysis transforms time-domain financial data into the frequency domain, revealing cyclical patterns and periodicities that might not be visible in standard price charts. This document explains the theory behind the spectral analysis visualizations in our system.

## The Frequency Domain Perspective

Financial time series are traditionally analyzed in the time domain, where we observe how prices change over time. Spectral analysis offers a complementary view by decomposing price movements into their constituent frequencies.

This approach allows us to:
1. Identify hidden cycles in market data
2. Detect subtle patterns that repeat over different time scales
3. Filter out noise to focus on significant market movements
4. Compare the spectral characteristics of different assets

## Fast Fourier Transform (FFT)

The FFT Power Spectrum visualization converts price or return data into frequency components, showing which cycles contribute most to the observed price movements.

### Mathematical Foundation

The Fourier transform decomposes a time series into a sum of sine and cosine functions of different frequencies:

```
X(f) = ∫ x(t) * e^(-j2πft) dt
```

Where:
- X(f) is the frequency domain representation
- x(t) is the time domain signal (e.g., returns)
- e^(-j2πft) is the complex exponential function

### Interpreting the FFT Power Spectrum

- **X-axis**: Represents frequency (cycles per day)
- **Y-axis**: Represents power (squared amplitude of the frequency component)
- **Peaks**: Indicate strong cyclical components

For example, a peak at 0.05 cycles/day suggests a 20-day cycle in the data. Peaks at lower frequencies correspond to longer-term cycles.

## Discrete Wavelet Transform (DWT)

Unlike the FFT which uses a fixed time window, the DWT uses wavelets to capture both frequency and time information.

### Advantages Over FFT

- Better handles non-stationary data (changing statistical properties)
- Provides time localization of frequency changes
- Can detect when cycles start and end

### Interpretation

In our DWT visualization:
- Different scales (y-axis) correspond to different cycle lengths
- The color intensity shows the strength of the signal at that scale and time
- Changes in color patterns indicate regime shifts in the market's cyclical behavior

## Continuous Wavelet Transform (CWT) Scalogram

The CWT extends the wavelet analysis to provide a continuous view of how cycles evolve over time.

### Key Features

- **Heatmap representation**: Colors indicate power at different scales and times
- **Cone of influence**: Marks regions with potential edge effects
- **Phase arrows**: Show relative phase between different components

### Market Applications

- Identifying when market regimes change
- Detecting coherence between different assets or indicators
- Finding leading and lagging relationships between variables

## Practical Applications in Trading

### Cycle Identification

Identified cycles can be used to anticipate potential turning points in the market. For example, if the FFT shows a strong 10-day cycle, traders might look for reversals at these intervals.

### Noise Filtering

By identifying the frequency components that represent noise, traders can filter them out to focus on meaningful price movements.

### Regime Detection

Changes in the spectral profile often precede changes in market behavior. By monitoring the evolution of the frequency spectrum, traders can adapt their strategies to changing market conditions.

## Limitations

- Assumes some level of cyclical behavior exists
- May not perform well during highly irregular market periods
- Requires careful parameter selection (window size, wavelet type, etc.)
- Past cycles do not guarantee future repetition

## Related Concepts

- **Hilbert Transform**: Used for instantaneous frequency analysis
- **Empirical Mode Decomposition**: Adaptive method for non-linear, non-stationary data
- **Singular Spectrum Analysis**: Combines elements of classical time series analysis, multivariate statistics, and signal processing