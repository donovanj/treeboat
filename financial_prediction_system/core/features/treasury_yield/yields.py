"""
Treasury yield features module
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_treasury_yield_features(
    builder: FeatureBuilder,
    yields_data: pd.DataFrame,
    rate_change_windows: List[int] = [1, 5, 10, 20],
    percentile_windows: List[int] = [60, 120, 252]
) -> FeatureBuilder:
    """
    Add features based on Treasury yield data
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    yields_data : pd.DataFrame
        DataFrame containing Treasury yield data with date as index
        and columns for different maturities (mo1, mo2, mo3, mo6, yr1, yr2, yr5, yr10, yr30)
    rate_change_windows : List[int], default=[1, 5, 10, 20]
        Window sizes for rate change calculations
    percentile_windows : List[int], default=[60, 120, 252]
        Window sizes for percentile calculations
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    # Clean up yields data
    # First, create a copy to avoid modifying the original
    yields = yields_data.copy()
    
    # Handle any missing values with forward fill, then backward fill
    yields = yields.ffill().bfill()
    
    # Ensure yields_data is aligned with the builder's data index
    yields = yields.reindex(data.index, method='ffill')
    
    # List of all yield columns
    yield_cols = ['mo1', 'mo2', 'mo3', 'mo6', 'yr1', 'yr2', 'yr5', 'yr10', 'yr30']
    
    # Keep only columns that exist in the dataframe
    available_yield_cols = [col for col in yield_cols if col in yields.columns]
    
    if not available_yield_cols:
        return builder  # No valid yield columns
    
    # 1. Raw yield values
    for col in available_yield_cols:
        builder.features[f'treasury_{col}'] = yields[col]
    
    # 2. Yield curve spreads
    # Common spreads used in market analysis
    spread_pairs = [
        ('yr10', 'yr2'),    # 10yr-2yr (classic recession indicator)
        ('yr10', 'mo3'),    # 10yr-3mo (NY Fed recession indicator)
        ('yr30', 'yr10'),   # 30yr-10yr (long end steepness)
        ('yr5', 'yr2'),     # 5yr-2yr (intermediate curve)
        ('yr2', 'mo3'),     # 2yr-3mo (short end steepness)
        ('yr5', 'mo3'),     # 5yr-3mo
    ]
    
    for long_term, short_term in spread_pairs:
        if long_term in available_yield_cols and short_term in available_yield_cols:
            builder.features[f'spread_{long_term}_{short_term}'] = yields[long_term] - yields[short_term]
    
    # 3. Rate change features
    for col in available_yield_cols:
        for window in rate_change_windows:
            # Absolute change in basis points
            builder.features[f'{col}_change_{window}d'] = yields[col] - yields[col].shift(window)
            
            # Percent change
            builder.features[f'{col}_pct_change_{window}d'] = yields[col].pct_change(periods=window)
    
    # 4. Yield curve inversion flags
    # Check for common yield curve inversions
    inversion_pairs = [
        ('yr10', 'yr2'),    # 10yr-2yr inversion
        ('yr10', 'mo3'),    # 10yr-3mo inversion
        ('yr5', 'yr2'),     # 5yr-2yr inversion
    ]
    
    for long_term, short_term in inversion_pairs:
        if long_term in available_yield_cols and short_term in available_yield_cols:
            # Binary flag for inversion
            builder.features[f'inversion_{long_term}_{short_term}'] = (
                yields[long_term] < yields[short_term]
            ).astype(int)
            
            # Days since inversion began
            inversion_flag = builder.features[f'inversion_{long_term}_{short_term}']
            
            # Find sequences of consecutive inversions
            # Reset counter when not inverted, increment when inverted
            inversion_counter = np.zeros(len(inversion_flag))
            counter = 0
            
            for i in range(len(inversion_flag)):
                if inversion_flag.iloc[i] == 1:
                    counter += 1
                    inversion_counter[i] = counter
                else:
                    counter = 0
                    inversion_counter[i] = 0
            
            builder.features[f'days_inverted_{long_term}_{short_term}'] = pd.Series(
                inversion_counter, index=inversion_flag.index
            )
    
    # 5. Yield curve steepness
    if 'yr30' in available_yield_cols and 'mo3' in available_yield_cols:
        builder.features['yield_curve_steepness'] = yields['yr30'] - yields['mo3']
    elif 'yr10' in available_yield_cols and 'mo3' in available_yield_cols:
        builder.features['yield_curve_steepness'] = yields['yr10'] - yields['mo3']
    
    # 6. Historical percentile ranks for rates and spreads
    for col in available_yield_cols:
        for window in percentile_windows:
            if len(yields) >= window:
                builder.features[f'{col}_percentile_{window}d'] = yields[col].rolling(window=window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
                )
    
    # 7. Rate momentum features
    # Captures acceleration in rate changes across multiple timeframes
    short_windows = [1, 5]
    long_windows = [10, 20]
    
    for col in available_yield_cols:
        for short_win in short_windows:
            for long_win in long_windows:
                if short_win < long_win:
                    # Short-term vs long-term rate change momentum
                    short_change = yields[col].pct_change(periods=short_win)
                    long_change = yields[col].pct_change(periods=long_win)
                    
                    # Annualized changes for comparison
                    days_per_year = 252
                    short_annualized = ((1 + short_change) ** (days_per_year / short_win)) - 1
                    long_annualized = ((1 + long_change) ** (days_per_year / long_win)) - 1
                    
                    # Momentum ratio (short-term change relative to long-term change)
                    builder.features[f'{col}_momentum_{short_win}_{long_win}'] = short_annualized - long_annualized
    
    # 8. Real yield approximation
    # Using 10yr yield minus inflation expectations
    if 'yr10' in available_yield_cols:
        # Simplified approach assuming 2% inflation target
        # In a real system, you'd use inflation data or TIPS spreads
        builder.features['real_yield_approx'] = yields['yr10'] - 2.0
    
    # 9. Yield curve polynomial fit
    # Calculate the curvature of the yield curve using polynomial regression
    # Requires at least 3 points for meaningful calculation
    if len(available_yield_cols) >= 3:
        # Convert maturity labels to months for numerical analysis
        maturity_map = {
            'mo1': 1/12,
            'mo2': 2/12,
            'mo3': 3/12,
            'mo6': 6/12,
            'yr1': 1,
            'yr2': 2,
            'yr5': 5,
            'yr10': 10,
            'yr30': 30
        }
        
        # Calculate yield curve curvature for each date
        curve_coeffs = []
        for idx in yields.index:
            x_values = [maturity_map[col] for col in available_yield_cols]
            y_values = [yields.loc[idx, col] for col in available_yield_cols]
            
            # Filter out any NaN values
            valid_indices = ~np.isnan(y_values)
            x_valid = np.array(x_values)[valid_indices]
            y_valid = np.array(y_values)[valid_indices]
            
            if len(x_valid) >= 3:  # Need at least 3 points for quadratic fit
                try:
                    # Fit a 2nd degree polynomial (quadratic)
                    coeffs = np.polyfit(x_valid, y_valid, 2)
                    curve_coeffs.append(coeffs[0])  # Coefficient of x² term represents curvature
                except:
                    curve_coeffs.append(np.nan)
            else:
                curve_coeffs.append(np.nan)
        
        builder.features['yield_curve_curvature'] = pd.Series(curve_coeffs, index=yields.index)
    
    return builder