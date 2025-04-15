"""
Market regime features module
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_market_regime_features(
    builder: FeatureBuilder,
    index_data: Dict[str, pd.DataFrame],
    vix_key: str = 'vix_prices',
    spx_key: str = 'spx_prices',
    lookback_periods: List[int] = [20, 60, 120, 252]
) -> FeatureBuilder:
    """
    Add features that identify market regimes and conditions
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    index_data : Dict[str, pd.DataFrame]
        Dictionary of index data frames, with keys like 'spx_prices', 'ndx_prices', etc.
    vix_key : str, default='vix_prices'
        Key for VIX data in the index_data dictionary
    spx_key : str, default='spx_prices'
        Key for S&P 500 data in the index_data dictionary
    lookback_periods : List[int], default=[20, 60, 120, 252]
        Periods for regime classification lookbacks
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    # Check if required index data is available
    if vix_key not in index_data or spx_key not in index_data:
        return builder  # Skip if required index data is not available
    
    vix_df = index_data[vix_key].copy()
    spx_df = index_data[spx_key].copy()
    
    # Find the closing price columns
    vix_close_cols = [col for col in vix_df.columns if 'close' in col.lower()]
    spx_close_cols = [col for col in spx_df.columns if 'close' in col.lower()]
    
    if not vix_close_cols or not spx_close_cols:
        return builder  # Skip if necessary columns are not available
    
    vix_price_col = vix_close_cols[0]
    spx_price_col = spx_close_cols[0]
    
    # Ensure data is aligned with the main data's index
    vix_df = vix_df.reindex(data.index, method='ffill')
    spx_df = spx_df.reindex(data.index, method='ffill')
    
    # Calculate SPX returns
    spx_returns = spx_df[spx_price_col].pct_change()
    
    # 1. Volatility Regime Classification
    # Using VIX percentiles to classify
    for period in lookback_periods:
        if len(vix_df) >= period:
            # Calculate VIX percentile
            vix_pctl = vix_df[vix_price_col].rolling(window=period).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
            )
            
            # Classify volatility regime
            builder.features[f'vol_regime_{period}'] = pd.cut(
                vix_pctl, 
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=[1, 2, 3, 4, 5]  # 1=very low vol, 5=very high vol
            ).astype(float)
    
    # 2. Bull/Bear Market Classification
    for period in lookback_periods:
        if len(spx_df) >= period:
            # Calculate if index is above/below moving averages
            ma = spx_df[spx_price_col].rolling(window=period).mean()
            builder.features[f'spx_above_ma_{period}'] = (spx_df[spx_price_col] > ma).astype(int)
            
            # Classify trend based on recent performance
            spx_performance = spx_df[spx_price_col].pct_change(periods=period)
            builder.features[f'spx_trend_{period}'] = pd.cut(
                spx_performance,
                bins=[-float('inf'), -0.2, -0.1, 0, 0.1, 0.2, float('inf')],
                labels=[-3, -2, -1, 1, 2, 3]  # -3=strong bear, 3=strong bull
            ).astype(float)
    
    # 3. Market Breadth Features
    # We're approximating breadth with index behavior since actual breadth data isn't provided
    for period in [5, 10, 20]:
        # Calculate volatility of the SPX as a breadth proxy
        spx_vol = spx_returns.rolling(window=period).std()
        spx_vol_pctl = spx_vol.rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )
        builder.features[f'spx_vol_regime_{period}'] = pd.cut(
            spx_vol_pctl,
            bins=[0, 0.33, 0.66, 1.0],
            labels=[1, 2, 3]  # 1=low vol, 3=high vol
        ).astype(float)
    
    # 4. Combined VIX-SPX regime
    if len(vix_df) >= 20 and len(spx_df) >= 20:
        # VIX 20-day percentile
        vix_20d_pctl = vix_df[vix_price_col].rolling(window=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True
        )
        
        # SPX 20-day performance
        spx_20d_perf = spx_df[spx_price_col].pct_change(periods=20)
        
        # Combine into a single regime feature
        # 1: Bull market, low vol (ideal)
        # 2: Bull market, high vol (caution)
        # 3: Bear market, low vol (bottoming)
        # 4: Bear market, high vol (panic)
        conditions = [
            (spx_20d_perf > 0) & (vix_20d_pctl < 0.5),
            (spx_20d_perf > 0) & (vix_20d_pctl >= 0.5),
            (spx_20d_perf <= 0) & (vix_20d_pctl < 0.5),
            (spx_20d_perf <= 0) & (vix_20d_pctl >= 0.5),
        ]
        values = [1, 2, 3, 4]
        builder.features['market_regime'] = np.select(conditions, values, default=np.nan)
    
    # 5. Sector rotation features (using multiple indexes)
    # Calculate relative strength of technology vs broader market
    if 'ndx_prices' in index_data and 'spx_prices' in index_data:
        ndx_df = index_data['ndx_prices'].copy().reindex(data.index, method='ffill')
        ndx_close_cols = [col for col in ndx_df.columns if 'close' in col.lower()]
        if ndx_close_cols:
            ndx_price_col = ndx_close_cols[0]
            
            # Calculate relative strength
            for period in [5, 10, 20, 60]:
                ndx_perf = ndx_df[ndx_price_col].pct_change(periods=period)
                spx_perf = spx_df[spx_price_col].pct_change(periods=period)
                builder.features[f'tech_vs_market_{period}'] = ndx_perf - spx_perf
    
    # 6. Semiconductor index relative strength (sector leadership indicator)
    if 'sox_prices' in index_data and 'spx_prices' in index_data:
        sox_df = index_data['sox_prices'].copy().reindex(data.index, method='ffill')
        sox_close_cols = [col for col in sox_df.columns if 'close' in col.lower()]
        if sox_close_cols:
            sox_price_col = sox_close_cols[0]
            
            # Calculate relative strength
            for period in [5, 10, 20, 60]:
                sox_perf = sox_df[sox_price_col].pct_change(periods=period)
                spx_perf = spx_df[spx_price_col].pct_change(periods=period)
                builder.features[f'semi_vs_market_{period}'] = sox_perf - spx_perf
    
    # 7. Small cap vs large cap rotation (market breadth indicator)
    if 'rut_prices' in index_data and 'spx_prices' in index_data:
        rut_df = index_data['rut_prices'].copy().reindex(data.index, method='ffill')
        rut_close_cols = [col for col in rut_df.columns if 'close' in col.lower()]
        if rut_close_cols:
            rut_price_col = rut_close_cols[0]
            
            # Calculate relative strength
            for period in [5, 10, 20, 60]:
                rut_perf = rut_df[rut_price_col].pct_change(periods=period)
                spx_perf = spx_df[spx_price_col].pct_change(periods=period)
                builder.features[f'small_vs_large_{period}'] = rut_perf - spx_perf
    
    # 8. Combined health indicators
    health_indicators = []
    
    # Add SPX trend indicators
    for period in [20, 50, 200]:
        if f'spx_above_ma_{period}' in builder.features:
            health_indicators.append(f'spx_above_ma_{period}')
    
    # VIX below 20 indicator (if available)
    if len(vix_df) > 0:
        builder.features['vix_below_20'] = (vix_df[vix_price_col] < 20).astype(int)
        health_indicators.append('vix_below_20')
    
    # Combine health indicators into a single score
    if health_indicators:
        builder.features['market_health_score'] = sum(builder.features[col] for col in health_indicators)
    
    return builder