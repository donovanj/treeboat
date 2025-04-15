"""
Sector relative strength features module
"""
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_sector_relative_features(
    builder: FeatureBuilder,
    index_data: Dict[str, pd.DataFrame],
    security_sector: str = None,
    windows: List[int] = [5, 10, 20, 60]
) -> FeatureBuilder:
    """
    Add features capturing the relationship between a security and relevant sector indexes
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    index_data : Dict[str, pd.DataFrame]
        Dictionary of index data frames, with keys like 'spx_prices', 'ndx_prices', etc.
    security_sector : str, optional
        Known sector of the security, if available
    windows : List[int], default=[5, 10, 20, 60]
        Window sizes for calculating rolling performance metrics
        
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    # Find the closing price column in the main data
    close_cols = [col for col in data.columns if 'close' in col.lower()]
    if not close_cols:
        return builder  # Skip if no close data available
    price_col = close_cols[0]
    
    # Calculate security returns
    security_returns = data[price_col].pct_change()
    
    # Map of sector indexes to their market sectors
    sector_index_mapping = {
        'ndx': 'technology',
        'sox': 'semiconductors',
        'osx': 'oil_services',
        'rut': 'small_cap',
        'dji': 'large_cap_industrial',
        'spx': 'broad_market'
    }
    
    # 1. Calculate relative strength against each sector index
    for index_name, index_df in index_data.items():
        # Skip if the index data doesn't have enough data
        if len(index_df) < max(windows):
            continue
        
        # Extract sector prefix
        index_prefix = index_name.split('_')[0].lower()
        sector = sector_index_mapping.get(index_prefix, index_prefix)
        
        # Find index close price
        index_close_cols = [col for col in index_df.columns if 'close' in col.lower()]
        if not index_close_cols:
            continue
        index_price_col = index_close_cols[0]
        
        # Ensure index data is aligned with security data
        index_df = index_df.reindex(data.index, method='ffill')
        
        # Calculate index returns
        index_returns = index_df[index_price_col].pct_change()
        
        # Relative strength (performance compared to sector)
        for window in windows:
            # Rolling relative performance
            sec_performance = (1 + security_returns).rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )
            idx_performance = (1 + index_returns).rolling(window=window).apply(
                lambda x: (1 + x).prod() - 1, raw=True
            )
            
            # Store relative strength vs this sector
            builder.features[f'rel_to_{sector}_{window}d'] = sec_performance - idx_performance
            
            # Calculate percentile of this relative strength over a longer window
            long_window = min(252, len(builder.features))
            if len(builder.features) >= long_window:
                rel_strength_series = builder.features[f'rel_to_{sector}_{window}d']
                if not rel_strength_series.isna().all():
                    builder.features[f'rel_{sector}_percentile_{window}d'] = rel_strength_series.rolling(
                        window=long_window).apply(
                        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if not pd.Series(x).isna().all() else np.nan,
                        raw=True
                    )
                    
            # Calculate whether security is outperforming the sector
            builder.features[f'outperf_{sector}_{window}d'] = (
                builder.features[f'rel_to_{sector}_{window}d'] > 0
            ).astype(int)
    
    # 2. If security sector is known, create focused relative strength features
    if security_sector and security_sector in sector_index_mapping.values():
        # Find indexes that match this sector
        matching_indexes = [k for k, v in sector_index_mapping.items() if v == security_sector]
        
        # Get the relevant sector index data
        for index_prefix in matching_indexes:
            # Find the index data
            sector_index_key = next((k for k in index_data.keys() if k.startswith(index_prefix)), None)
            
            if sector_index_key and sector_index_key in index_data:
                sector_df = index_data[sector_index_key].copy()
                sector_close_cols = [col for col in sector_df.columns if 'close' in col.lower()]
                
                if sector_close_cols:
                    sector_price_col = sector_close_cols[0]
                    sector_df = sector_df.reindex(data.index, method='ffill')
                    sector_returns = sector_df[sector_price_col].pct_change()
                    
                    # Beta to sector (more important than beta to market for sector stocks)
                    for window in [30, 60, 120]:
                        if len(security_returns) >= window and len(sector_returns) >= window:
                            # Calculate rolling beta using regression
                            rolling_betas = []
                            for i in range(window - 1, len(security_returns)):
                                if i >= window - 1:
                                    sec_window = security_returns.iloc[i-window+1:i+1].values
                                    idx_window = sector_returns.iloc[i-window+1:i+1].values
                                    
                                    # Filter out NaN values
                                    valid_indices = ~(np.isnan(sec_window) | np.isnan(idx_window))
                                    if sum(valid_indices) > window // 2:  # Require at least half the window
                                        X = idx_window[valid_indices].reshape(-1, 1)
                                        X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
                                        y = sec_window[valid_indices]
                                        
                                        try:
                                            beta, _ = np.linalg.lstsq(X, y, rcond=None)[0]
                                            rolling_betas.append(beta)
                                        except:
                                            rolling_betas.append(np.nan)
                                    else:
                                        rolling_betas.append(np.nan)
                                else:
                                    rolling_betas.append(np.nan)
                            
                            # Add to features
                            if len(rolling_betas) == len(security_returns):
                                builder.features[f'sector_beta_{window}d'] = pd.Series(
                                    rolling_betas, index=security_returns.index
                                )
    
    # 3. Cross-sector relative strength (useful for sector rotation strategies)
    # Calculate relative strength between key sectors
    sector_pairs = [
        ('technology', 'broad_market'),  # Tech vs SPX
        ('semiconductors', 'technology'),  # Semis vs Tech
        ('small_cap', 'large_cap_industrial'),  # Small vs Large
    ]
    
    for sector1, sector2 in sector_pairs:
        # Find indexes that match these sectors
        sector1_indexes = [k for k, v in sector_index_mapping.items() if v == sector1]
        sector2_indexes = [k for k, v in sector_index_mapping.items() if v == sector2]
        
        # Get first available index for each sector
        sector1_key = next((k for idx in sector1_indexes for k in index_data.keys() 
                           if k.startswith(idx)), None)
        sector2_key = next((k for idx in sector2_indexes for k in index_data.keys() 
                           if k.startswith(idx)), None)
        
        if sector1_key and sector2_key and sector1_key in index_data and sector2_key in index_data:
            s1_df = index_data[sector1_key].copy()
            s2_df = index_data[sector2_key].copy()
            
            s1_close_cols = [col for col in s1_df.columns if 'close' in col.lower()]
            s2_close_cols = [col for col in s2_df.columns if 'close' in col.lower()]
            
            if s1_close_cols and s2_close_cols:
                s1_price_col = s1_close_cols[0]
                s2_price_col = s2_close_cols[0]
                
                s1_df = s1_df.reindex(data.index, method='ffill')
                s2_df = s2_df.reindex(data.index, method='ffill')
                
                s1_returns = s1_df[s1_price_col].pct_change()
                s2_returns = s2_df[s2_price_col].pct_change()
                
                # Calculate relative performance
                for window in windows:
                    s1_performance = (1 + s1_returns).rolling(window=window).apply(
                        lambda x: (1 + x).prod() - 1, raw=True
                    )
                    s2_performance = (1 + s2_returns).rolling(window=window).apply(
                        lambda x: (1 + x).prod() - 1, raw=True
                    )
                    
                    # Store sector relative strength
                    feature_name = f'{sector1}_vs_{sector2}_{window}d'
                    builder.features[feature_name] = s1_performance - s2_performance
                    
                    # Flag if security sector is outperforming
                    if security_sector:
                        if security_sector == sector1:
                            builder.features[f'in_leading_sector_{window}d'] = (
                                builder.features[feature_name] > 0
                            ).astype(int)
                        elif security_sector == sector2:
                            builder.features[f'in_leading_sector_{window}d'] = (
                                builder.features[feature_name] < 0
                            ).astype(int)
    
    return builder