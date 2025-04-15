"""
Time seasonality features module
"""

import pandas as pd
from financial_prediction_system.core.features.feature_builder import FeatureBuilder

def add_date_features(builder: FeatureBuilder) -> FeatureBuilder:
    """
    Add date-based features
    
    Parameters
    ----------
    builder : FeatureBuilder
        The feature builder instance
    
    Returns
    -------
    FeatureBuilder
        The builder instance for method chaining
    """
    data = builder.data
    
    if not isinstance(data.index, pd.DatetimeIndex):
        return builder  # Skip if index is not datetime
    
    # Day of week, month, quarter features
    builder.features['day_of_week'] = data.index.dayofweek
    builder.features['day_of_month'] = data.index.day
    builder.features['month'] = data.index.month
    builder.features['quarter'] = data.index.quarter
    
    # Is month end/start, quarter end/start
    builder.features['is_month_end'] = data.index.is_month_end.astype(int)
    builder.features['is_month_start'] = data.index.is_month_start.astype(int)
    builder.features['is_quarter_end'] = data.index.is_quarter_end.astype(int)
    builder.features['is_quarter_start'] = data.index.is_quarter_start.astype(int)
    
    return builder