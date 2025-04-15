"""
Target Builder Module

This module implements the Builder pattern for creating target variables with flexible
composition for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

class TargetBuilder:
    """Builder for creating target variables with flexible composition"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the target builder
        
        Parameters
        ----------
        data : pd.DataFrame
            The input data
        """
        self.data = data
        self.targets = {}

    def add_target_set(self, target_set_name: str, **kwargs) -> 'TargetBuilder':
        """
        Add a predefined set of targets
        
        Parameters
        ----------
        target_set_name : str
            Name of the target set to add
        **kwargs : dict
            Parameters to pass to the target generator
        
        Returns
        -------
        self : TargetBuilder
            The builder instance for method chaining
        """
        if target_set_name == 'price':
            from financial_prediction_system.core.targets.price_targets import add_price_targets
            add_price_targets(self, **kwargs)
        elif target_set_name == 'drawdown':
            from financial_prediction_system.core.targets.price_targets import add_drawdown_targets
            add_drawdown_targets(self, **kwargs)
        elif target_set_name == 'alpha':
            from financial_prediction_system.core.targets.alpha_targets import add_alpha_targets
            add_alpha_targets(self, **kwargs)
        elif target_set_name == 'volatility':
            from financial_prediction_system.core.targets.volatility_targets import add_volatility_targets
            add_volatility_targets(self, **kwargs)
        elif target_set_name == 'jump_risk':
            from financial_prediction_system.core.targets.volatility_targets import add_jump_risk_targets
            add_jump_risk_targets(self, **kwargs)
        elif target_set_name == 'tail_risk':
            from financial_prediction_system.core.targets.volatility_targets import add_tail_risk_targets
            add_tail_risk_targets(self, **kwargs)
        else:
            raise ValueError(f"Unknown target set: {target_set_name}")
        
        return self
    
    def build(self) -> pd.DataFrame:
        """
        Build the target set
        
        Returns
        -------
        pd.DataFrame
            The complete target set
        """
        # Convert targets dictionary to DataFrame
        targets_df = pd.DataFrame(self.targets)
        
        # Drop NaN values that may have been introduced by rolling windows
        targets_df = targets_df.dropna()
        
        return targets_df
    
    def get_target(self, name: str) -> Optional[pd.Series]:
        """
        Get a specific target by name
        
        Parameters
        ----------
        name : str
            The name of the target
        
        Returns
        -------
        Optional[pd.Series]
            The target series or None if not found
        """
        return self.targets.get(name)
    
    def transform_targets(self, method: str = 'zscore', **kwargs) -> 'TargetBuilder':
        """
        Transform targets using a specified method
        
        Parameters
        ----------
        method : str, default='zscore'
            Method to use for transformation: 'zscore', 'minmax', 'log', 'boxcox', etc.
        **kwargs : dict
            Additional parameters for the transformation method
        
        Returns
        -------
        self : TargetBuilder
            The builder instance for method chaining
        """
        from scipy import stats
        
        for target_name, target_series in self.targets.items():
            if method == 'zscore':
                # Z-score normalization
                self.targets[target_name] = (target_series - target_series.mean()) / target_series.std()
            elif method == 'minmax':
                # Min-max scaling
                min_val = target_series.min()
                max_val = target_series.max()
                if max_val > min_val:
                    self.targets[target_name] = (target_series - min_val) / (max_val - min_val)
            elif method == 'log':
                # Log transformation (with offset for non-positive values)
                if target_series.min() <= 0:
                    offset = abs(target_series.min()) + 1
                    self.targets[target_name] = np.log(target_series + offset)
                else:
                    self.targets[target_name] = np.log(target_series)
            elif method == 'boxcox':
                # Box-Cox transformation
                if target_series.min() <= 0:
                    offset = abs(target_series.min()) + 1
                    self.targets[target_name], _ = stats.boxcox(target_series + offset)
                else:
                    self.targets[target_name], _ = stats.boxcox(target_series)
            elif method == 'rank':
                # Rank transformation
                self.targets[target_name] = target_series.rank(pct=True)
            else:
                raise ValueError(f"Unknown transformation method: {method}")
                
        return self


class TargetDirector:
    """
    Director class that uses the TargetBuilder to construct targets
    according to predefined strategies
    """
    
    @staticmethod
    def create_price_return_targets(data: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Create standard price return targets
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with price columns
        periods : List[int], default=[1, 5, 10, 20]
            Forward periods to calculate returns for
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing target variables
        """
        builder = TargetBuilder(data)
        builder.add_target_set('price', periods=periods)
        return builder.build()
    
    @staticmethod
    def create_trading_strategy_targets(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a comprehensive set of targets for trading strategy development
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with price, volume, and possibly benchmark columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing target variables
        """
        builder = TargetBuilder(data)
        
        # Add directional targets for multiple timeframes
        builder.add_target_set('price', 
                               periods=[1, 5, 10, 20],
                               return_type='direction')
        
        # Add continuous return targets
        builder.add_target_set('price',
                               periods=[1, 5, 10, 20],
                               return_type='log')
        
        # Add drawdown prediction targets
        builder.add_target_set('drawdown', 
                               periods=[5, 10, 20],
                               classification=True)
        
        # Add volatility targets if price data available
        if 'high' in data.columns and 'low' in data.columns:
            builder.add_target_set('volatility',
                                  periods=[5, 10, 20],
                                  volatility_type='garman_klass')
        
        # Add alpha targets if benchmark data available
        if 'benchmark' in data.columns:
            builder.add_target_set('alpha',
                                  periods=[5, 10, 20],
                                  alpha_type='information_ratio')
        
        return builder.build()
    
    @staticmethod
    def create_volatility_prediction_targets(data: pd.DataFrame) -> pd.DataFrame:
        """
        Create targets specifically for volatility prediction
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with price columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing volatility target variables
        """
        builder = TargetBuilder(data)
        
        # Add realized volatility targets
        builder.add_target_set('volatility',
                              periods=[1, 5, 10, 20],
                              volatility_type='realized')
        
        # Add jump risk targets
        builder.add_target_set('jump_risk',
                              periods=[5, 10, 20],
                              classification=True)
        
        # Add tail risk targets
        builder.add_target_set('tail_risk',
                              periods=[10, 20],
                              quantiles=[0.01, 0.05])
        
        return builder.build() 