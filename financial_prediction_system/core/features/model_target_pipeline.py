"""
Model Target Pipeline Module

This module provides functionality for creating target variables for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class ModelTargetPipeline:
    """Creates and transforms target variables for machine learning models"""
    
    def __init__(self, data: Dict[str, Any]):
        """
        Initialize the target pipeline
        
        Parameters
        ----------
        data : Dict[str, Any]
            Raw data containing price information
        """
        self.df = pd.DataFrame(data)
        
    def build_classification_target(self, horizon: int = 5, threshold: float = 0.0) -> np.ndarray:
        """
        Build binary classification target based on future returns
        
        Parameters
        ----------
        horizon : int, default=5
            Number of periods to look ahead
        threshold : float, default=0.0
            Return threshold for positive class
            
        Returns
        -------
        np.ndarray
            Binary target values
        """
        if 'close' not in self.df.columns:
            return np.array([])
            
        # Calculate forward returns
        returns = self.df['close'].pct_change(horizon).shift(-horizon)
        
        # Create binary target
        target = (returns > threshold).astype(int)
        
        # Fill NaN values at the end of the series
        target = target.fillna(0)
        
        return target.values
        
    def build_regression_target(self, horizon: int = 5, 
                              target_type: str = 'returns') -> np.ndarray:
        """
        Build regression target
        
        Parameters
        ----------
        horizon : int, default=5
            Number of periods to look ahead
        target_type : str, default='returns'
            Type of target to build ('returns' or 'volatility')
            
        Returns
        -------
        np.ndarray
            Target values
        """
        if 'close' not in self.df.columns:
            return np.array([])
            
        if target_type == 'returns':
            # Forward returns
            target = self.df['close'].pct_change(horizon).shift(-horizon)
            
        elif target_type == 'volatility':
            # Forward volatility
            returns = self.df['close'].pct_change()
            target = returns.rolling(horizon).std().shift(-horizon)
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
            
        # Fill NaN values at the end of the series
        target = target.fillna(0)
        
        return target.values
        
    def build_custom_target(self, formula: str, params: Dict[str, Any] = None) -> np.ndarray:
        """
        Build custom target using provided formula
        
        Parameters
        ----------
        formula : str
            Formula to calculate target. Available variables: close, high, low, volume
        params : Dict[str, Any], optional
            Additional parameters for formula calculation
            
        Returns
        -------
        np.ndarray
            Target values
        """
        try:
            # Create local variables for formula
            locals_dict = {
                'df': self.df,
                'np': np,
                'pd': pd,
                'params': params or {}
            }
            
            # Execute formula in safe context
            exec(formula, {'__builtins__': {}}, locals_dict)
            
            # Get result from locals
            if 'result' not in locals_dict:
                raise ValueError("Formula must assign to 'result' variable")
                
            target = locals_dict['result']
            
            # Convert to numpy array
            if isinstance(target, (pd.Series, pd.DataFrame)):
                target = target.values
            elif not isinstance(target, np.ndarray):
                target = np.array(target)
                
            return target
            
        except Exception as e:
            print(f"Error calculating custom target: {str(e)}")
            return np.array([])