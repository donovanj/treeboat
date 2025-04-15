"""
Model Target Pipeline Module

This module integrates the TargetBuilder with the models system by creating
a model-compatible target pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from financial_prediction_system.core.targets.target_builder import TargetBuilder, TargetDirector

class ModelTargetPipeline:
    """
    Pipeline that connects target building to the model system
    
    This class prepares targets for model training and prediction
    while ensuring compatibility with the model interfaces.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model target pipeline
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary for the pipeline
        """
        self.config = config or {}
        self.target_builder = None
        self.targets = None
        self.target_columns = None
        
    def prepare_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare targets from input data
        
        Parameters
        ----------
        data : pd.DataFrame
            The input data containing raw price/market data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing prepared target variables
        """
        # Create target builder
        self.target_builder = TargetBuilder(data)
        
        # Get target type from configuration
        target_type = self.config.get('target_type', 'price')
        
        # Add targets based on configuration
        if target_type == 'price':
            self.target_builder.add_target_set('price', 
                                              periods=self.config.get('periods', [1, 5, 10, 20]),
                                              return_type=self.config.get('return_type', 'log'),
                                              lookback_normalization=self.config.get('lookback_normalization', False),
                                              threshold_based=self.config.get('threshold_based', False),
                                              thresholds=self.config.get('thresholds'))
        
        elif target_type == 'volatility':
            self.target_builder.add_target_set('volatility',
                                              periods=self.config.get('periods', [5, 10, 20]),
                                              volatility_type=self.config.get('volatility_type', 'realized'),
                                              classification=self.config.get('classification', False),
                                              annualize=self.config.get('annualize', True))
            
        elif target_type == 'alpha':
            self.target_builder.add_target_set('alpha',
                                             periods=self.config.get('periods', [1, 5, 10, 20]),
                                             alpha_type=self.config.get('alpha_type', 'standard'),
                                             risk_adjusted=self.config.get('risk_adjusted', False),
                                             annualize=self.config.get('annualize', False),
                                             classification=self.config.get('classification', False))
            
        elif target_type == 'drawdown':
            self.target_builder.add_target_set('drawdown',
                                             periods=self.config.get('periods', [5, 10, 20]),
                                             classification=self.config.get('classification', False))
            
        elif target_type == 'jump_risk':
            self.target_builder.add_target_set('jump_risk',
                                            periods=self.config.get('periods', [5, 10, 20]),
                                            jump_threshold=self.config.get('jump_threshold', 0.02),
                                            classification=self.config.get('classification', True))
            
        elif target_type == 'tail_risk':
            self.target_builder.add_target_set('tail_risk',
                                             periods=self.config.get('periods', [10, 20, 60]),
                                             quantiles=self.config.get('quantiles', [0.01, 0.05]))
            
        elif target_type == 'trading_strategy':
            # Use the director for complex target composition
            return TargetDirector.create_trading_strategy_targets(data)
            
        elif target_type == 'volatility_prediction':
            # Use the director for complex target composition
            return TargetDirector.create_volatility_prediction_targets(data)
            
        elif target_type == 'custom':
            # Allow for custom target composition
            target_sets = self.config.get('target_sets', [])
            for target_set in target_sets:
                self.target_builder.add_target_set(
                    target_set.get('name'),
                    **target_set.get('params', {})
                )
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Apply transformations if configured
        transform_method = self.config.get('transform_method')
        if transform_method:
            transform_params = self.config.get('transform_params', {})
            self.target_builder.transform_targets(
                method=transform_method,
                **transform_params
            )
        
        # Build targets
        self.targets = self.target_builder.build()
        
        # Save target columns for future reference
        self.target_columns = self.targets.columns.tolist()
        
        return self.targets
    
    def get_specific_target(self, target_name: str = None) -> pd.Series:
        """
        Get a specific target from the prepared targets
        
        Parameters
        ----------
        target_name : str, optional
            Name of the target to retrieve. If None, uses the primary target from config.
            
        Returns
        -------
        pd.Series
            The selected target variable
        """
        if self.targets is None:
            raise ValueError("Targets not prepared. Call prepare_targets first.")
        
        if target_name is None:
            # Use primary target from configuration
            target_name = self.config.get('primary_target')
            
            if target_name is None:
                # If no specific target is configured, use the first available
                if len(self.target_columns) > 0:
                    target_name = self.target_columns[0]
                else:
                    raise ValueError("No targets available")
        
        if target_name not in self.targets.columns:
            raise ValueError(f"Target '{target_name}' not found in prepared targets")
            
        return self.targets[target_name]
    
    def prepare_sequence_targets(self, targets: pd.DataFrame, sequence_length: int = 20) -> np.ndarray:
        """
        Prepare sequence targets for time series models like LSTM or Transformer
        
        Parameters
        ----------
        targets : pd.DataFrame
            Target data
        sequence_length : int, default=20
            Length of sequences to create
            
        Returns
        -------
        np.ndarray
            Target values aligned with sequences
        """
        if targets is None or len(targets) < sequence_length:
            raise ValueError(f"Not enough data to create sequences of length {sequence_length}")
        
        # For each sequence, we use the target value at the end of the sequence
        # This shifts the targets appropriately for sequence models
        target_values = targets.iloc[sequence_length-1:].values
        
        return target_values
    
    def get_classification_metrics(self, target_name: str = None) -> Dict[str, Any]:
        """
        Get relevant metrics for classification targets
        
        Parameters
        ----------
        target_name : str, optional
            Name of the target to analyze. If None, uses the primary target.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with target metrics (class balance, etc.)
        """
        if self.targets is None:
            raise ValueError("Targets not prepared. Call prepare_targets first.")
            
        target = self.get_specific_target(target_name)
        
        # Check if this is a classification target
        # (either by checking dtype or by analyzing values)
        is_classification = (
            pd.api.types.is_categorical_dtype(target) or 
            pd.api.types.is_integer_dtype(target) and target.nunique() < 10
        )
        
        if not is_classification:
            return {"is_classification": False}
        
        # Calculate class distribution
        class_counts = target.value_counts().to_dict()
        class_proportions = target.value_counts(normalize=True).to_dict()
        
        # Calculate entropy of the distribution
        from scipy.stats import entropy
        dist_entropy = entropy(list(class_proportions.values()), base=2)
        
        return {
            "is_classification": True,
            "num_classes": len(class_counts),
            "class_counts": class_counts,
            "class_proportions": class_proportions,
            "entropy": dist_entropy
        }
    
    def get_regression_metrics(self, target_name: str = None) -> Dict[str, Any]:
        """
        Get relevant metrics for regression targets
        
        Parameters
        ----------
        target_name : str, optional
            Name of the target to analyze. If None, uses the primary target.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with target metrics (range, distribution, etc.)
        """
        if self.targets is None:
            raise ValueError("Targets not prepared. Call prepare_targets first.")
            
        target = self.get_specific_target(target_name)
        
        # Check if this is a continuous/regression target
        is_numeric = pd.api.types.is_numeric_dtype(target)
        
        if not is_numeric:
            return {"is_regression": False}
        
        # Calculate basic statistics
        stats = {
            "is_regression": True,
            "min": target.min(),
            "max": target.max(),
            "mean": target.mean(),
            "median": target.median(),
            "std": target.std(),
            "skew": target.skew(),
            "kurtosis": target.kurtosis(),
        }
        
        # Calculate percentiles
        for p in [1, 5, 25, 75, 95, 99]:
            stats[f"percentile_{p}"] = target.quantile(p/100)
        
        return stats 