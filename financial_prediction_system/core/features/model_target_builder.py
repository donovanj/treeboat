import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler

class ModelTargetPipeline:
    """Pipeline for constructing and transforming model target variables"""
    
    def __init__(self, data: Dict[str, Any]):
        self.df = pd.DataFrame(data)
        self.scaler = StandardScaler()
    
    def build_classification_target(
        self,
        horizon: int = 5,
        threshold: float = 0.0,
        type: str = 'binary'
    ) -> np.ndarray:
        """
        Build classification target
        
        Args:
            horizon: Forward-looking period for returns
            threshold: Return threshold for classification
            type: 'binary' or 'multi' for 2 or 3 classes
        """
        if 'close' not in self.df.columns:
            return None
        
        # Calculate forward returns
        returns = self.df['close'].pct_change(horizon).shift(-horizon)
        
        if type == 'binary':
            # Binary classification (up/down)
            target = (returns > threshold).astype(int)
        else:
            # Multi-class (down/neutral/up)
            target = pd.cut(
                returns,
                bins=[-np.inf, -threshold, threshold, np.inf],
                labels=[0, 1, 2]
            ).astype(int)
        
        return target.values
    
    def build_regression_target(
        self,
        horizon: int = 5,
        type: str = 'returns',
        vol_adjust: bool = False
    ) -> np.ndarray:
        """
        Build regression target
        
        Args:
            horizon: Forward-looking period
            type: 'returns' or 'log_returns'
            vol_adjust: Whether to volatility-adjust the target
        """
        if 'close' not in self.df.columns:
            return None
        
        # Calculate forward returns
        if type == 'log_returns':
            returns = np.log(self.df['close'] / self.df['close'].shift(1))
        else:
            returns = self.df['close'].pct_change(horizon)
            
        # Shift to get forward returns
        target = returns.shift(-horizon)
        
        if vol_adjust:
            # Calculate rolling volatility
            vol = returns.rolling(window=21).std()
            target = target / (vol + 1e-10)  # Add small constant to avoid division by zero
        
        return target.values
    
    def build_probabilistic_target(
        self,
        horizon: int = 5,
        n_bins: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build probabilistic target (discretized returns distribution)
        
        Args:
            horizon: Forward-looking period
            n_bins: Number of bins for discretization
            
        Returns:
            Tuple of (bin_labels, bin_edges)
        """
        if 'close' not in self.df.columns:
            return None, None
        
        # Calculate forward returns
        returns = self.df['close'].pct_change(horizon).shift(-horizon)
        
        # Create bins based on return distribution
        bin_edges = np.percentile(
            returns.dropna(),
            np.linspace(0, 100, n_bins + 1)
        )
        
        # Assign returns to bins
        target = pd.cut(
            returns,
            bins=bin_edges,
            labels=False,
            include_lowest=True
        )
        
        return target.values, bin_edges
    
    def build_ranking_target(
        self,
        horizon: int = 5,
        universe_returns: Optional[pd.Series] = None
    ) -> np.ndarray:
        """
        Build ranking target (relative performance vs universe)
        
        Args:
            horizon: Forward-looking period
            universe_returns: Returns of the investment universe
        """
        if 'close' not in self.df.columns:
            return None
        
        # Calculate stock returns
        stock_returns = self.df['close'].pct_change(horizon).shift(-horizon)
        
        if universe_returns is not None:
            # Calculate relative performance percentile
            merged = pd.concat([stock_returns, universe_returns], axis=1)
            merged.columns = ['stock', 'universe']
            
            # Calculate rolling percentile rank
            target = merged.groupby(level=0).apply(
                lambda x: x['stock'].rank(pct=True)
            )
        else:
            # If no universe provided, just rank the stock's returns
            target = stock_returns.rank(pct=True)
        
        return target.values
    
    def build_regime_target(
        self,
        horizon: int = 5,
        n_regimes: int = 3,
        method: str = 'volatility'
    ) -> np.ndarray:
        """
        Build market regime target
        
        Args:
            horizon: Forward-looking period
            n_regimes: Number of regime states
            method: 'volatility' or 'trend' based regime classification
        """
        if 'close' not in self.df.columns:
            return None
        
        returns = self.df['close'].pct_change()
        
        if method == 'volatility':
            # Volatility-based regimes
            vol = returns.rolling(window=horizon).std()
            target = pd.qcut(vol, q=n_regimes, labels=False)
            
        else:
            # Trend-based regimes
            ma_fast = self.df['close'].rolling(window=horizon).mean()
            ma_slow = self.df['close'].rolling(window=horizon * 2).mean()
            
            trend = (ma_fast - ma_slow) / ma_slow
            target = pd.qcut(trend, q=n_regimes, labels=False)
        
        return target.values
    
    def transform_target(
        self,
        target: np.ndarray,
        method: str = 'standardize'
    ) -> np.ndarray:
        """
        Transform target variable
        
        Args:
            target: Target array to transform
            method: 'standardize' or 'normalize'
        """
        if target is None or len(target) == 0:
            return None
        
        # Handle NaN values
        target = np.nan_to_num(target, nan=0)
        
        if method == 'standardize':
            # Standardize to mean=0, std=1
            return self.scaler.fit_transform(target.reshape(-1, 1)).ravel()
            
        elif method == 'normalize':
            # Normalize to [0, 1]
            min_val = np.min(target)
            max_val = np.max(target)
            return (target - min_val) / (max_val - min_val + 1e-10)
        
        return target
    
    def calculate_target_stats(self, target: np.ndarray) -> Dict[str, Any]:
        """Calculate basic statistics for the target variable"""
        if target is None or len(target) == 0:
            return {}
            
        valid_mask = ~np.isnan(target)
        clean_target = target[valid_mask]
        
        if len(clean_target) == 0:
            return {}
            
        return {
            'mean': float(np.mean(clean_target)),
            'std': float(np.std(clean_target)),
            'min': float(np.min(clean_target)),
            'max': float(np.max(clean_target)),
            'null_count': int(np.sum(~valid_mask)),
            'unique_values': int(len(np.unique(clean_target))),
            'skewness': float(pd.Series(clean_target).skew()),
            'kurtosis': float(pd.Series(clean_target).kurt())
        }