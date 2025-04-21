import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import redis
import json
import logging
from ...core.features.feature_builder import FeatureBuilder
from ...core.features.feature_composition import FeatureComposition
from ..repositories.feature_repository import SQLFeatureRepository

logger = logging.getLogger(__name__)

class FeatureCalculationService:
    """Service for calculating and caching features"""
    
    def __init__(
        self,
        feature_repository: SQLFeatureRepository,
        redis_client: redis.Redis,
        cache_expiry: int = 3600  # 1 hour default
    ):
        self.repository = feature_repository
        self.redis_client = redis_client
        self.cache_expiry = cache_expiry
    
    def _get_cache_key(self, symbol: str, feature_id: int, start_date: str, end_date: str) -> str:
        """Generate cache key for feature data"""
        return f"feature:{symbol}:{feature_id}:{start_date}:{end_date}"
    
    def _cache_feature_data(self, key: str, data: np.ndarray) -> None:
        """Cache feature data in Redis"""
        try:
            # Convert numpy array to list for JSON serialization
            serialized_data = json.dumps({
                'data': data.tolist(),
                'cached_at': datetime.utcnow().isoformat()
            })
            self.redis_client.setex(key, self.cache_expiry, serialized_data)
        except Exception as e:
            logger.error(f"Error caching feature data: {str(e)}")
    
    def _get_cached_feature_data(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached feature data from Redis"""
        try:
            cached = self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                return np.array(data['data'])
            return None
        except Exception as e:
            logger.error(f"Error retrieving cached feature data: {str(e)}")
            return None
    
    def calculate_features(
        self,
        composition: FeatureComposition,
        data: Dict[str, Any],
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calculate features for a composition
        
        Args:
            composition: Feature composition to calculate
            data: Market data dictionary
            use_cache: Whether to use cached results
            
        Returns:
            Tuple of (feature_matrix, metrics)
        """
        if use_cache:
            cache_key = self._get_cache_key(
                composition.symbol,
                composition.id,
                min(data['date']),
                max(data['date'])
            )
            cached_data = self._get_cached_feature_data(cache_key)
            if cached_data is not None:
                return cached_data, {}
        
        # Calculate features
        builder = FeatureBuilder(data)
        
        for feature_config in composition.features:
            if feature_config.type == "technical":
                builder.add_technical_features(**feature_config.parameters)
            elif feature_config.type == "volatility":
                builder.add_volatility_features(**feature_config.parameters)
            elif feature_config.type == "volume":
                builder.add_volume_features(**feature_config.parameters)
            elif feature_config.type == "market_regime":
                builder.add_market_regime_features(**feature_config.parameters)
            elif feature_config.type == "sector":
                builder.add_sector_features(data.get('sector_data', {}), **feature_config.parameters)
            elif feature_config.type == "yield_curve":
                builder.add_yield_curve_features(data.get('yield_data', {}), **feature_config.parameters)
        
        feature_matrix = builder.build()
        metrics = builder.calculate_metrics()
        
        # Cache results
        if use_cache:
            self._cache_feature_data(cache_key, feature_matrix)
        
        return feature_matrix, metrics
    
    def calculate_feature_importance(
        self,
        composition: FeatureComposition,
        data: Dict[str, Any],
        target: np.ndarray
    ) -> Dict[str, float]:
        """Calculate feature importance metrics"""
        feature_matrix, _ = self.calculate_features(composition, data)
        
        if feature_matrix.size == 0 or target.size == 0:
            return {}
        
        feature_names = [f.name or f"feature_{i}" for i, f in enumerate(composition.features)]
        importance_metrics = {}
        
        # Calculate correlations
        for i, name in enumerate(feature_names):
            if i < feature_matrix.shape[1]:
                correlation = np.corrcoef(feature_matrix[:, i], target)[0, 1]
                importance_metrics[f"{name}_correlation"] = correlation
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(feature_matrix, target)
            for name, score in zip(feature_names, mi_scores):
                importance_metrics[f"{name}_mutual_info"] = score
                
        except Exception as e:
            logger.error(f"Error calculating mutual information: {str(e)}")
        
        return importance_metrics
    
    def update_feature_metrics(
        self,
        composition_id: int,
        metrics: Dict[str, float]
    ) -> None:
        """Update stored feature metrics"""
        try:
            self.repository.save_metrics(composition_id, metrics)
        except Exception as e:
            logger.error(f"Error updating feature metrics: {str(e)}")
    
    def get_feature_history(
        self,
        composition: FeatureComposition,
        start_date: datetime,
        end_date: datetime,
        data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Get historical feature values with dates"""
        feature_matrix, _ = self.calculate_features(composition, data)
        dates = pd.to_datetime(data['date'])
        
        feature_names = [f.name or f"feature_{i}" for i, f in enumerate(composition.features)]
        
        df = pd.DataFrame(
            feature_matrix,
            index=dates,
            columns=feature_names
        )
        
        return df.loc[start_date:end_date]
    
    def compare_features(
        self,
        composition_ids: List[int],
        data: Dict[str, Any],
        target: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compare multiple features or feature compositions"""
        results = {}
        
        for comp_id in composition_ids:
            composition = self.repository.get_feature(comp_id)
            if not composition:
                continue
                
            feature_matrix, metrics = self.calculate_features(composition, data)
            
            results[composition.name] = {
                'metrics': metrics,
                'n_features': len(composition.features)
            }
            
            if target is not None:
                importance = self.calculate_feature_importance(composition, data, target)
                results[composition.name]['importance'] = importance
        
        return results