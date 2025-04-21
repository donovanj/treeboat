import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.metrics import mutual_info_score, r2_score
import logging
from .feature_builder import FeatureBuilder
from .feature_composition import FeatureComposition
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)

class FeatureBacktester:
    """Evaluates feature performance through historical backtesting"""
    
    def __init__(
        self,
        data: Dict[str, Any],
        target_type: str = 'returns',
        cv_folds: int = 5,
        horizon: int = 5
    ):
        self.data = data
        self.target_type = target_type
        self.cv_folds = cv_folds
        self.horizon = horizon
        
        # Initialize metrics storage
        self.feature_metrics = {}
        self.backtest_results = {}
    
    def prepare_target(self) -> np.ndarray:
        """Prepare target variable based on specified type"""
        prices = self.data['close']
        
        if self.target_type == 'returns':
            # Calculate forward returns
            target = pd.Series(prices).pct_change(self.horizon).shift(-self.horizon)
            
        elif self.target_type == 'log_returns':
            # Calculate log returns
            target = np.log(prices[self.horizon:] / prices[:-self.horizon])
            target = np.pad(target, (0, self.horizon), constant_values=np.nan)
            
        elif self.target_type == 'binary':
            # Binary classification (up/down)
            returns = pd.Series(prices).pct_change(self.horizon).shift(-self.horizon)
            target = (returns > 0).astype(int)
            
        else:
            raise ValueError(f"Unsupported target type: {self.target_type}")
            
        return target.values
    
    def calculate_feature_metrics(
        self,
        features: np.ndarray,
        target: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate various metrics for feature evaluation"""
        metrics = {}
        
        # Remove NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features_clean = features[valid_mask]
        target_clean = target[valid_mask]
        
        if len(features_clean) < 2:
            return {}
            
        for i in range(features_clean.shape[1]):
            feature = features_clean[:, i]
            
            # Basic statistics
            metrics[f'feature_{i}_mean'] = np.mean(feature)
            metrics[f'feature_{i}_std'] = np.std(feature)
            metrics[f'feature_{i}_skew'] = stats.skew(feature)
            metrics[f'feature_{i}_kurtosis'] = stats.kurtosis(feature)
            
            # Correlation metrics
            metrics[f'feature_{i}_correlation'] = np.corrcoef(feature, target_clean)[0, 1]
            metrics[f'feature_{i}_rank_corr'] = stats.spearmanr(feature, target_clean)[0]
            
            # Information metrics
            # Discretize continuous variables for mutual information
            f_discrete = pd.qcut(feature, q=10, labels=False, duplicates='drop')
            t_discrete = pd.qcut(target_clean, q=10, labels=False, duplicates='drop')
            metrics[f'feature_{i}_mutual_info'] = mutual_info_score(f_discrete, t_discrete)
            
            # Predictability metrics
            slope, intercept, r_value, _, _ = stats.linregress(feature, target_clean)
            metrics[f'feature_{i}_r2'] = r_value ** 2
            metrics[f'feature_{i}_slope'] = slope
        
        return metrics
    
    def evaluate_feature_stability(
        self,
        features: np.ndarray,
        windows: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Evaluate feature stability across different time windows"""
        if windows is None:
            windows = [22, 63, 126, 252]  # 1M, 3M, 6M, 1Y
            
        stability_metrics = {}
        
        for window in windows:
            rolling_means = pd.DataFrame(features).rolling(window=window).mean()
            rolling_stds = pd.DataFrame(features).rolling(window=window).std()
            
            # Calculate coefficient of variation
            cv = rolling_stds / rolling_means
            
            for i in range(features.shape[1]):
                stability_metrics[f'feature_{i}_cv_{window}d'] = cv.iloc[:, i].mean()
                stability_metrics[f'feature_{i}_cv_std_{window}d'] = cv.iloc[:, i].std()
        
        return stability_metrics
    
    def perform_cross_validation(
        self,
        features: np.ndarray,
        target: np.ndarray,
        model: BaseModel
    ) -> Dict[str, Any]:
        """Perform cross-validation to evaluate feature predictive power"""
        cv_metrics = {}
        
        # Remove NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features_clean = features[valid_mask]
        target_clean = target[valid_mask]
        
        # Split data into folds
        fold_size = len(features_clean) // self.cv_folds
        
        for fold in range(self.cv_folds):
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < self.cv_folds - 1 else len(features_clean)
            
            # Create train/test split
            test_mask = np.zeros(len(features_clean), dtype=bool)
            test_mask[start_idx:end_idx] = True
            
            X_train = features_clean[~test_mask]
            y_train = target_clean[~test_mask]
            X_test = features_clean[test_mask]
            y_test = target_clean[test_mask]
            
            # Train and evaluate model
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            # Calculate metrics
            if self.target_type == 'binary':
                accuracy = np.mean(predictions == y_test)
                cv_metrics[f'fold_{fold}_accuracy'] = accuracy
            else:
                r2 = r2_score(y_test, predictions)
                cv_metrics[f'fold_{fold}_r2'] = r2
        
        # Calculate average metrics
        if self.target_type == 'binary':
            cv_metrics['mean_accuracy'] = np.mean([
                v for k, v in cv_metrics.items() if k.endswith('accuracy')
            ])
        else:
            cv_metrics['mean_r2'] = np.mean([
                v for k, v in cv_metrics.items() if k.endswith('r2')
            ])
        
        return cv_metrics
    
    def backtest_features(
        self,
        composition: FeatureComposition,
        model: Optional[BaseModel] = None
    ) -> Dict[str, Any]:
        """Run complete feature backtest"""
        try:
            # Calculate features
            builder = FeatureBuilder(self.data)
            
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
                    builder.add_sector_features(
                        self.data.get('sector_data', {}),
                        **feature_config.parameters
                    )
                elif feature_config.type == "yield_curve":
                    builder.add_yield_curve_features(
                        self.data.get('yield_data', {}),
                        **feature_config.parameters
                    )
            
            features = builder.build()
            target = self.prepare_target()
            
            # Calculate all metrics
            results = {
                'feature_metrics': self.calculate_feature_metrics(features, target),
                'stability_metrics': self.evaluate_feature_stability(features)
            }
            
            if model is not None:
                results['cv_metrics'] = self.perform_cross_validation(
                    features, target, model
                )
            
            # Store results
            self.backtest_results[composition.id] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error in feature backtest: {str(e)}")
            return {
                'error': str(e),
                'feature_metrics': {},
                'stability_metrics': {},
                'cv_metrics': {}
            }
    
    def get_feature_importance(
        self,
        composition_id: str
    ) -> List[Tuple[str, float]]:
        """Get feature importance ranking"""
        if composition_id not in self.backtest_results:
            return []
            
        results = self.backtest_results[composition_id]
        feature_metrics = results['feature_metrics']
        
        # Combine correlation and mutual information scores
        importance_scores = []
        feature_count = len([k for k in feature_metrics.keys() if k.endswith('correlation')]) 
        
        for i in range(feature_count):
            corr = abs(feature_metrics.get(f'feature_{i}_correlation', 0))
            mi = feature_metrics.get(f'feature_{i}_mutual_info', 0)
            r2 = feature_metrics.get(f'feature_{i}_r2', 0)
            
            # Weighted combination of metrics
            score = (0.4 * corr + 0.4 * mi + 0.2 * r2)
            importance_scores.append((f'feature_{i}', score))
        
        # Sort by importance score
        return sorted(importance_scores, key=lambda x: x[1], reverse=True)
    
    def get_feature_recommendations(
        self,
        composition_id: str
    ) -> List[str]:
        """Get recommendations for feature improvements"""
        if composition_id not in self.backtest_results:
            return []
            
        results = self.backtest_results[composition_id]
        recommendations = []
        
        # Check feature correlations
        for k, v in results['feature_metrics'].items():
            if k.endswith('correlation'):
                if abs(v) < 0.1:
                    recommendations.append(
                        f"Feature {k.split('_')[1]} has very low correlation with target"
                    )
        
        # Check stability
        stability = results['stability_metrics']
        for k, v in stability.items():
            if k.endswith('cv_252d') and v > 1.0:  # High annual CV
                recommendations.append(
                    f"Feature {k.split('_')[1]} shows high instability over 1-year period"
                )
        
        # Check predictive power if CV metrics exist
        cv_metrics = results.get('cv_metrics', {})
        if cv_metrics:
            if self.target_type == 'binary':
                if cv_metrics.get('mean_accuracy', 0) < 0.55:
                    recommendations.append(
                        "Features show weak predictive power, consider adding more features"
                    )
            else:
                if cv_metrics.get('mean_r2', 0) < 0.1:
                    recommendations.append(
                        "Features explain very little variance in target, consider feature engineering"
                    )
        
        return recommendations