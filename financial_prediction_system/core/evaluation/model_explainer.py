"""Model explanation utilities using SHAP

This module provides explanations for machine learning models
using SHAP (SHapley Additive exPlanations).
"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Provides SHAP-based explanations for machine learning models.
    
    This class wraps different SHAP explainers to provide a consistent
    interface for explaining various types of models.
    """
    
    def __init__(self, model: Any, model_type: str, feature_names: List[str] = None):
        """Initialize the explainer.
        
        Args:
            model: The trained model to explain
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel', etc.)
            feature_names: Names of features (optional)
        """
        self.model = model
        self.model_type = model_type.lower()
        self.feature_names = feature_names
        self.explainer = None
        self._create_explainer()
    
    def _create_explainer(self):
        """Create the appropriate SHAP explainer based on model type."""
        try:
            # Select appropriate explainer based on model type
            if self.model_type in ['xgboost', 'lightgbm', 'catboost', 'randomforest', 'gbm', 'tree']:
                # For tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type in ['linear', 'logistic', 'ridge', 'lasso']:
                # For linear models
                self.explainer = shap.LinearExplainer(self.model, self.background_data if hasattr(self, 'background_data') else None)
            elif self.model_type in ['nn', 'neural', 'deep', 'transformer', 'lstm']:
                # For deep learning models
                self.explainer = shap.DeepExplainer(self.model, self.background_data if hasattr(self, 'background_data') else None)
            else:
                # Default to KernelExplainer for black-box models
                logger.info(f"Using KernelExplainer for model type: {self.model_type}")
                self.explainer = shap.KernelExplainer(self.model.predict, self.background_data if hasattr(self, 'background_data') else None)
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {str(e)}")
            raise
    
    def set_background_data(self, background_data: Union[pd.DataFrame, np.ndarray]):
        """Set background data for explainers that require it.
        
        Args:
            background_data: A sample of background data for SHAP to use as reference
        """
        self.background_data = background_data
        # Re-initialize explainer with background data
        self._create_explainer()
    
    def explain(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Generate SHAP explanation values for the provided data.
        
        Args:
            data: Data to explain predictions for
            
        Returns:
            Dictionary containing explanation results
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call set_background_data first for this type of model.")
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(data)
            
            # Format depends on model type and output shape
            if isinstance(shap_values, list):
                # Multi-output case (e.g., multi-class classification)
                shap_results = {"class_" + str(i): values.tolist() for i, values in enumerate(shap_values)}
            else:
                # Single output case
                shap_results = {"shap_values": shap_values.tolist()}
            
            # Add expected value to results
            if hasattr(self.explainer, "expected_value"):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, list):
                    shap_results["expected_values"] = [float(v) for v in expected_value]
                else:
                    shap_results["expected_value"] = float(expected_value)
                    
            # Create feature importance summary
            feature_importance = self._calculate_feature_importance(shap_values)
            
            # Combine all results
            results = {
                "shap_results": shap_results,
                "feature_importance": feature_importance
            }
            
            return results
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            raise
    
    def _calculate_feature_importance(self, shap_values) -> Dict[str, float]:
        """Calculate overall feature importance from SHAP values.
        
        Args:
            shap_values: SHAP values from explainer
            
        Returns:
            Dictionary of feature names and their importance scores
        """
        feature_names = self.feature_names if self.feature_names else [f"feature_{i}" for i in range(shap_values.shape[-1] if isinstance(shap_values, np.ndarray) else shap_values[0].shape[-1])]
        
        try:
            # Handle different shapes of shap_values
            if isinstance(shap_values, list):
                # Multi-output case - average across all outputs
                importance = np.abs(np.array(shap_values)).mean(axis=0).mean(axis=0)
            else:
                # Single output case
                importance = np.abs(shap_values).mean(axis=0)
            
            # Create sorted importance dictionary
            importance_dict = dict(sorted(
                zip(feature_names, importance.tolist()), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            return importance_dict
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def save_summary_plot(self, data: Union[pd.DataFrame, np.ndarray], 
                          output_path: str, 
                          plot_type: str = 'bar', 
                          max_display: int = 20) -> Optional[str]:
        """Generate and save a SHAP summary plot.
        
        Args:
            data: Data to explain
            output_path: Path to save the plot
            plot_type: Type of plot ('bar', 'dot', 'violin', etc.)
            max_display: Maximum number of features to display
            
        Returns:
            Path to the saved plot file or None if there was an error
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(data)
            
            # Create plot
            if plot_type == 'bar':
                shap.summary_plot(shap_values, data, plot_type='bar', feature_names=self.feature_names, max_display=max_display, show=False)
            else:
                shap.summary_plot(shap_values, data, feature_names=self.feature_names, max_display=max_display, show=False)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error saving SHAP summary plot: {str(e)}")
            return None
    
    def generate_waterfall_plot(self, data: Union[pd.DataFrame, np.ndarray], 
                               instance_index: int, 
                               output_path: str) -> Optional[str]:
        """Generate and save a SHAP waterfall plot for a specific instance.
        
        Args:
            data: Data containing the instance to explain
            instance_index: Index of the instance to explain
            output_path: Path to save the plot
            
        Returns:
            Path to the saved plot file or None if there was an error
        """
        try:
            plt.figure(figsize=(10, 8))
            
            # Get instance data
            instance = data.iloc[instance_index] if isinstance(data, pd.DataFrame) else data[instance_index]
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(data)
            
            # Format for waterfall plot depends on model type
            if isinstance(shap_values, list):
                # For multi-class, use the predicted class
                predicted_class = np.argmax(self.model.predict(instance.reshape(1, -1))[0]) if hasattr(self.model, 'predict') else 0
                instance_shap_values = shap_values[predicted_class][instance_index]
                expected_value = self.explainer.expected_value[predicted_class]
            else:
                instance_shap_values = shap_values[instance_index]
                expected_value = self.explainer.expected_value
            
            # Create waterfall plot
            shap.plots.waterfall(shap.Explanation(
                values=instance_shap_values, 
                base_values=expected_value, 
                data=instance,
                feature_names=self.feature_names if self.feature_names else None
            ), show=False)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
        except Exception as e:
            logger.error(f"Error generating waterfall plot: {str(e)}")
            return None
    
    def explain_prediction(self, instance: Union[pd.DataFrame, np.ndarray], 
                           num_features: int = 10) -> Dict[str, Any]:
        """Explain a single prediction in detail.
        
        Args:
            instance: Single data instance to explain
            num_features: Number of top features to include in explanation
            
        Returns:
            Dictionary containing the explanation
        """
        # Ensure instance is properly formatted
        if isinstance(instance, pd.DataFrame) and len(instance) > 1:
            instance = instance.iloc[0:1]
        elif isinstance(instance, np.ndarray) and instance.ndim > 1 and len(instance) > 1:
            instance = instance[0:1]
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(instance)
        
        # Format explanation based on model type
        if isinstance(shap_values, list):
            # Multi-class case
            result = {}
            for i, class_shap in enumerate(shap_values):
                values = class_shap[0] if class_shap.ndim > 1 else class_shap
                if self.feature_names:
                    # Sort features by absolute SHAP value
                    features_shap = sorted(zip(self.feature_names, values), 
                                           key=lambda x: abs(x[1]), 
                                           reverse=True)
                    # Take top N features
                    top_features = features_shap[:num_features]
                    result[f"class_{i}"] = {
                        "base_value": float(self.explainer.expected_value[i]),
                        "prediction": float(self.explainer.expected_value[i] + sum(values)),
                        "top_features": [{
                            "feature": feature,
                            "shap_value": float(value),
                            "direction": "positive" if value > 0 else "negative"
                        } for feature, value in top_features]
                    }
                else:
                    # Without feature names
                    top_indices = np.argsort(np.abs(values))[::-1][:num_features]
                    result[f"class_{i}"] = {
                        "base_value": float(self.explainer.expected_value[i]),
                        "prediction": float(self.explainer.expected_value[i] + sum(values)),
                        "top_features": [{
                            "feature": f"feature_{j}",
                            "shap_value": float(values[j]),
                            "direction": "positive" if values[j] > 0 else "negative"
                        } for j in top_indices]
                    }
        else:
            # Single output case
            values = shap_values[0] if shap_values.ndim > 1 else shap_values
            if self.feature_names:
                # Sort features by absolute SHAP value
                features_shap = sorted(zip(self.feature_names, values), 
                                       key=lambda x: abs(x[1]), 
                                       reverse=True)
                # Take top N features
                top_features = features_shap[:num_features]
                result = {
                    "base_value": float(self.explainer.expected_value),
                    "prediction": float(self.explainer.expected_value + sum(values)),
                    "top_features": [{
                        "feature": feature,
                        "shap_value": float(value),
                        "direction": "positive" if value > 0 else "negative"
                    } for feature, value in top_features]
                }
            else:
                # Without feature names
                top_indices = np.argsort(np.abs(values))[::-1][:num_features]
                result = {
                    "base_value": float(self.explainer.expected_value),
                    "prediction": float(self.explainer.expected_value + sum(values)),
                    "top_features": [{
                        "feature": f"feature_{j}",
                        "shap_value": float(values[j]),
                        "direction": "positive" if values[j] > 0 else "negative"
                    } for j in top_indices]
                }
        
        return result
    
    def save_explainer(self, path: str) -> bool:
        """Save the explainer to disk.
        
        Args:
            path: Path to save the explainer
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save explainer
            with open(path, 'wb') as f:
                pickle.dump(self.explainer, f)
                
            return True
        except Exception as e:
            logger.error(f"Error saving explainer: {str(e)}")
            return False
    
    @staticmethod
    def load_explainer(path: str, model: Any = None, model_type: str = None, feature_names: List[str] = None) -> 'ModelExplainer':
        """Load a saved explainer from disk.
        
        Args:
            path: Path to the saved explainer
            model: Optional model to associate with the explainer
            model_type: Optional model type
            feature_names: Optional feature names
            
        Returns:
            ModelExplainer instance
        """
        try:
            # Ensure the explainer exists
            if not os.path.exists(path):
                raise FileNotFoundError(f"Explainer not found at {path}")
                
            # Load explainer
            with open(path, 'rb') as f:
                explainer = pickle.load(f)
                
            # Create ModelExplainer instance
            model_explainer = ModelExplainer.__new__(ModelExplainer)
            model_explainer.explainer = explainer
            model_explainer.model = model
            model_explainer.model_type = model_type
            model_explainer.feature_names = feature_names
            
            return model_explainer
        except Exception as e:
            logger.error(f"Error loading explainer: {str(e)}")
            raise 