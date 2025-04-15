"""
Training Pipeline Module

This module implements a complete pipeline for training prediction models,
connecting feature builders, target builders, model factories, and evaluation components.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.core.features.model_feature_builder import ModelFeaturePipeline
from financial_prediction_system.core.targets.model_target_pipeline import ModelTargetPipeline
from financial_prediction_system.core.models.factory import ModelFactory
from financial_prediction_system.core.evaluation.metrics import calculate_metrics


class TrainingObserver:
    """Observer interface for training events."""
    
    def update(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Update method called when a training event occurs
        
        Parameters
        ----------
        event_type : str
            Type of event (e.g., 'epoch_completed', 'training_completed')
        data : Dict[str, Any]
            Data associated with the event
        """
        pass


class TrainingLogger(TrainingObserver):
    """Concrete observer that logs training events."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the training logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log training events."""
        if event_type == 'training_started':
            self.logger.info(f"Training started with model: {data.get('model_type')}")
        elif event_type == 'epoch_completed':
            epoch = data.get('epoch')
            loss = data.get('loss', 0)
            self.logger.info(f"Epoch {epoch} completed, loss: {loss:.6f}")
        elif event_type == 'training_completed':
            self.logger.info(f"Training completed in {data.get('duration', 0):.2f} seconds")
        elif event_type == 'evaluation_result':
            metrics = data.get('metrics', {})
            metrics_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            self.logger.info(f"Evaluation results: {metrics_str}")
        elif event_type == 'error':
            self.logger.error(f"Error during training: {data.get('message')}")


class ModelTrainer:
    """
    Orchestrates the process of training a prediction model
    
    This class follows the Observer pattern to notify interested
    parties about training events.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the model trainer
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            Configuration dictionary for the training process
        """
        self.config = config or {}
        self.model = None
        self.feature_pipeline = None
        self.target_pipeline = None
        self.observers = []
    
    def add_observer(self, observer: TrainingObserver) -> None:
        """
        Add an observer to be notified of training events
        
        Parameters
        ----------
        observer : TrainingObserver
            Observer to add
        """
        self.observers.append(observer)
    
    def remove_observer(self, observer: TrainingObserver) -> None:
        """
        Remove an observer
        
        Parameters
        ----------
        observer : TrainingObserver
            Observer to remove
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def notify_observers(self, event_type: str, data: Dict[str, Any] = None) -> None:
        """
        Notify all observers of an event
        
        Parameters
        ----------
        event_type : str
            Type of event
        data : Dict[str, Any], optional
            Data associated with the event
        """
        data = data or {}
        for observer in self.observers:
            observer.update(event_type, data)
    
    def train(self, data: pd.DataFrame, target_column: str = None, model_type: str = 'transformer',
              validation_data: Optional[pd.DataFrame] = None, model_params: Dict[str, Any] = None,
              feature_config: Dict[str, Any] = None, target_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train a model
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
        target_column : str, optional
            Name of the target column if using direct target, not generated targets
        model_type : str, default='transformer'
            Type of model to train (must be registered with ModelFactory)
        validation_data : pd.DataFrame, optional
            Validation data
        model_params : Dict[str, Any], optional
            Parameters for model initialization
        feature_config : Dict[str, Any], optional
            Configuration for feature extraction
        target_config : Dict[str, Any], optional
            Configuration for target preparation
            
        Returns
        -------
        Dict[str, Any]
            Results of the training process
        """
        import time
        start_time = time.time()
        
        try:
            # Prepare feature configuration
            feature_config = feature_config or self.config.get('feature_config', {})
            
            # Prepare target configuration
            target_config = target_config or self.config.get('target_config', {})
            
            # Notify training started
            self.notify_observers('training_started', {
                'model_type': model_type,
                'data_shape': data.shape,
                'feature_config': feature_config,
                'target_config': target_config
            })
            
            # Initialize feature pipeline
            self.feature_pipeline = ModelFeaturePipeline(config=feature_config)
            
            # Decide how to handle targets
            if target_column is not None and target_column in data.columns:
                # Use provided target column directly
                data_copy = data.copy()
                target_series = data_copy[target_column]
                data_copy = data_copy.drop(columns=[target_column])
                
                # Prepare features (without target)
                features = self.feature_pipeline.prepare_features(data_copy)[0]
                targets = pd.DataFrame({target_column: target_series})
                
                # For validation data
                if validation_data is not None:
                    val_data_copy = validation_data.copy()
                    val_target_series = val_data_copy[target_column]
                    val_data_copy = val_data_copy.drop(columns=[target_column])
                    
                    val_features = self.feature_pipeline.prepare_features(val_data_copy)[0]
                    val_targets = pd.DataFrame({target_column: val_target_series})
                else:
                    val_features = val_targets = None
                    
            else:
                # Generate targets using target pipeline
                self.target_pipeline = ModelTargetPipeline(config=target_config)
                
                # Prepare features (without extracting target from data)
                features, _ = self.feature_pipeline.prepare_features(data)
                
                # Prepare targets
                targets = self.target_pipeline.prepare_targets(data)
                
                # Get primary target name
                primary_target_name = target_config.get('primary_target')
                if primary_target_name is None and self.target_pipeline.target_columns:
                    primary_target_name = self.target_pipeline.target_columns[0]
                
                # For validation data
                if validation_data is not None:
                    val_features, _ = self.feature_pipeline.prepare_features(validation_data)
                    val_targets = self.target_pipeline.prepare_targets(validation_data)
                else:
                    val_features = val_targets = None
            
            # Get model parameters
            model_params = model_params or self.config.get('model_params', {})
            
            # Set input size from feature dimensions
            if 'input_size' not in model_params:
                model_params['input_size'] = features.shape[1]
            
            # Create model
            self.model = ModelFactory.create_model(model_type, **model_params)
            
            # Check if model is a sequence model
            is_sequence_model = model_type in ['transformer', 'lstm']
            sequence_length = model_params.get('sequence_length', 20)
            
            # Prepare data according to model type
            if is_sequence_model:
                # Prepare sequence data for sequence models
                if self.target_pipeline is not None:
                    # Use target pipeline for sequence preparation
                    features_np, targets_np = self.feature_pipeline.prepare_sequence_data(
                        features, None, sequence_length=sequence_length
                    )
                    targets_np = self.target_pipeline.prepare_sequence_targets(
                        targets, sequence_length=sequence_length
                    )
                else:
                    # Use feature pipeline for sequence preparation
                    features_np, targets_np = self.feature_pipeline.prepare_sequence_data(
                        features, targets[target_column], sequence_length=sequence_length
                    )
                
                # Prepare validation sequence data if provided
                val_features_np = val_targets_np = None
                if val_features is not None and val_targets is not None:
                    if self.target_pipeline is not None:
                        val_features_np, _ = self.feature_pipeline.prepare_sequence_data(
                            val_features, None, sequence_length=sequence_length
                        )
                        val_targets_np = self.target_pipeline.prepare_sequence_targets(
                            val_targets, sequence_length=sequence_length
                        )
                    else:
                        val_features_np, val_targets_np = self.feature_pipeline.prepare_sequence_data(
                            val_features, val_targets[target_column], sequence_length=sequence_length
                        )
            else:
                # Use standard data for non-sequence models
                features_np = features.values
                
                if self.target_pipeline is not None and primary_target_name is not None:
                    # Use the specific primary target
                    targets_np = targets[primary_target_name].values
                elif target_column is not None:
                    # Use the provided target column
                    targets_np = targets[target_column].values
                else:
                    # Default to first target column
                    targets_np = targets.iloc[:, 0].values
                
                # Validation data
                val_features_np = val_features.values if val_features is not None else None
                
                if val_targets is not None:
                    if self.target_pipeline is not None and primary_target_name is not None:
                        val_targets_np = val_targets[primary_target_name].values
                    elif target_column is not None:
                        val_targets_np = val_targets[target_column].values
                    else:
                        val_targets_np = val_targets.iloc[:, 0].values
                else:
                    val_targets_np = None
            
            # Prepare validation data tuple if available
            validation_data_tuple = None
            if val_features_np is not None and val_targets_np is not None:
                validation_data_tuple = (val_features_np, val_targets_np)
            
            # Define training callback to capture epoch progress
            def training_callback(epoch, loss):
                self.notify_observers('epoch_completed', {'epoch': epoch, 'loss': loss})
            
            # Train the model
            training_history = self.model.train(
                features_np, targets_np, 
                validation_data=validation_data_tuple,
                callback=training_callback
            )
            
            # Calculate training duration
            duration = time.time() - start_time
            
            # Notify training completed
            self.notify_observers('training_completed', {
                'duration': duration,
                'history': training_history
            })
            
            # Evaluate model if validation data is available
            if validation_data_tuple is not None:
                metrics = self.model.evaluate(val_features_np, val_targets_np)
                self.notify_observers('evaluation_result', {'metrics': metrics})
            
            # Save model if path provided
            model_save_path = self.config.get('model_save_path')
            if model_save_path:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                self.model.save(model_save_path)
            
            # Return results
            return {
                'model': self.model,
                'feature_pipeline': self.feature_pipeline,
                'target_pipeline': self.target_pipeline,
                'training_history': training_history,
                'duration': duration
            }
            
        except Exception as e:
            self.notify_observers('error', {'message': str(e)})
            raise
    
    def _prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Prepare sequences for time series models
        
        Parameters
        ----------
        data : np.ndarray
            Input data of shape (samples, features)
        sequence_length : int
            Length of sequences to create
            
        Returns
        -------
        np.ndarray
            Sequences of shape (n_sequences, sequence_length, features)
        """
        n_samples, n_features = data.shape
        n_sequences = n_samples - sequence_length + 1
        
        # Create sequences
        sequences = np.zeros((n_sequences, sequence_length, n_features))
        for i in range(n_sequences):
            sequences[i] = data[i:i + sequence_length]
        
        return sequences


def train_model(data: pd.DataFrame, target_col: str = None, model_type: str = 'transformer',
                model_params: Dict[str, Any] = None, feature_config: Dict[str, Any] = None,
                target_config: Dict[str, Any] = None, save_path: str = None, 
                log_level: int = logging.INFO) -> Dict[str, Any]:
    """
    Convenience function to train a model
    
    Parameters
    ----------
    data : pd.DataFrame
        Training data
    target_col : str, optional
        Name of the target column if using direct target, not generated targets
    model_type : str, default='transformer'
        Type of model to train
    model_params : Dict[str, Any], optional
        Parameters for model initialization
    feature_config : Dict[str, Any], optional
        Configuration for feature extraction
    target_config : Dict[str, Any], optional
        Configuration for target preparation
    save_path : str, optional
        Path to save the trained model
    log_level : int, default=logging.INFO
        Logging level
        
    Returns
    -------
    Dict[str, Any]
        Results of the training process
    """
    # Create config
    config = {
        'model_params': model_params or {},
        'feature_config': feature_config or {},
        'target_config': target_config or {},
        'model_save_path': save_path
    }
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Add logger observer
    logger = TrainingLogger(log_level)
    trainer.add_observer(logger)
    
    # Create train/validation split (70/30)
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data, test_size=0.3, shuffle=False)
    
    # Train model
    results = trainer.train(train_data, target_col, model_type, val_data)
    
    return results
