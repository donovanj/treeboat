import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from financial_prediction_system.core.ensemble.gradient_boosting import GradientBoostingEnsemble


class SimpleRegressionModel(nn.Module):
    """Simple regression model for testing GradientBoosting"""
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        super(SimpleRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestGradientBoostingEnsemble:
    """Tests for the Gradient Boosting ensemble implementation."""
    
    @pytest.fixture
    def model_factory(self):
        """Model factory fixture."""
        return lambda: SimpleRegressionModel(input_dim=5, hidden_dim=10, output_dim=1)
    
    @pytest.fixture
    def regression_dataset(self):
        """Regression dataset fixture."""
        torch.manual_seed(42)
        features = torch.rand(100, 5)
        targets = torch.rand(100, 1)  # Continuous targets for regression
        return TensorDataset(features, targets)
    
    @pytest.fixture
    def binary_dataset(self):
        """Binary classification dataset fixture."""
        torch.manual_seed(42)
        features = torch.rand(100, 5)
        targets = torch.randint(0, 2, (100, 1)).float()  # Binary targets (0 or 1)
        return TensorDataset(features, targets)
    
    def test_init(self, model_factory):
        """Test initialization with default parameters."""
        # Arrange & Act
        ensemble = GradientBoostingEnsemble(base_model_factory=model_factory)
        
        # Assert
        assert ensemble.n_estimators == 100
        assert ensemble.learning_rate == 0.1
        assert ensemble.subsample == 1.0
        assert ensemble.loss == 'mse'
        assert isinstance(ensemble.device, torch.device)
        assert len(ensemble.models) == 0  # Models are added during fit
        assert ensemble.initial_prediction is None
    
    def test_init_custom_params(self, model_factory):
        """Test initialization with custom parameters."""
        # Arrange & Act
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=50,
            learning_rate=0.05,
            subsample=0.8,
            loss='log_loss',
            device=torch.device('cpu')
        )
        
        # Assert
        assert ensemble.n_estimators == 50
        assert ensemble.learning_rate == 0.05
        assert ensemble.subsample == 0.8
        assert ensemble.loss == 'log_loss'
        assert ensemble.device == torch.device('cpu')
    
    def test_fit_regression(self, model_factory, regression_dataset):
        """Test fitting GradientBoosting on regression data."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=3,
            loss='mse',
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_per_model=2,
            verbose=False
        )
        
        # Assert
        assert ensemble.initial_prediction is not None
        assert len(ensemble.models) == 3
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_fit_binary_classification(self, model_factory, binary_dataset):
        """Test fitting GradientBoosting on binary classification data."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=3,
            loss='log_loss',
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_per_model=2,
            verbose=False
        )
        
        # Assert
        assert ensemble.initial_prediction is not None
        assert len(ensemble.models) == 3
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_forward_regression(self, model_factory, regression_dataset):
        """Test forward pass for regression."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            loss='mse',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_per_model=2,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = regression_dataset[0:5]
        
        # Act
        with torch.no_grad():
            output = ensemble(features)
        
        # Assert
        assert output.shape == (5, 1)
        assert output.dtype == torch.float32
    
    def test_forward_binary_classification(self, model_factory, binary_dataset):
        """Test forward pass for binary classification."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            loss='log_loss',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_per_model=2,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = binary_dataset[0:5]
        
        # Act
        with torch.no_grad():
            output = ensemble(features)
        
        # Assert
        assert output.shape == (5, 1)
        assert output.dtype == torch.float32
        assert torch.all((output >= 0.0) & (output <= 1.0))  # Probabilities between 0 and 1
    
    def test_fit_with_validation_and_early_stopping(self, model_factory, regression_dataset):
        """Test fitting with validation data and early stopping."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=10,  # Set higher to test early stopping
            device=torch.device('cpu')
        )
        
        # Split dataset for validation
        train_size = int(0.8 * len(regression_dataset))
        val_size = len(regression_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            regression_dataset, [train_size, val_size]
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            batch_size=32,
            epochs_per_model=2,
            early_stopping=2,  # Stop after 2 rounds of no improvement
            verbose=False
        )
        
        # Assert
        assert len(ensemble.models) > 0
        assert len(ensemble.models) <= 10  # Should stop early
        assert 'val_loss' in history
        assert len(history['val_loss']) > 0
    
    def test_error_when_not_fitted(self, model_factory):
        """Test that error is raised if forward is called before fit."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=3,
            device=torch.device('cpu')
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError):
            features = torch.rand(5, 5)
            ensemble(features)
    
    def test_subsample_less_than_one(self, model_factory, regression_dataset):
        """Test that subsampling works correctly."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            subsample=0.5,  # Use half the data for each estimator
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_per_model=2,
            verbose=False
        )
        
        # Assert - Just verify it runs without errors
        assert len(ensemble.models) == 2
    
    def test_save_and_load(self, model_factory, regression_dataset, tmp_path):
        """Test saving and loading the ensemble."""
        # Arrange
        ensemble = GradientBoostingEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            device=torch.device('cpu')
        )
        
        # Fit the ensemble
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_per_model=2,
            verbose=False
        )
        
        save_path = tmp_path / "gb_test.pt"
        
        # Act - Save
        ensemble.save(str(save_path))
        
        # Act - Load
        loaded_ensemble = GradientBoostingEnsemble.load(
            str(save_path),
            base_model_factory=model_factory,
            device=torch.device('cpu')
        )
        
        # Assert
        assert loaded_ensemble.n_estimators == ensemble.n_estimators
        assert loaded_ensemble.learning_rate == ensemble.learning_rate
        assert loaded_ensemble.subsample == ensemble.subsample
        assert loaded_ensemble.loss == ensemble.loss
        assert loaded_ensemble.initial_prediction == ensemble.initial_prediction
        assert len(loaded_ensemble.models) == len(ensemble.models)
        
        # Test predictions match
        features, _ = regression_dataset[0:5]
        with torch.no_grad():
            original_output = ensemble(features)
            loaded_output = loaded_ensemble(features)
            
        assert torch.allclose(original_output, loaded_output) 