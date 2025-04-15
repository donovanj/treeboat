import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from financial_prediction_system.core.ensemble.ada_boost import AdaBoostEnsemble


class SimpleModel(nn.Module):
    """Simple model for testing AdaBoost"""
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestAdaBoostEnsemble:
    """Tests for the AdaBoost ensemble implementation."""
    
    @pytest.fixture
    def model_factory(self):
        """Model factory fixture."""
        return lambda: SimpleModel(input_dim=5, hidden_dim=10, output_dim=1)
    
    @pytest.fixture
    def binary_dataset(self):
        """Binary classification dataset fixture."""
        torch.manual_seed(42)
        features = torch.rand(100, 5)
        # Create binary targets (0 or 1)
        targets = torch.randint(0, 2, (100, 1)).float()
        return TensorDataset(features, targets)
    
    @pytest.fixture
    def multiclass_dataset(self):
        """Multi-class classification dataset fixture."""
        torch.manual_seed(42)
        features = torch.rand(100, 5)
        # Create multi-class targets (0, 1, or 2)
        targets = torch.randint(0, 3, (100,))
        return TensorDataset(features, targets)
    
    def test_init(self, model_factory):
        """Test initialization with default parameters."""
        # Arrange & Act
        ensemble = AdaBoostEnsemble(base_model_factory=model_factory, n_estimators=5)
        
        # Assert
        assert ensemble.n_estimators == 5
        assert ensemble.learning_rate == 1.0
        assert isinstance(ensemble.device, torch.device)
        assert len(ensemble.models) == 0  # Models are added during fit
        assert len(ensemble.model_weights) == 0
        
    def test_init_custom_params(self, model_factory):
        """Test initialization with custom parameters."""
        # Arrange & Act
        ensemble = AdaBoostEnsemble(
            base_model_factory=model_factory,
            n_estimators=10,
            learning_rate=0.5,
            device=torch.device('cpu')
        )
        
        # Assert
        assert ensemble.n_estimators == 10
        assert ensemble.learning_rate == 0.5
        assert ensemble.device == torch.device('cpu')
    
    def test_fit_binary_classification(self, model_factory, binary_dataset):
        """Test fitting AdaBoost on binary classification data."""
        # Arrange
        ensemble = AdaBoostEnsemble(
            base_model_factory=model_factory,
            n_estimators=3,
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_per_model=2
        )
        
        # Assert
        assert len(ensemble.models) == 3
        assert len(ensemble.model_weights) == 3
        assert all(isinstance(w, float) for w in ensemble.model_weights)
        assert len(history) == 3  # One history dict per model
    
    def test_forward_binary_classification(self, model_factory, binary_dataset):
        """Test forward pass for binary classification."""
        # Arrange
        ensemble = AdaBoostEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            device=torch.device('cpu')
        )
        ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_per_model=2
        )
        
        # Sample features from the dataset
        features, _ = binary_dataset[0:5]
        
        # Act
        with torch.no_grad():
            output = ensemble(features)
        
        # Assert
        assert output.shape == (5, 1)
        assert torch.all((output == 1.0) | (output == -1.0))  # Binary classification outputs
    
    def test_fit_with_validation(self, model_factory, binary_dataset):
        """Test fitting with validation data."""
        # Arrange
        ensemble = AdaBoostEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            device=torch.device('cpu')
        )
        
        # Split dataset for validation
        train_size = int(0.8 * len(binary_dataset))
        val_size = len(binary_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            binary_dataset, [train_size, val_size]
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            batch_size=32,
            epochs_per_model=2
        )
        
        # Assert
        assert len(ensemble.models) == 2
        assert len(history) == 2
        for model_history in history:
            assert 'val_loss' in model_history
            assert 'val_accuracy' in model_history
    
    def test_error_when_no_models(self, model_factory):
        """Test that error is raised if forward is called before fit."""
        # Arrange
        ensemble = AdaBoostEnsemble(
            base_model_factory=model_factory,
            n_estimators=3,
            device=torch.device('cpu')
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError):
            features = torch.rand(5, 5)
            ensemble(features)
    
    def test_save_and_load(self, model_factory, binary_dataset, tmp_path):
        """Test saving and loading the ensemble."""
        # Arrange
        ensemble = AdaBoostEnsemble(
            base_model_factory=model_factory,
            n_estimators=2,
            device=torch.device('cpu')
        )
        
        # Fit the ensemble
        ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_per_model=2
        )
        
        save_path = tmp_path / "adaboost_test.pt"
        
        # Act - Save
        ensemble.save(str(save_path))
        
        # Act - Load
        loaded_ensemble = AdaBoostEnsemble.load(
            str(save_path),
            base_model_factory=model_factory,
            device=torch.device('cpu')
        )
        
        # Assert
        assert loaded_ensemble.n_estimators == ensemble.n_estimators
        assert loaded_ensemble.learning_rate == ensemble.learning_rate
        assert len(loaded_ensemble.models) == len(ensemble.models)
        assert loaded_ensemble.model_weights == ensemble.model_weights
        
        # Test predictions match
        features, _ = binary_dataset[0:5]
        with torch.no_grad():
            original_output = ensemble(features)
            loaded_output = loaded_ensemble(features)
            
        assert torch.all(original_output == loaded_output) 