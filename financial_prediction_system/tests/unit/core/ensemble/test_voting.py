import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from financial_prediction_system.core.ensemble.voting import VotingEnsemble


class RegressionModel(nn.Module):
    """Simple regression model for testing Voting"""
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class BinaryClassificationModel(nn.Module):
    """Simple binary classification model for testing Voting"""
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MultiClassModel(nn.Module):
    """Multi-class model for testing Voting"""
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=3):
        super(MultiClassModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestVotingEnsemble:
    """Tests for the Voting ensemble implementation."""
    
    @pytest.fixture
    def regression_models(self):
        """Create a set of regression models."""
        models = [
            RegressionModel(input_dim=5, hidden_dim=8, output_dim=1),
            RegressionModel(input_dim=5, hidden_dim=12, output_dim=1),
            RegressionModel(input_dim=5, hidden_dim=16, output_dim=1)
        ]
        return models
    
    @pytest.fixture
    def binary_classification_models(self):
        """Create a set of binary classification models."""
        models = [
            BinaryClassificationModel(input_dim=5, hidden_dim=8, output_dim=1),
            BinaryClassificationModel(input_dim=5, hidden_dim=12, output_dim=1),
            BinaryClassificationModel(input_dim=5, hidden_dim=16, output_dim=1)
        ]
        return models
    
    @pytest.fixture
    def multiclass_models(self):
        """Create a set of multi-class classification models."""
        models = [
            MultiClassModel(input_dim=5, hidden_dim=8, output_dim=3),
            MultiClassModel(input_dim=5, hidden_dim=12, output_dim=3),
            MultiClassModel(input_dim=5, hidden_dim=16, output_dim=3)
        ]
        return models
    
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
    
    @pytest.fixture
    def multiclass_dataset(self):
        """Multi-class classification dataset fixture."""
        torch.manual_seed(42)
        features = torch.rand(100, 5)
        targets = torch.randint(0, 3, (100,))  # Multi-class targets (0, 1, or 2)
        return TensorDataset(features, targets)
    
    def test_init_regression(self, regression_models):
        """Test initialization for regression ensemble."""
        # Arrange & Act
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression'
        )
        
        # Assert
        assert ensemble.task_type == 'regression'
        assert ensemble.voting == 'soft'  # Default for regression
        assert len(ensemble.models) == 3
        assert isinstance(ensemble.device, torch.device)
        assert torch.allclose(ensemble.weights, torch.tensor([1/3, 1/3, 1/3]))
    
    def test_init_classification(self, binary_classification_models):
        """Test initialization for classification ensemble."""
        # Arrange & Act
        ensemble = VotingEnsemble(
            models=binary_classification_models,
            task_type='classification',
            voting='hard'
        )
        
        # Assert
        assert ensemble.task_type == 'classification'
        assert ensemble.voting == 'hard'
        assert len(ensemble.models) == 3
    
    def test_init_custom_weights(self, regression_models):
        """Test initialization with custom weights."""
        # Arrange
        weights = [0.5, 0.3, 0.2]
        
        # Act
        ensemble = VotingEnsemble(
            models=regression_models,
            weights=weights,
            task_type='regression'
        )
        
        # Assert
        assert torch.allclose(ensemble.weights, torch.tensor(weights))
    
    def test_error_hard_voting_regression(self, regression_models):
        """Test error when using hard voting with regression."""
        # Act & Assert
        with pytest.raises(ValueError):
            VotingEnsemble(
                models=regression_models,
                task_type='regression',
                voting='hard'
            )
    
    def test_fit_regression(self, regression_models, regression_dataset):
        """Test fitting VotingEnsemble on regression data."""
        # Arrange
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs=2,
            verbose=False
        )
        
        # Assert
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3  # One for each model
        for model_loss in history['train_loss']:
            assert len(model_loss) > 0
    
    def test_fit_binary_classification(self, binary_classification_models, binary_dataset):
        """Test fitting VotingEnsemble on binary classification data."""
        # Arrange
        ensemble = VotingEnsemble(
            models=binary_classification_models,
            task_type='classification',
            voting='soft',
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs=2,
            verbose=False
        )
        
        # Assert
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3  # One for each model
    
    def test_fit_multiclass(self, multiclass_models, multiclass_dataset):
        """Test fitting VotingEnsemble on multi-class data."""
        # Arrange
        ensemble = VotingEnsemble(
            models=multiclass_models,
            task_type='classification',
            voting='soft',
            device=torch.device('cpu')
        )
        
        # Act - Use the specialized fit_multiclass method
        history = ensemble.fit_multiclass(
            train_dataset=multiclass_dataset,
            batch_size=32,
            epochs=2,
            verbose=False
        )
        
        # Assert
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert len(history['train_loss']) == 3  # One for each model
    
    def test_forward_regression(self, regression_models, regression_dataset):
        """Test forward pass for regression."""
        # Arrange
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs=1,  # Just one epoch for testing
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
    
    def test_soft_voting_binary_classification(self, binary_classification_models, binary_dataset):
        """Test soft voting for binary classification."""
        # Arrange
        ensemble = VotingEnsemble(
            models=binary_classification_models,
            task_type='classification',
            voting='soft',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs=1,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = binary_dataset[0:5]
        
        # Act
        with torch.no_grad():
            proba = ensemble(features)
            predictions = ensemble.predict(features)
        
        # Assert
        assert proba.shape[0] == 5  # 5 samples
        assert torch.all((proba >= 0.0) & (proba <= 1.0))  # Probabilities between 0 and 1
        
        assert predictions.shape == (5,)
        assert torch.all((predictions == 0) | (predictions == 1))  # Binary predictions
    
    def test_hard_voting_binary_classification(self, binary_classification_models, binary_dataset):
        """Test hard voting for binary classification."""
        # Arrange
        ensemble = VotingEnsemble(
            models=binary_classification_models,
            task_type='classification',
            voting='hard',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs=1,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = binary_dataset[0:5]
        
        # Act
        with torch.no_grad():
            proba = ensemble.predict_proba(features)
            predictions = ensemble.predict(features)
        
        # Assert
        assert torch.all((predictions == 0) | (predictions == 1))  # Binary predictions
    
    def test_soft_voting_multiclass(self, multiclass_models, multiclass_dataset):
        """Test soft voting for multi-class classification."""
        # Arrange
        ensemble = VotingEnsemble(
            models=multiclass_models,
            task_type='classification',
            voting='soft',
            device=torch.device('cpu')
        )
        
        # Use the specialized fit_multiclass method
        ensemble.fit_multiclass(
            train_dataset=multiclass_dataset,
            batch_size=32,
            epochs=1,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = multiclass_dataset[0:5]
        
        # Act
        with torch.no_grad():
            proba = ensemble(features)
            predictions = ensemble.predict(features)
        
        # Assert
        assert proba.shape == (5, 3)  # (samples, classes)
        assert torch.all((proba >= 0.0) & (proba <= 1.0))  # Probabilities between 0 and 1
        assert torch.allclose(torch.sum(proba, dim=1), torch.ones(5))  # Sum to 1
        
        assert predictions.shape == (5,)
        assert torch.all((predictions >= 0) & (predictions <= 2))  # Class indices 0, 1, 2
    
    def test_predict_proba_regression_error(self, regression_models, regression_dataset):
        """Test predict_proba raises error for regression."""
        # Arrange
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs=1,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = regression_dataset[0:5]
        
        # Act & Assert
        with pytest.raises(ValueError):
            ensemble.predict_proba(features)
    
    def test_fit_with_validation(self, regression_models, regression_dataset):
        """Test fitting with validation data."""
        # Arrange
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression',
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
            epochs=1,
            verbose=False
        )
        
        # Assert
        assert 'val_loss' in history
        assert len(history['val_loss']) == 3  # One for each model
        for model_val_loss in history['val_loss']:
            assert len(model_val_loss) > 0
    
    def test_evaluate(self, regression_models, regression_dataset):
        """Test evaluate method."""
        # Arrange
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs=1,
            verbose=False
        )
        
        # Act
        metrics = ensemble.evaluate(
            dataset=regression_dataset,
            batch_size=32
        )
        
        # Assert
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
    
    def test_save_and_load(self, regression_models, regression_dataset, tmp_path):
        """Test saving and loading the ensemble."""
        # Arrange
        ensemble = VotingEnsemble(
            models=regression_models,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs=1,
            verbose=False
        )
        
        save_path = tmp_path / "voting_test.pt"
        
        # Act - Save
        ensemble.save(str(save_path))
        
        # Create new models with same architecture for loading
        new_models = [
            RegressionModel(input_dim=5, hidden_dim=8, output_dim=1),
            RegressionModel(input_dim=5, hidden_dim=12, output_dim=1),
            RegressionModel(input_dim=5, hidden_dim=16, output_dim=1)
        ]
        
        # Act - Load
        loaded_ensemble = VotingEnsemble.load(
            str(save_path),
            models=new_models,
            device=torch.device('cpu')
        )
        
        # Assert
        assert loaded_ensemble.task_type == ensemble.task_type
        assert loaded_ensemble.voting == ensemble.voting
        assert len(loaded_ensemble.models) == len(ensemble.models)
        assert torch.allclose(loaded_ensemble.weights, ensemble.weights)
        
        # Test predictions match
        features, _ = regression_dataset[0:5]
        with torch.no_grad():
            original_output = ensemble(features)
            loaded_output = loaded_ensemble(features)
            
        assert torch.allclose(original_output, loaded_output) 