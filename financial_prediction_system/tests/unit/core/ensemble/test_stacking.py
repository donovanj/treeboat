import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from financial_prediction_system.core.ensemble.stacking import StackingEnsemble


class BaseModel(nn.Module):
    """Simple base model for testing Stacking"""
    def __init__(self, input_dim=5, hidden_dim=10, output_dim=1):
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class MetaModel(nn.Module):
    """Simple meta-model for testing Stacking"""
    def __init__(self, input_dim, hidden_dim=8, output_dim=1):
        super(MetaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class TestStackingEnsemble:
    """Tests for the Stacking ensemble implementation."""
    
    @pytest.fixture
    def regression_models(self):
        """Create base models for regression."""
        models = [
            BaseModel(input_dim=5, hidden_dim=8, output_dim=1),
            BaseModel(input_dim=5, hidden_dim=12, output_dim=1),
            BaseModel(input_dim=5, hidden_dim=16, output_dim=1)
        ]
        return models
    
    @pytest.fixture
    def classification_models(self):
        """Create base models for binary classification."""
        models = [
            BaseModel(input_dim=5, hidden_dim=8, output_dim=1),
            BaseModel(input_dim=5, hidden_dim=12, output_dim=1),
            BaseModel(input_dim=5, hidden_dim=16, output_dim=1)
        ]
        return models
    
    @pytest.fixture
    def regression_meta_model(self):
        """Create meta-model for regression."""
        # Input dim = sum of base model outputs (3 models with 1 output each)
        return MetaModel(input_dim=3, hidden_dim=8, output_dim=1)
    
    @pytest.fixture
    def classification_meta_model(self):
        """Create meta-model for binary classification."""
        # Input dim = sum of base model outputs (3 models with 1 output each)
        return MetaModel(input_dim=3, hidden_dim=8, output_dim=1)
    
    @pytest.fixture
    def regression_meta_model_with_features(self):
        """Create meta-model for regression with original features."""
        # Input dim = sum of base model outputs + original features
        return MetaModel(input_dim=3+5, hidden_dim=8, output_dim=1)
    
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
    
    def test_init(self, regression_models, regression_meta_model):
        """Test initialization with default parameters."""
        # Arrange & Act
        ensemble = StackingEnsemble(
            base_models=regression_models,
            meta_model=regression_meta_model,
            task_type='regression'
        )
        
        # Assert
        assert ensemble.task_type == 'regression'
        assert ensemble.use_features is False
        assert len(ensemble.base_models) == 3
        assert isinstance(ensemble.meta_model, MetaModel)
        assert isinstance(ensemble.device, torch.device)
    
    def test_init_custom_params(self, classification_models, classification_meta_model):
        """Test initialization with custom parameters."""
        # Arrange & Act
        ensemble = StackingEnsemble(
            base_models=classification_models,
            meta_model=classification_meta_model,
            task_type='classification',
            use_features=True,
            device=torch.device('cpu')
        )
        
        # Assert
        assert ensemble.task_type == 'classification'
        assert ensemble.use_features is True
        assert ensemble.device == torch.device('cpu')
    
    def test_fit_regression(self, regression_models, regression_meta_model, regression_dataset):
        """Test fitting Stacking on regression data."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=regression_models,
            meta_model=regression_meta_model,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_base=2,
            epochs_meta=2,
            n_folds=2,
            verbose=False
        )
        
        # Assert
        assert isinstance(history, dict)
        assert 'base_models_train_loss' in history
        assert 'meta_model_train_loss' in history
        assert len(history['base_models_train_loss']) == 3  # One for each base model
        assert len(history['meta_model_train_loss']) > 0
    
    def test_fit_classification(self, classification_models, classification_meta_model, binary_dataset):
        """Test fitting Stacking on classification data."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=classification_models,
            meta_model=classification_meta_model,
            task_type='classification',
            device=torch.device('cpu')
        )
        
        # Act
        history = ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_base=2,
            epochs_meta=2,
            n_folds=2,
            verbose=False
        )
        
        # Assert
        assert isinstance(history, dict)
        assert 'base_models_train_loss' in history
        assert 'meta_model_train_loss' in history
    
    def test_forward_regression(self, regression_models, regression_meta_model, regression_dataset):
        """Test forward pass for regression."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=regression_models,
            meta_model=regression_meta_model,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        # Train with very few epochs just for testing
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_base=1,
            epochs_meta=1,
            n_folds=2,
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
    
    def test_forward_classification(self, classification_models, classification_meta_model, binary_dataset):
        """Test forward pass for classification."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=classification_models,
            meta_model=classification_meta_model,
            task_type='classification',
            device=torch.device('cpu')
        )
        
        # Train with very few epochs just for testing
        ensemble.fit(
            train_dataset=binary_dataset,
            batch_size=32,
            epochs_base=1,
            epochs_meta=1,
            n_folds=2,
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
    
    def test_fit_with_validation(self, regression_models, regression_meta_model, regression_dataset):
        """Test fitting with validation data."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=regression_models,
            meta_model=regression_meta_model,
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
            epochs_base=1,
            epochs_meta=1,
            n_folds=2,
            verbose=False
        )
        
        # Assert
        assert 'base_models_val_loss' in history
        assert 'meta_model_val_loss' in history
        assert len(history['base_models_val_loss']) == 3  # One for each base model
        assert len(history['meta_model_val_loss']) > 0
    
    def test_with_original_features(self, regression_models, regression_meta_model_with_features, regression_dataset):
        """Test stacking with original features included."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=regression_models,
            meta_model=regression_meta_model_with_features,
            task_type='regression',
            use_features=True,
            device=torch.device('cpu')
        )
        
        # Act - Train with minimal epochs just for testing
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_base=1,
            epochs_meta=1,
            n_folds=2,
            verbose=False
        )
        
        # Sample features from the dataset
        features, _ = regression_dataset[0:5]
        
        # Act - Forward pass
        with torch.no_grad():
            output = ensemble(features)
        
        # Assert
        assert output.shape == (5, 1)
        assert output.dtype == torch.float32
    
    def test_save_and_load(self, regression_models, regression_meta_model, regression_dataset, tmp_path):
        """Test saving and loading the ensemble."""
        # Arrange
        ensemble = StackingEnsemble(
            base_models=regression_models,
            meta_model=regression_meta_model,
            task_type='regression',
            device=torch.device('cpu')
        )
        
        # Train with minimal epochs just for testing
        ensemble.fit(
            train_dataset=regression_dataset,
            batch_size=32,
            epochs_base=1,
            epochs_meta=1,
            n_folds=2,
            verbose=False
        )
        
        save_path = tmp_path / "stacking_test.pt"
        
        # Act - Save
        ensemble.save(str(save_path))
        
        # Create new models with same architecture for loading
        new_base_models = [
            BaseModel(input_dim=5, hidden_dim=8, output_dim=1),
            BaseModel(input_dim=5, hidden_dim=12, output_dim=1),
            BaseModel(input_dim=5, hidden_dim=16, output_dim=1)
        ]
        new_meta_model = MetaModel(input_dim=3, hidden_dim=8, output_dim=1)
        
        # Act - Load
        loaded_ensemble = StackingEnsemble.load(
            str(save_path),
            base_models=new_base_models,
            meta_model=new_meta_model,
            device=torch.device('cpu')
        )
        
        # Assert
        assert loaded_ensemble.task_type == ensemble.task_type
        assert loaded_ensemble.use_features == ensemble.use_features
        assert len(loaded_ensemble.base_models) == len(ensemble.base_models)
        
        # Test predictions match
        features, _ = regression_dataset[0:5]
        with torch.no_grad():
            original_output = ensemble(features)
            loaded_output = loaded_ensemble(features)
            
        assert torch.allclose(original_output, loaded_output, rtol=1e-5, atol=1e-5) 