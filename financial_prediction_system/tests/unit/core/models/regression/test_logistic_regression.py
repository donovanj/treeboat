import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from financial_prediction_system.core.models.regression.logistic_regression import LogisticRegressionModel
from financial_prediction_system.core.models.factory import ModelFactory
from financial_prediction_system.core.models.base import PredictionModel


class TestLogisticRegressionModel:
    """Tests for the Logistic Regression model."""
    
    def test_init(self):
        """Test initialization with default parameters."""
        # Arrange & Act
        model = LogisticRegressionModel()
        
        # Assert
        assert model.model.C == 1.0
        assert model.model.penalty == 'l2'
        assert model.model.solver == 'lbfgs'
        assert model.model.max_iter == 100
        assert model.model.multi_class == 'auto'
        assert model.model.random_state is None
        assert model.model.class_weight is None
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        # Arrange & Act
        model = LogisticRegressionModel(
            C=0.5,
            penalty='l1',
            solver='liblinear',
            max_iter=200,
            multi_class='ovr',
            random_state=42,
            class_weight='balanced'
        )
        
        # Assert
        assert model.model.C == 0.5
        assert model.model.penalty == 'l1'
        assert model.model.solver == 'liblinear'
        assert model.model.max_iter == 200
        assert model.model.multi_class == 'ovr'
        assert model.model.random_state == 42
        assert model.model.class_weight == 'balanced'
    
    def test_implements_prediction_model_interface(self):
        """Test that LogisticRegressionModel implements PredictionModel interface."""
        # Arrange & Act
        model = LogisticRegressionModel()
        
        # Assert
        assert isinstance(model, PredictionModel)
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'evaluate')
        assert hasattr(model, 'save')
        assert hasattr(model, 'load')
    
    def test_train_numpy_input(self):
        """Test training with numpy arrays."""
        # Arrange
        model = LogisticRegressionModel(random_state=42)
        features = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, size=100)
        
        # Act
        result = model.train(features, targets)
        
        # Assert
        assert result is model  # Should return self
        assert hasattr(model.model, 'coef_')
        assert hasattr(model.model, 'intercept_')
    
    def test_train_torch_input(self):
        """Test training with torch tensors."""
        # Arrange
        model = LogisticRegressionModel(random_state=42)
        features = torch.rand(100, 5)
        targets = torch.randint(0, 2, size=(100,))
        
        # Act
        result = model.train(features, targets)
        
        # Assert
        assert result is model  # Should return self
        assert hasattr(model.model, 'coef_')
        assert hasattr(model.model, 'intercept_')
    
    def test_predict_numpy_input(self):
        """Test prediction with numpy arrays."""
        # Arrange
        model = LogisticRegressionModel(random_state=42)
        features_train = np.random.rand(100, 5)
        targets_train = np.random.randint(0, 2, size=100)
        model.train(features_train, targets_train)
        
        features_test = np.random.rand(10, 5)
        
        # Act
        predictions = model.predict(features_test)
        
        # Assert
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (10,)
        assert np.all((predictions == 0) | (predictions == 1))
    
    def test_predict_torch_input(self):
        """Test prediction with torch tensors."""
        # Arrange
        model = LogisticRegressionModel(random_state=42)
        features_train = np.random.rand(100, 5)
        targets_train = np.random.randint(0, 2, size=100)
        model.train(features_train, targets_train)
        
        features_test = torch.rand(10, 5)
        
        # Act
        predictions = model.predict(features_test)
        
        # Assert
        assert isinstance(predictions, torch.Tensor)
        assert predictions.shape == (10,)
        assert torch.all((predictions == 0) | (predictions == 1))
    
    def test_predict_proba(self):
        """Test probability prediction."""
        # Arrange
        model = LogisticRegressionModel(random_state=42)
        features_train = np.random.rand(100, 5)
        targets_train = np.random.randint(0, 2, size=100)
        model.train(features_train, targets_train)
        
        features_test = np.random.rand(10, 5)
        
        # Act
        probabilities = model.predict_proba(features_test)
        
        # Assert
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (10, 2)  # Binary classification
        assert np.all((probabilities >= 0) & (probabilities <= 1))
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Arrange
        model = LogisticRegressionModel(random_state=42)
        features = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, size=100)
        model.train(features, targets)
        
        # Act
        metrics = model.evaluate(features, targets)
        
        # Assert
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'coefficients' in metrics
        assert 'intercept' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    @patch('joblib.dump')
    def test_save(self, mock_dump):
        """Test model saving."""
        # Arrange
        model = LogisticRegressionModel()
        save_path = "models/lr_model.pkl"
        
        # Act
        model.save(save_path)
        
        # Assert
        mock_dump.assert_called_once_with(model.model, save_path)
    
    @patch('joblib.load')
    def test_load(self, mock_load):
        """Test model loading."""
        # Arrange
        model = LogisticRegressionModel()
        load_path = "models/lr_model.pkl"
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Act
        result = model.load(load_path)
        
        # Assert
        mock_load.assert_called_once_with(load_path)
        assert model.model is mock_model
        assert result is model  # Should return self


class TestLogisticRegressionWithFactory:
    """Tests for the Logistic Regression model with factory."""
    
    def test_factory_registration(self):
        """Test that the model is registered with the factory."""
        assert "logistic_regression" in ModelFactory.get_available_models()
    
    def test_factory_create_logistic_regression(self):
        """Test creating the model through the factory."""
        # Act
        model = ModelFactory.create("logistic_regression", C=0.5, solver='liblinear', random_state=42)
        
        # Assert
        assert isinstance(model, LogisticRegressionModel)
        assert model.model.C == 0.5
        assert model.model.solver == 'liblinear'
        assert model.model.random_state == 42