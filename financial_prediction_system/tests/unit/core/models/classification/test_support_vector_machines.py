import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from financial_prediction_system.core.models.classification.support_vector_machines import SVMModel
from financial_prediction_system.core.models.factory import ModelFactory
from financial_prediction_system.core.models.base import PredictionModel


class TestSVMModel:
    """Tests for the Support Vector Machines model."""
    
    def test_init(self):
        """Test initialization with default parameters."""
        # Arrange & Act
        model = SVMModel()
        
        # Assert
        assert model.model.C == 1.0
        assert model.model.kernel == 'rbf'
        assert model.model.degree == 3
        assert model.model.gamma == 'scale'
        assert model.model.probability is True
        assert model.model.random_state is None
        assert model.model.class_weight is None
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        # Arrange & Act
        model = SVMModel(
            C=0.5,
            kernel='poly',
            degree=4,
            gamma='auto',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        # Assert
        assert model.model.C == 0.5
        assert model.model.kernel == 'poly'
        assert model.model.degree == 4
        assert model.model.gamma == 'auto'
        assert model.model.probability is True
        assert model.model.random_state == 42
        assert model.model.class_weight == 'balanced'
    
    def test_implements_prediction_model_interface(self):
        """Test that SVMModel implements PredictionModel interface."""
        # Arrange & Act
        model = SVMModel()
        
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
        model = SVMModel(random_state=42)
        features = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, size=100)
        
        # Act
        result = model.train(features, targets)
        
        # Assert
        assert result is model  # Should return self
        assert hasattr(model.model, 'support_vectors_')
    
    def test_train_torch_input(self):
        """Test training with torch tensors."""
        # Arrange
        model = SVMModel(random_state=42)
        features = torch.rand(100, 5)
        targets = torch.randint(0, 2, size=(100,))
        
        # Act
        result = model.train(features, targets)
        
        # Assert
        assert result is model  # Should return self
        assert hasattr(model.model, 'support_vectors_')
    
    def test_predict_numpy_input(self):
        """Test prediction with numpy arrays."""
        # Arrange
        model = SVMModel(random_state=42)
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
        model = SVMModel(random_state=42)
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
        model = SVMModel(random_state=42, probability=True)
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
    
    def test_predict_proba_error_when_probability_false(self):
        """Test error when probability is False."""
        # This test requires mocking the SVC model to avoid training
        # Arrange
        model = SVMModel()
        model.model = MagicMock()
        model.model.predict_proba = None  # Simulate probability=False
        features_test = np.random.rand(10, 5)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Model was not trained with probability=True"):
            model.predict_proba(features_test)
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Arrange
        model = SVMModel(random_state=42)
        features = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, size=100)
        model.train(features, targets)
        
        # Act
        metrics = model.evaluate(features, targets)
        
        # Assert
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'n_support_vectors' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    @patch('joblib.dump')
    def test_save(self, mock_dump):
        """Test model saving."""
        # Arrange
        model = SVMModel()
        save_path = "models/svm_model.pkl"
        
        # Act
        model.save(save_path)
        
        # Assert
        mock_dump.assert_called_once_with(model.model, save_path)
    
    @patch('joblib.load')
    def test_load(self, mock_load):
        """Test model loading."""
        # Arrange
        model = SVMModel()
        load_path = "models/svm_model.pkl"
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Act
        result = model.load(load_path)
        
        # Assert
        mock_load.assert_called_once_with(load_path)
        assert model.model is mock_model
        assert result is model  # Should return self


class TestSVMWithFactory:
    """Tests for the SVM model with factory."""
    
    def test_factory_registration(self):
        """Test that the model is registered with the factory."""
        assert "svm" in ModelFactory.get_available_models()
    
    def test_factory_create_svm(self):
        """Test creating the model through the factory."""
        # Act
        model = ModelFactory.create("svm", C=0.5, kernel='linear', random_state=42)
        
        # Assert
        assert isinstance(model, SVMModel)
        assert model.model.C == 0.5
        assert model.model.kernel == 'linear'
        assert model.model.random_state == 42
        
    