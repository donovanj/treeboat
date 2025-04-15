import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from financial_prediction_system.core.models.classification.gradient_boosting import GradientBoostingModel
from financial_prediction_system.core.models.factory import ModelFactory
from financial_prediction_system.core.models.base import PredictionModel


class TestGradientBoostingModel:
    """Tests for the Gradient Boosting model."""
    
    def test_init(self):
        """Test initialization with default parameters."""
        # Arrange & Act
        model = GradientBoostingModel()
        
        # Assert
        assert model.model.n_estimators == 100
        assert model.model.learning_rate == 0.1
        assert model.model.max_depth == 3
        assert model.model.min_samples_split == 2
        assert model.model.min_samples_leaf == 1
        assert model.model.subsample == 1.0
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        # Arrange & Act
        model = GradientBoostingModel(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        # Assert
        assert model.model.n_estimators == 200
        assert model.model.learning_rate == 0.05
        assert model.model.max_depth == 5
        assert model.model.min_samples_split == 5
        assert model.model.min_samples_leaf == 2
        assert model.model.subsample == 0.8
        assert model.model.random_state == 42
    
    def test_implements_prediction_model_interface(self):
        """Test that GradientBoostingModel implements PredictionModel interface."""
        # Arrange & Act
        model = GradientBoostingModel()
        
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
        model = GradientBoostingModel(random_state=42)
        features = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, size=100)
        
        # Act
        result = model.train(features, targets)
        
        # Assert
        assert result is model  # Should return self
        assert hasattr(model.model, 'feature_importances_')
    
    def test_train_torch_input(self):
        """Test training with torch tensors."""
        # Arrange
        model = GradientBoostingModel(random_state=42)
        features = torch.rand(100, 5)
        targets = torch.randint(0, 2, size=(100,))
        
        # Act
        result = model.train(features, targets)
        
        # Assert
        assert result is model  # Should return self
        assert hasattr(model.model, 'feature_importances_')
    
    def test_predict_numpy_input(self):
        """Test prediction with numpy arrays."""
        # Arrange
        model = GradientBoostingModel(random_state=42)
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
        model = GradientBoostingModel(random_state=42)
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
        model = GradientBoostingModel(random_state=42)
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
    
    def test_staged_predict_proba(self):
        """Test staged probability prediction."""
        # Arrange
        model = GradientBoostingModel(n_estimators=5, random_state=42)
        features_train = np.random.rand(100, 5)
        targets_train = np.random.randint(0, 2, size=100)
        model.train(features_train, targets_train)
        
        features_test = np.random.rand(10, 5)
        
        # Act
        staged_probs = model.staged_predict_proba(features_test)
        
        # Assert
        assert isinstance(staged_probs, list)
        assert len(staged_probs) == 5  # One for each estimator
        for probs in staged_probs:
            assert isinstance(probs, np.ndarray)
            assert probs.shape == (10, 2)
            assert np.all((probs >= 0) & (probs <= 1))
            assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_evaluate(self):
        """Test model evaluation."""
        # Arrange
        model = GradientBoostingModel(random_state=42)
        features = np.random.rand(100, 5)
        targets = np.random.randint(0, 2, size=100)
        model.train(features, targets)
        
        # Act
        metrics = model.evaluate(features, targets)
        
        # Assert
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'feature_importances' in metrics
        assert 'train_score' in metrics
        assert isinstance(metrics['accuracy'], float)
        assert 0 <= metrics['accuracy'] <= 1
    
    @patch('joblib.dump')
    def test_save(self, mock_dump):
        """Test model saving."""
        # Arrange
        model = GradientBoostingModel()
        save_path = "models/gb_model.pkl"
        
        # Act
        model.save(save_path)
        
        # Assert
        mock_dump.assert_called_once_with(model.model, save_path)
    
    @patch('joblib.load')
    def test_load(self, mock_load):
        """Test model loading."""
        # Arrange
        model = GradientBoostingModel()
        load_path = "models/gb_model.pkl"
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        
        # Act
        result = model.load(load_path)
        
        # Assert
        mock_load.assert_called_once_with(load_path)
        assert model.model is mock_model
        assert result is model  # Should return self


class TestGradientBoostingWithFactory:
    """Tests for the Gradient Boosting model with factory."""
    
    def test_factory_registration(self):
        """Test that the model is registered with the factory."""
        assert "gradient_boosting" in ModelFactory.get_available_models()
    
    def test_factory_create_gradient_boosting(self):
        """Test creating the model through the factory."""
        # Act
        model = ModelFactory.create("gradient_boosting", n_estimators=150, random_state=42)
        
        # Assert
        assert isinstance(model, GradientBoostingModel)
        assert model.model.n_estimators == 150
        assert model.model.random_state == 42
        
    