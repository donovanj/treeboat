import pytest
import numpy as np

from financial_prediction_system.core.models.base import PredictionModel


class MockModel(PredictionModel):
    """Mock implementation of PredictionModel for testing."""
    
    def __init__(self):
        self.trained = False
        self.saved_path = None
        self.loaded_path = None
    
    def train(self, features, targets, **params):
        """Mock training implementation."""
        self.trained = True
        self.training_features = features
        self.training_targets = targets
        self.training_params = params
        return {"loss": 0.1, "accuracy": 0.9}
    
    def predict(self, features):
        """Mock prediction implementation."""
        return np.zeros(len(features))
    
    def evaluate(self, features, targets):
        """Mock evaluation implementation."""
        return {"loss": 0.2, "accuracy": 0.8}
    
    def save(self, path):
        """Mock save implementation."""
        self.saved_path = path
    
    def load(self, path):
        """Mock load implementation."""
        self.loaded_path = path


class TestPredictionModel:
    """Tests for the PredictionModel interface."""
    
    def test_model_interface_implementation(self):
        """Test that a model implementing the interface works correctly."""
        # Arrange
        model = MockModel()
        features = np.random.rand(10, 5)
        targets = np.random.rand(10)
        
        # Act
        training_result = model.train(features, targets, epochs=10)
        predictions = model.predict(features)
        eval_result = model.evaluate(features, targets)
        model.save("model.pt")
        model.load("model.pt")
        
        # Assert
        assert model.trained is True
        assert np.array_equal(model.training_features, features)
        assert np.array_equal(model.training_targets, targets)
        assert model.training_params == {"epochs": 10}
        assert training_result == {"loss": 0.1, "accuracy": 0.9}
        assert predictions.shape == (10,)
        assert eval_result == {"loss": 0.2, "accuracy": 0.8}
        assert model.saved_path == "model.pt"
        assert model.loaded_path == "model.pt" 