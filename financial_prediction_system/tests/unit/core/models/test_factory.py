import pytest

from financial_prediction_system.core.models.base import PredictionModel
from financial_prediction_system.core.models.factory import ModelFactory


class MockLSTMModel(PredictionModel):
    """Mock LSTM model for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
    
    def train(self, features, targets, **params):
        pass
    
    def predict(self, features):
        pass
    
    def evaluate(self, features, targets):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class MockRandomForestModel(PredictionModel):
    """Mock Random Forest model for testing."""
    
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
    
    def train(self, features, targets, **params):
        pass
    
    def predict(self, features):
        pass
    
    def evaluate(self, features, targets):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class TestModelFactory:
    """Tests for the ModelFactory."""
    
    def setup_method(self):
        """Setup for each test - clear and register test models."""
        # Clear the registry
        ModelFactory._models = {}
        
        # Register test models
        ModelFactory.register("lstm", MockLSTMModel)
        ModelFactory.register("random_forest", MockRandomForestModel)
    
    def test_create_model_returns_correct_type(self):
        """Test that the factory creates the correct model type."""
        # Act
        lstm_model = ModelFactory.create_model("lstm")
        rf_model = ModelFactory.create_model("random_forest")
        
        # Assert
        assert isinstance(lstm_model, MockLSTMModel)
        assert isinstance(rf_model, MockRandomForestModel)
    
    def test_create_model_passes_parameters(self):
        """Test that the factory passes parameters to the model constructor."""
        # Act
        lstm_model = ModelFactory.create_model("lstm", input_dim=5, hidden_dim=10)
        rf_model = ModelFactory.create_model("random_forest", n_estimators=200)
        
        # Assert
        assert lstm_model.input_dim == 5
        assert lstm_model.hidden_dim == 10
        assert rf_model.n_estimators == 200
    
    def test_create_model_with_unknown_type_raises_error(self):
        """Test that the factory raises an error for unknown model types."""
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported model type: unknown"):
            ModelFactory.create_model("unknown")
    
    def test_register_adds_model_to_registry(self):
        """Test that registering a model adds it to the registry."""
        # Arrange
        class NewModel(PredictionModel):
            def train(self, features, targets, **params): pass
            def predict(self, features): pass
            def evaluate(self, features, targets): pass
            def save(self, path): pass
            def load(self, path): pass
        
        # Act
        ModelFactory.register("new_model", NewModel)
        
        # Assert
        assert "new_model" in ModelFactory._models
        assert ModelFactory._models["new_model"] == NewModel 