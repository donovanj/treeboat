import pytest
from unittest.mock import MagicMock

from financial_prediction_system.core.models.training import TrainingObserver, ModelTrainer


class MockTrainingObserver(TrainingObserver):
    """Mock observer for testing."""
    
    def __init__(self):
        self.events = []
    
    def update(self, event_type, data):
        """Record event and data."""
        self.events.append((event_type, data))


class TestModelTrainer:
    """Tests for the ModelTrainer class."""
    
    def test_add_observer(self):
        """Test adding an observer."""
        # Arrange
        trainer = ModelTrainer()
        observer = MockTrainingObserver()
        
        # Act
        trainer.add_observer(observer)
        
        # Assert
        assert observer in trainer.observers
        assert len(trainer.observers) == 1
    
    def test_remove_observer(self):
        """Test removing an observer."""
        # Arrange
        trainer = ModelTrainer()
        observer = MockTrainingObserver()
        trainer.add_observer(observer)
        
        # Act
        trainer.remove_observer(observer)
        
        # Assert
        assert observer not in trainer.observers
        assert len(trainer.observers) == 0
    
    def test_notify_observers(self):
        """Test that observers are notified."""
        # Arrange
        trainer = ModelTrainer()
        observer1 = MockTrainingObserver()
        observer2 = MockTrainingObserver()
        trainer.add_observer(observer1)
        trainer.add_observer(observer2)
        
        event_type = "epoch_complete"
        data = {"epoch": 1, "loss": 0.5}
        
        # Act
        trainer.notify_observers(event_type, data)
        
        # Assert
        assert len(observer1.events) == 1
        assert observer1.events[0] == (event_type, data)
        assert len(observer2.events) == 1
        assert observer2.events[0] == (event_type, data)
    
    def test_notify_observers_with_empty_list(self):
        """Test that notify doesn't crash with no observers."""
        # Arrange
        trainer = ModelTrainer()
        
        # Act & Assert - should not raise
        trainer.notify_observers("event", {"data": "value"})
    
    def test_observer_update_called(self):
        """Test that observer's update method is called."""
        # Arrange
        trainer = ModelTrainer()
        mock_observer = MagicMock()
        trainer.add_observer(mock_observer)
        
        event_type = "training_complete"
        data = {"accuracy": 0.95}
        
        # Act
        trainer.notify_observers(event_type, data)
        
        # Assert
        mock_observer.update.assert_called_once_with(event_type, data) 