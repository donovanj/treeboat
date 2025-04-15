from abc import ABC, abstractmethod
from typing import List, Any, Dict


class TrainingObserver(ABC):
    """Observer interface for training events."""
    
    @abstractmethod
    def update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle training event.
        
        Args:
            event_type: Type of event that occurred
            data: Data associated with the event
        """
        pass


class ModelTrainer:
    """Subject that notifies observers about training events."""
    
    def __init__(self):
        self.observers: List[TrainingObserver] = []
        
    def add_observer(self, observer: TrainingObserver) -> None:
        """Add an observer to the notification list.
        
        Args:
            observer: The observer to add
        """
        self.observers.append(observer)
        
    def remove_observer(self, observer: TrainingObserver) -> None:
        """Remove an observer from the notification list.
        
        Args:
            observer: The observer to remove
        """
        self.observers.remove(observer)
        
    def notify_observers(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify all observers of an event.
        
        Args:
            event_type: Type of event that occurred
            data: Data associated with the event
        """
        for observer in self.observers:
            observer.update(event_type, data) 