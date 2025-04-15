from typing import Dict, Type, Any

from .base import PredictionModel


class ModelFactory:
    """Factory for creating prediction models."""
    
    _models: Dict[str, Type[PredictionModel]] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[PredictionModel]) -> None:
        """Register a model class with the factory.
        
        Args:
            model_type: String identifier for the model
            model_class: The model class to register
        """
        cls._models[model_type] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, **params: Any) -> PredictionModel:
        """Create a model of the specified type.
        
        Args:
            model_type: The type of model to create
            **params: Parameters to pass to the model constructor
            
        Returns:
            An instance of the requested model
            
        Raises:
            ValueError: If the model type is not registered
        """
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return cls._models[model_type](**params) 