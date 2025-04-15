from typing import Dict, Type, Any, List

from .base import PredictionModel


class ModelFactory:
    """Factory for creating prediction models."""
    
    _models: Dict[str, Type[PredictionModel]] = {}
    _target_builders: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, model_type: str, model_class: Type[PredictionModel]) -> None:
        """Register a model class with the factory.
        
        Args:
            model_type: String identifier for the model
            model_class: The model class to register
        """
        cls._models[model_type] = model_class
    
    @classmethod
    def register_target_builder(cls, builder_type: str, builder_function: Any) -> None:
        """Register a target builder function with the factory.
        
        Args:
            builder_type: String identifier for the target builder
            builder_function: The target builder function to register
        """
        cls._target_builders[builder_type] = builder_function
    
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
    
    @classmethod
    def create(cls, model_type: str, **params: Any) -> PredictionModel:
        """Alias for create_model.
        
        Args:
            model_type: The type of model to create
            **params: Parameters to pass to the model constructor
            
        Returns:
            An instance of the requested model
            
        Raises:
            ValueError: If the model type is not registered
        """
        return cls.create_model(model_type, **params)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get a list of all registered model types.
        
        Returns:
            List of registered model type names
        """
        return list(cls._models.keys())
    
    @classmethod
    def get_model_documentation(cls) -> Dict[str, Dict[str, str]]:
        """Get documentation for all registered models.
        
        Returns:
            Dictionary mapping model names to their documentation
        """
        docs = {}
        for model_name, model_class in cls._models.items():
            docs[model_name] = {
                "description": model_class.__doc__.strip().split("\n\n")[0] if model_class.__doc__ else "No description available",
                "type": "classification" if "Classifier" in model_class.__name__ or model_name in ["random_forest", "svm", "gradient_boosting"] else "regression"
            }
        return docs
    
    @classmethod
    def create_target_builder(cls, builder_type: str, builder: Any, **params: Any) -> Any:
        """Create a target using the registered builder function.
        
        Args:
            builder_type: The type of target builder to use
            builder: The target builder instance
            **params: Parameters to pass to the target builder function
            
        Returns:
            The target builder with targets added
            
        Raises:
            ValueError: If the builder type is not registered
        """
        if builder_type not in cls._target_builders:
            raise ValueError(f"Unsupported target builder type: {builder_type}")
        
        # Call the registered target builder function with the builder and params
        return cls._target_builders[builder_type](builder, **params) 