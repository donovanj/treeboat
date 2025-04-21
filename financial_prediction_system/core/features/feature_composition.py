from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class FeatureConfig:
    type: str
    parameters: Dict[str, Any]
    name: Optional[str] = None
    description: Optional[str] = None

class FeatureComposition:
    """Manages a composition of multiple features and their configurations"""
    
    def __init__(
        self,
        name: str,
        features: List[FeatureConfig],
        symbol: str,
        description: Optional[str] = None
    ):
        self.name = name
        self.features = features
        self.symbol = symbol
        self.description = description
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def add_feature(self, feature: FeatureConfig) -> None:
        """Add a new feature to the composition"""
        self.features.append(feature)
        self.updated_at = datetime.utcnow()
    
    def remove_feature(self, index: int) -> None:
        """Remove a feature from the composition"""
        if 0 <= index < len(self.features):
            self.features.pop(index)
            self.updated_at = datetime.utcnow()
    
    def update_feature(self, index: int, new_config: FeatureConfig) -> None:
        """Update an existing feature's configuration"""
        if 0 <= index < len(self.features):
            self.features[index] = new_config
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the composition to a dictionary for storage"""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'description': self.description,
            'features': [
                {
                    'type': f.type,
                    'parameters': f.parameters,
                    'name': f.name,
                    'description': f.description
                }
                for f in self.features
            ],
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureComposition':
        """Create a FeatureComposition instance from a dictionary"""
        features = [
            FeatureConfig(
                type=f['type'],
                parameters=f['parameters'],
                name=f.get('name'),
                description=f.get('description')
            )
            for f in data['features']
        ]
        
        composition = cls(
            name=data['name'],
            features=features,
            symbol=data['symbol'],
            description=data.get('description')
        )
        
        composition.created_at = datetime.fromisoformat(data['created_at'])
        composition.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return composition
    
    def to_json(self) -> str:
        """Convert the composition to a JSON string"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureComposition':
        """Create a FeatureComposition instance from a JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate the feature composition and return any errors"""
        errors = []
        
        if not self.name:
            errors.append("Composition name is required")
            
        if not self.symbol:
            errors.append("Symbol is required")
            
        if not self.features:
            errors.append("At least one feature is required")
        
        for i, feature in enumerate(self.features):
            if not feature.type:
                errors.append(f"Feature {i}: Type is required")
                
            if not isinstance(feature.parameters, dict):
                errors.append(f"Feature {i}: Parameters must be a dictionary")
        
        return errors
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in the composition"""
        return [
            f.name or f"{f.type}_{i}"
            for i, f in enumerate(self.features)
        ]
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get dictionary of feature descriptions"""
        return {
            name: f.description or "No description available"
            for name, f in zip(self.get_feature_names(), self.features)
        }
    
    def merge(self, other: 'FeatureComposition') -> 'FeatureComposition':
        """Merge another composition into this one"""
        merged_features = self.features + other.features
        
        return FeatureComposition(
            name=f"{self.name}_{other.name}",
            features=merged_features,
            symbol=self.symbol,
            description=f"Merged composition of {self.name} and {other.name}"
        )