from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session

Base = declarative_base()

class Feature(Base):
    """Feature model for database storage"""
    __tablename__ = 'features'

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    formula = Column(Text, nullable=False)
    type = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Optional metadata
    mean = Column(Float, nullable=True)
    std = Column(Float, nullable=True)
    price_correlation = Column(Float, nullable=True)
    returns_correlation = Column(Float, nullable=True)
    description = Column(Text, nullable=True)

class IFeatureRepository(ABC):
    """Interface for feature repository"""
    
    @abstractmethod
    def save_feature(self, feature: Feature) -> Feature:
        """Save a new feature"""
        pass
    
    @abstractmethod
    def get_feature(self, feature_id: int) -> Optional[Feature]:
        """Get a feature by ID"""
        pass
    
    @abstractmethod
    def get_features_by_symbol(self, symbol: str) -> List[Feature]:
        """Get all features for a symbol"""
        pass
    
    @abstractmethod
    def get_all_features(self) -> List[Feature]:
        """Get all features"""
        pass
    
    @abstractmethod
    def update_feature(self, feature: Feature) -> Feature:
        """Update an existing feature"""
        pass
    
    @abstractmethod
    def delete_feature(self, feature_id: int) -> bool:
        """Delete a feature"""
        pass

class SQLFeatureRepository(IFeatureRepository):
    """SQL implementation of feature repository"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def save_feature(self, feature: Feature) -> Feature:
        self.session.add(feature)
        self.session.commit()
        return feature
    
    def get_feature(self, feature_id: int) -> Optional[Feature]:
        return self.session.query(Feature).filter(Feature.id == feature_id).first()
    
    def get_features_by_symbol(self, symbol: str) -> List[Feature]:
        return self.session.query(Feature).filter(Feature.symbol == symbol).all()
    
    def get_all_features(self) -> List[Feature]:
        return self.session.query(Feature).all()
    
    def update_feature(self, feature: Feature) -> Feature:
        existing = self.get_feature(feature.id)
        if existing:
            for key, value in feature.__dict__.items():
                if not key.startswith('_'):
                    setattr(existing, key, value)
            self.session.commit()
            return existing
        return None
    
    def delete_feature(self, feature_id: int) -> bool:
        feature = self.get_feature(feature_id)
        if feature:
            self.session.delete(feature)
            self.session.commit()
            return True
        return False