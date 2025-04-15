"""
Database model store module for storing and retrieving ML models and their metadata.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean, Text, func
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from financial_prediction_system.infrastructure.database.connection import Base

class MLModelResult(Base):
    """SQLAlchemy model for storing ML model results"""
    __tablename__ = "ml_model_results"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=False)
    model_type = Column(String, nullable=False)
    target_type = Column(String, nullable=False)
    feature_sets = Column(JSON, nullable=False)  # Store as JSON list
    hyperparameters = Column(JSON, nullable=True)  # Store as JSON
    train_start_date = Column(DateTime, nullable=False)
    train_end_date = Column(DateTime, nullable=False)
    test_start_date = Column(DateTime, nullable=True)
    test_end_date = Column(DateTime, nullable=True)
    
    # Model performance metrics
    test_mse = Column(Float, nullable=True)
    test_r2 = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    cv_mean_mse = Column(Float, nullable=True)
    cv_std_mse = Column(Float, nullable=True)
    cv_mean_r2 = Column(Float, nullable=True)
    cv_std_r2 = Column(Float, nullable=True)
    cv_mean_mae = Column(Float, nullable=True)
    cv_std_mae = Column(Float, nullable=True)
    
    # Model binary storage
    model_path = Column(String, nullable=True)  # Path to stored model file
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    feature_importance = relationship("FeatureImportance", back_populates="model_result", cascade="all, delete-orphan")
    backtests = relationship("BacktestResult", back_populates="model_result", cascade="all, delete-orphan")
    ensemble_memberships = relationship("EnsembleModelMapping", back_populates="model", cascade="all, delete-orphan")

class FeatureImportance(Base):
    """SQLAlchemy model for storing feature importance data"""
    __tablename__ = "ml_feature_importance"
    
    id = Column(Integer, primary_key=True, index=True)
    model_result_id = Column(Integer, ForeignKey("ml_model_results.id", ondelete="CASCADE"), nullable=False)
    feature = Column(String, nullable=False)
    importance = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    model_result = relationship("MLModelResult", back_populates="feature_importance")

class EnsembleModel(Base):
    """SQLAlchemy model for storing ensemble models"""
    __tablename__ = "ml_ensemble_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    ensemble_method = Column(String, nullable=False)
    hyperparameters = Column(JSON, nullable=True)  # Store as JSON
    model_path = Column(String, nullable=True)  # Path to stored ensemble model file
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    model_mappings = relationship("EnsembleModelMapping", back_populates="ensemble", cascade="all, delete-orphan")
    backtests = relationship("BacktestResult", back_populates="ensemble", cascade="all, delete-orphan")

class EnsembleModelMapping(Base):
    """SQLAlchemy model for mapping models to ensembles"""
    __tablename__ = "ml_ensemble_model_mappings"
    
    ensemble_id = Column(Integer, ForeignKey("ml_ensemble_models.id", ondelete="CASCADE"), primary_key=True)
    model_id = Column(Integer, ForeignKey("ml_model_results.id", ondelete="CASCADE"), primary_key=True)
    weight = Column(Float, nullable=True)  # Optional weight for weighted ensembles
    
    # Relationships
    ensemble = relationship("EnsembleModel", back_populates="model_mappings")
    model = relationship("MLModelResult", back_populates="ensemble_memberships")

class BacktestResult(Base):
    """SQLAlchemy model for storing backtest results"""
    __tablename__ = "ml_backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_model_results.id", ondelete="SET NULL"), nullable=True)
    ensemble_id = Column(Integer, ForeignKey("ml_ensemble_models.id", ondelete="SET NULL"), nullable=True)
    symbol = Column(String, index=True, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    performance_metrics = Column(JSON, nullable=False)  # Store as JSON
    selected_features = Column(JSON, nullable=True)  # Store as JSON list
    parameters = Column(JSON, nullable=True)  # Store backtest parameters as JSON
    trades = Column(JSON, nullable=True)  # Store trade history as JSON
    equity_curve = Column(JSON, nullable=True)  # Store equity curve data as JSON
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    model_result = relationship("MLModelResult", back_populates="backtests")
    ensemble = relationship("EnsembleModel", back_populates="backtests")
    
    # Ensure we have either model_id or ensemble_id
    __table_args__ = (
        # Add a check constraint to ensure either model_id or ensemble_id is not null
        # But not both can be non-null at the same time
        # This constraint is defined in the database as check_model_or_ensemble
        {}
    )

class ModelRepository:
    """Repository class for ML model CRUD operations"""
    
    def __init__(self, db_session):
        """Initialize with a database session"""
        self.db = db_session
    
    def create_model(self, model_data: Dict[str, Any]) -> MLModelResult:
        """Create a new ML model record"""
        model = MLModelResult(**model_data)
        self.db.add(model)
        self.db.commit()
        self.db.refresh(model)
        return model
    
    def get_model(self, model_id: int) -> Optional[MLModelResult]:
        """Get a model by ID"""
        return self.db.query(MLModelResult).filter(MLModelResult.id == model_id).first()
    
    def list_models(self, filters: Dict[str, Any] = None, skip: int = 0, limit: int = 100) -> List[MLModelResult]:
        """List models with optional filtering"""
        query = self.db.query(MLModelResult)
        
        if filters:
            if "symbol" in filters and filters["symbol"]:
                query = query.filter(MLModelResult.symbol == filters["symbol"])
            
            if "model_type" in filters and filters["model_type"]:
                query = query.filter(MLModelResult.model_type == filters["model_type"])
                
            if "target_type" in filters and filters["target_type"]:
                query = query.filter(MLModelResult.target_type == filters["target_type"])
            
            if "created_after" in filters and filters["created_after"]:
                query = query.filter(MLModelResult.created_at >= filters["created_after"])
                
            if "min_r2" in filters and filters["min_r2"] is not None:
                query = query.filter(MLModelResult.test_r2 >= filters["min_r2"])
        
        # Order by newest first
        query = query.order_by(MLModelResult.created_at.desc())
        
        # Apply pagination
        return query.offset(skip).limit(limit).all()
    
    def count_models(self, filters: Dict[str, Any] = None) -> int:
        """Count total models with optional filtering"""
        query = self.db.query(func.count(MLModelResult.id))
        
        if filters:
            if "symbol" in filters and filters["symbol"]:
                query = query.filter(MLModelResult.symbol == filters["symbol"])
            
            if "model_type" in filters and filters["model_type"]:
                query = query.filter(MLModelResult.model_type == filters["model_type"])
                
            if "target_type" in filters and filters["target_type"]:
                query = query.filter(MLModelResult.target_type == filters["target_type"])
                
            if "created_after" in filters and filters["created_after"]:
                query = query.filter(MLModelResult.created_at >= filters["created_after"])
                
            if "min_r2" in filters and filters["min_r2"] is not None:
                query = query.filter(MLModelResult.test_r2 >= filters["min_r2"])
        
        return query.scalar()
    
    def update_model(self, model_id: int, model_data: Dict[str, Any]) -> Optional[MLModelResult]:
        """Update a model by ID"""
        model = self.get_model(model_id)
        if not model:
            return None
        
        # Update attributes
        for key, value in model_data.items():
            if hasattr(model, key):
                setattr(model, key, value)
        
        self.db.commit()
        self.db.refresh(model)
        return model
    
    def delete_model(self, model_id: int) -> bool:
        """Delete a model by ID"""
        model = self.get_model(model_id)
        if not model:
            return False
        
        self.db.delete(model)
        self.db.commit()
        return True
    
    def add_feature_importance(self, model_id: int, feature_importance_data: List[Dict[str, Any]]) -> List[FeatureImportance]:
        """Add feature importance data for a model"""
        model = self.get_model(model_id)
        if not model:
            return []
        
        feature_importances = []
        for data in feature_importance_data:
            data["model_result_id"] = model_id
            feature_importance = FeatureImportance(**data)
            self.db.add(feature_importance)
            feature_importances.append(feature_importance)
        
        self.db.commit()
        return feature_importances
    
    def create_ensemble(self, ensemble_data: Dict[str, Any], model_ids: List[int], weights: Optional[List[float]] = None) -> EnsembleModel:
        """Create a new ensemble model"""
        # Create ensemble record
        ensemble = EnsembleModel(**{k: v for k, v in ensemble_data.items() if k != "model_ids"})
        self.db.add(ensemble)
        self.db.flush()  # Flush to get the ensemble ID
        
        # Create mappings for each model
        for i, model_id in enumerate(model_ids):
            weight = weights[i] if weights and i < len(weights) else None
            mapping = EnsembleModelMapping(
                ensemble_id=ensemble.id,
                model_id=model_id,
                weight=weight
            )
            self.db.add(mapping)
        
        self.db.commit()
        self.db.refresh(ensemble)
        return ensemble
    
    def get_ensemble(self, ensemble_id: int) -> Optional[EnsembleModel]:
        """Get an ensemble by ID"""
        return self.db.query(EnsembleModel).filter(EnsembleModel.id == ensemble_id).first()
    
    def list_ensembles(self, skip: int = 0, limit: int = 100) -> List[EnsembleModel]:
        """List ensembles with pagination"""
        return self.db.query(EnsembleModel).order_by(EnsembleModel.created_at.desc()).offset(skip).limit(limit).all()
    
    def count_ensembles(self) -> int:
        """Count total ensembles"""
        return self.db.query(func.count(EnsembleModel.id)).scalar()
    
    def create_backtest(self, backtest_data: Dict[str, Any]) -> BacktestResult:
        """Create a new backtest result"""
        backtest = BacktestResult(**backtest_data)
        self.db.add(backtest)
        self.db.commit()
        self.db.refresh(backtest)
        return backtest
    
    def get_backtest(self, backtest_id: int) -> Optional[BacktestResult]:
        """Get a backtest by ID"""
        return self.db.query(BacktestResult).filter(BacktestResult.id == backtest_id).first()
    
    def list_backtests(self, filters: Dict[str, Any] = None, skip: int = 0, limit: int = 100) -> List[BacktestResult]:
        """List backtests with optional filtering"""
        query = self.db.query(BacktestResult)
        
        if filters:
            if "symbol" in filters and filters["symbol"]:
                query = query.filter(BacktestResult.symbol == filters["symbol"])
                
            if "model_id" in filters and filters["model_id"]:
                query = query.filter(BacktestResult.model_id == filters["model_id"])
                
            if "created_after" in filters and filters["created_after"]:
                query = query.filter(BacktestResult.created_at >= filters["created_after"])
                
            if "min_return" in filters and filters["min_return"] is not None:
                # Assumes total_return is stored in the performance_metrics JSON
                # This is a simplification - would need JSON extraction in real implementation
                pass
                
        # Order by newest first
        query = query.order_by(BacktestResult.created_at.desc())
        
        # Apply pagination
        return query.offset(skip).limit(limit).all()
    
    def count_backtests(self, filters: Dict[str, Any] = None) -> int:
        """Count total backtests with optional filtering"""
        query = self.db.query(func.count(BacktestResult.id))
        
        if filters:
            if "symbol" in filters and filters["symbol"]:
                query = query.filter(BacktestResult.symbol == filters["symbol"])
                
            if "model_id" in filters and filters["model_id"]:
                query = query.filter(BacktestResult.model_id == filters["model_id"])
                
            if "created_after" in filters and filters["created_after"]:
                query = query.filter(BacktestResult.created_at >= filters["created_after"])
                
            # Skip min_return filtering as it requires JSON extraction
                
        return query.scalar()