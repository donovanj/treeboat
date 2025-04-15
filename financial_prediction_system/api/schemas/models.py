"""Schemas for model management endpoints"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

from .base import BaseResponse, PaginatedResponse

# Enums for valid values
class ModelTypeEnum(str, Enum):
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    LSTM = "lstm"
    TRANSFORMER = "transformer"

class TargetTypeEnum(str, Enum):
    PRICE = "price"
    DRAWDOWN = "drawdown"
    ALPHA = "alpha"
    VOLATILITY = "volatility"
    JUMP_RISK = "jump_risk"
    TAIL_RISK = "tail_risk"

class FeatureSetEnum(str, Enum):
    TECHNICAL = "technical"
    VOLUME = "volume"
    DATE = "date"
    TREASURY_RATE = "treasury_rate"
    TREASURY_RATE_EQUITY = "treasury_rate_equity_reationship"
    VOLATILITY = "volatility"
    PRICE_ACTION_RANGES = "price_action_ranges"
    PRICE_ACTION_GAPS = "price_action_gaps"
    SECTOR_BEHAVIOR = "sector_behavior"
    MARKET_INDEX_RELATIONSHIP = "market_index_relationship"
    MARKET_REGIME = "market_regime"

class EnsembleMethodEnum(str, Enum):
    ADA_BOOST = "ada_boost"
    GRADIENT_BOOSTING = "gradient_boosting"
    RANDOM_FORESTS = "random_forests"
    STACKING = "stacking"
    VOTING = "voting"

# Request Models
class CreateModelRequest(BaseModel):
    """Request model for creating a new prediction model"""
    symbol: str = Field(..., description="Trading symbol to model")
    model_type: ModelTypeEnum = Field(..., description="Type of model to create")
    target_type: TargetTypeEnum = Field(..., description="Type of target to predict")
    features: List[FeatureSetEnum] = Field(..., description="Feature sets to use")
    train_start_date: datetime = Field(..., description="Start date for training data")
    train_end_date: datetime = Field(..., description="End date for training data")
    test_start_date: Optional[datetime] = Field(None, description="Start date for test data")
    test_end_date: Optional[datetime] = Field(None, description="End date for test data")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "model_type": "random_forest",
                "target_type": "price",
                "features": ["technical", "volume", "market_regime"],
                "train_start_date": "2020-01-01",
                "train_end_date": "2021-12-31",
                "test_start_date": "2022-01-01",
                "test_end_date": "2022-12-31",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": 10
                }
            }
        }

class CreateEnsembleRequest(BaseModel):
    """Request model for creating an ensemble of models"""
    name: str = Field(..., description="Name for the ensemble")
    model_ids: List[int] = Field(..., description="IDs of models to include in the ensemble")
    ensemble_method: EnsembleMethodEnum = Field(..., description="Ensemble method to use")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Ensemble hyperparameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Market Regime Ensemble",
                "model_ids": [1, 2, 3],
                "ensemble_method": "stacking",
                "hyperparameters": {
                    "use_features": True
                }
            }
        }

class ModelFilterRequest(BaseModel):
    """Request model for filtering models"""
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    model_type: Optional[ModelTypeEnum] = Field(None, description="Filter by model type")
    target_type: Optional[TargetTypeEnum] = Field(None, description="Filter by target type")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    min_r2: Optional[float] = Field(None, description="Minimum R² score")
    page: int = Field(1, description="Page number")
    page_size: int = Field(20, description="Page size")

# Response Models
class ModelFeatureImportance(BaseModel):
    """Model for feature importance"""
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Feature importance score")

class ModelResult(BaseModel):
    """Model for ml_model_results table row"""
    id: int = Field(..., description="Model ID")
    symbol: str = Field(..., description="Symbol")
    model_type: str = Field(..., description="Model type")
    train_start_date: datetime = Field(..., description="Training start date")
    train_end_date: datetime = Field(..., description="Training end date")
    test_mse: Optional[float] = Field(None, description="Test Mean Squared Error")
    test_r2: Optional[float] = Field(None, description="Test R² score")
    test_mae: Optional[float] = Field(None, description="Test Mean Absolute Error")
    cv_mean_mse: Optional[float] = Field(None, description="Cross-validation mean MSE")
    cv_std_mse: Optional[float] = Field(None, description="Cross-validation standard deviation MSE")
    cv_mean_r2: Optional[float] = Field(None, description="Cross-validation mean R²")
    cv_std_r2: Optional[float] = Field(None, description="Cross-validation standard deviation R²")
    cv_mean_mae: Optional[float] = Field(None, description="Cross-validation mean MAE")
    cv_std_mae: Optional[float] = Field(None, description="Cross-validation standard deviation MAE")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
class ModelResponse(BaseResponse):
    """Response model for model details"""
    model: ModelResult = Field(..., description="Model details")
    feature_importance: Optional[List[ModelFeatureImportance]] = Field(None, description="Feature importance")

class ModelListResponse(PaginatedResponse):
    """Response model for listing models"""
    models: List[ModelResult] = Field(..., description="List of models")

class EnsembleModel(BaseModel):
    """Model for ensemble details"""
    id: int = Field(..., description="Ensemble ID")
    name: str = Field(..., description="Ensemble name")
    ensemble_method: str = Field(..., description="Ensemble method")
    model_ids: List[int] = Field(..., description="IDs of models in the ensemble")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class EnsembleResponse(BaseResponse):
    """Response model for ensemble details"""
    ensemble: EnsembleModel = Field(..., description="Ensemble details")

class ModelTrainingResponse(BaseResponse):
    """Response model for model training status"""
    model_id: int = Field(..., description="Model ID")
    status: str = Field(..., description="Training status")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics if available")
