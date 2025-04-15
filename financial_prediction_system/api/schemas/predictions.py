"""Schemas for prediction endpoints"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

from .base import BaseResponse
from .models import ModelTypeEnum, FeatureSetEnum

class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    model_id: int = Field(..., description="ID of the model to use")
    symbol: str = Field(..., description="Symbol to predict")
    prediction_date: Optional[datetime] = Field(None, description="Date for prediction (defaults to latest data)")
    custom_features: Optional[Dict[str, Any]] = Field(None, description="Optional custom feature values")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": 1,
                "symbol": "SPX",
                "prediction_date": "2023-04-15T00:00:00",
                "custom_features": {
                    "market_regime": "bullish"
                }
            }
        }

class EnsemblePredictionRequest(BaseModel):
    """Request model for making ensemble predictions"""
    ensemble_id: int = Field(..., description="ID of the ensemble to use")
    symbol: str = Field(..., description="Symbol to predict")
    prediction_date: Optional[datetime] = Field(None, description="Date for prediction (defaults to latest data)")
    custom_features: Optional[Dict[str, Any]] = Field(None, description="Optional custom feature values")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ensemble_id": 1,
                "symbol": "SPX",
                "prediction_date": "2023-04-15T00:00:00",
                "custom_features": {
                    "market_regime": "bullish"
                }
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for making batch predictions"""
    model_id: int = Field(..., description="ID of the model to use")
    symbols: List[str] = Field(..., description="Symbols to predict")
    prediction_date: Optional[datetime] = Field(None, description="Date for prediction (defaults to latest data)")
    custom_features: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Optional custom feature values by symbol")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": 1,
                "symbols": ["SPX", "NDX", "DJI"],
                "prediction_date": "2023-04-15T00:00:00",
                "custom_features": {
                    "SPX": {"market_regime": "bullish"},
                    "NDX": {"market_regime": "neutral"}
                }
            }
        }

class PredictionResult(BaseModel):
    """Model for a single prediction result"""
    symbol: str = Field(..., description="Symbol")
    prediction_date: datetime = Field(..., description="Date of prediction")
    prediction: Union[float, Dict[str, float]] = Field(..., description="Prediction value or probability distribution")
    confidence: Optional[float] = Field(None, description="Confidence score (if available)")
    feature_contributions: Optional[Dict[str, float]] = Field(None, description="Feature contributions to the prediction")

class PredictionResponse(BaseResponse):
    """Response model for prediction request"""
    result: PredictionResult = Field(..., description="Prediction result")
    model_id: int = Field(..., description="ID of the model used")
    model_type: str = Field(..., description="Type of the model used")

class BatchPredictionResponse(BaseResponse):
    """Response model for batch prediction request"""
    results: List[PredictionResult] = Field(..., description="Prediction results")
    model_id: int = Field(..., description="ID of the model used")
    model_type: str = Field(..., description="Type of the model used")

class EnsemblePredictionResponse(BaseResponse):
    """Response model for ensemble prediction request"""
    result: PredictionResult = Field(..., description="Ensemble prediction result")
    ensemble_id: int = Field(..., description="ID of the ensemble used")
    base_predictions: Optional[List[Dict[str, Any]]] = Field(None, description="Predictions from base models")
