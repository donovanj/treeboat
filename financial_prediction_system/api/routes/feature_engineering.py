from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from sqlalchemy.orm import Session
from financial_prediction_system.infrastructure.repositories.feature_repository import (
    Feature,
    SQLFeatureRepository
)
from financial_prediction_system.api.dependencies import get_db

router = APIRouter()
logger = logging.getLogger(__name__)

class FeaturePreviewRequest(BaseModel):
    formula: str
    data: Dict[str, List[Any]]

class FeatureSaveRequest(BaseModel):
    name: str
    formula: str
    type: str
    symbol: str
    description: Optional[str] = None

class FeatureResponse(BaseModel):
    id: int
    name: str
    formula: str
    type: str
    symbol: Optional[str]
    description: Optional[str]
    mean: Optional[float]
    std: Optional[float]
    price_correlation: Optional[float]
    returns_correlation: Optional[float]
    created_at: str
    updated_at: str

class TargetPreviewRequest(BaseModel):
    feature_values: List[float]
    data: Dict[str, List[Any]]
    target_type: str
    parameters: Dict[str, Any]

@router.post("/feature/preview")
async def preview_feature(request: FeaturePreviewRequest):
    """Executes feature formula on provided data and returns preview results"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Execute the formula in a safe context
        # Note: In production, you'd want to implement proper code sandboxing
        local_vars = {'df': df, 'np': np, 'pd': pd}
        exec(request.formula, {'__builtins__': {}}, local_vars)
        
        # Get the result (last variable assigned)
        feature_values = local_vars.get('result', None)
        if feature_values is None:
            raise HTTPException(status_code=400, detail="Formula must assign result variable")
            
        # Convert to list for JSON serialization
        feature_values = feature_values.tolist() if hasattr(feature_values, 'tolist') else list(feature_values)
        
        # Calculate basic metrics
        metrics = calculate_feature_metrics(df, feature_values)
        
        return {
            "feature": feature_values,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error previewing feature: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/feature/save", response_model=FeatureResponse)
async def save_feature(request: FeatureSaveRequest, db: Session = Depends(get_db)):
    """Saves a feature definition to the database"""
    try:
        repository = SQLFeatureRepository(db)
        
        # Preview the feature to get metrics
        preview_data = await preview_feature(FeaturePreviewRequest(
            formula=request.formula,
            data={}  # Empty for now, metrics will be updated later
        ))
        
        feature = Feature(
            name=request.name,
            formula=request.formula,
            type=request.type,
            symbol=request.symbol,
            description=request.description,
            mean=preview_data["metrics"].get("mean"),
            std=preview_data["metrics"].get("std"),
            price_correlation=preview_data["metrics"].get("price_correlation"),
            returns_correlation=preview_data["metrics"].get("returns_correlation")
        )
        
        saved_feature = repository.save_feature(feature)
        return FeatureResponse(
            id=saved_feature.id,
            name=saved_feature.name,
            formula=saved_feature.formula,
            type=saved_feature.type,
            symbol=saved_feature.symbol,
            description=saved_feature.description,
            mean=saved_feature.mean,
            std=saved_feature.std,
            price_correlation=saved_feature.price_correlation,
            returns_correlation=saved_feature.returns_correlation,
            created_at=saved_feature.created_at.isoformat(),
            updated_at=saved_feature.updated_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error saving feature: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/features/{symbol}", response_model=List[FeatureResponse])
async def get_features_by_symbol(symbol: str, db: Session = Depends(get_db)):
    """Gets all features for a symbol"""
    try:
        repository = SQLFeatureRepository(db)
        features = repository.get_features_by_symbol(symbol)
        return [
            FeatureResponse(
                id=feature.id,
                name=feature.name,
                formula=feature.formula,
                type=feature.type,
                symbol=feature.symbol,
                description=feature.description,
                mean=feature.mean,
                std=feature.std,
                price_correlation=feature.price_correlation,
                returns_correlation=feature.returns_correlation,
                created_at=feature.created_at.isoformat(),
                updated_at=feature.updated_at.isoformat()
            )
            for feature in features
        ]
        
    except Exception as e:
        logger.error(f"Error getting features: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/feature/{feature_id}")
async def delete_feature(feature_id: int, db: Session = Depends(get_db)):
    """Deletes a feature"""
    try:
        repository = SQLFeatureRepository(db)
        success = repository.delete_feature(feature_id)
        if not success:
            raise HTTPException(status_code=404, detail="Feature not found")
        return {"status": "success", "message": f"Feature {feature_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting feature: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/feature/target-preview")
async def preview_target(request: TargetPreviewRequest):
    """Calculates target values based on feature and selected target type"""
    try:
        df = pd.DataFrame(request.data)
        df['feature'] = request.feature_values
        
        target_values = calculate_target(df, request.target_type, request.parameters)
        if target_values is None:
            raise HTTPException(status_code=400, detail="Failed to calculate target values")
            
        target_values = target_values.tolist() if hasattr(target_values, 'tolist') else list(target_values)
        
        # Calculate feature-target relationship metrics
        metrics = calculate_feature_target_metrics(
            pd.Series(request.feature_values),
            pd.Series(target_values)
        )
        
        return {
            "target": target_values,
            "metrics": metrics
        }
        
    except Exception as e:
        logger.error(f"Error calculating target preview: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def calculate_feature_metrics(df: pd.DataFrame, feature_values: List[float]) -> Dict:
    """Calculates various metrics for the feature"""
    feature_series = pd.Series(feature_values)
    
    # Basic statistics
    stats = {
        "mean": feature_series.mean(),
        "std": feature_series.std(),
        "min": feature_series.min(),
        "max": feature_series.max(),
        "null_count": feature_series.isnull().sum()
    }
    
    # Correlation with price/returns if available
    if 'close' in df.columns:
        returns = df['close'].pct_change()
        stats["price_correlation"] = feature_series.corr(df['close'])
        stats["returns_correlation"] = feature_series.corr(returns)
    
    # Autocorrelation
    stats["autocorrelation"] = feature_series.autocorr() if len(feature_series) > 1 else None
    
    return stats

def calculate_target(df: pd.DataFrame, target_type: str, parameters: Dict[str, Any]) -> np.ndarray:
    """Calculate target values based on the specified type and parameters"""
    try:
        if target_type == 'forward_return':
            horizon = parameters.get('horizon', 5)
            return df['close'].pct_change(horizon).shift(-horizon)
            
        elif target_type == 'binary_direction':
            horizon = parameters.get('horizon', 5)
            threshold = parameters.get('threshold', 0)
            forward_returns = df['close'].pct_change(horizon).shift(-horizon)
            return (forward_returns > threshold).astype(int)
            
        elif target_type == 'volatility_adjusted_return':
            horizon = parameters.get('horizon', 5)
            vol_window = parameters.get('vol_window', 21)
            
            forward_returns = df['close'].pct_change(horizon).shift(-horizon)
            rolling_vol = df['close'].pct_change().rolling(vol_window).std()
            
            return forward_returns / (rolling_vol + 1e-10)  # Add small epsilon to avoid division by zero
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
            
    except Exception as e:
        logger.error(f"Error in calculate_target: {str(e)}")
        return None

def calculate_feature_target_metrics(feature: pd.Series, target: pd.Series) -> Dict:
    """Calculate relationship metrics between feature and target"""
    # Drop NaN values that might exist in either series
    valid_mask = ~(feature.isna() | target.isna())
    feature = feature[valid_mask]
    target = target[valid_mask]
    
    if len(feature) < 2:
        return {
            "correlation": None,
            "rank_correlation": None,
            "mutual_info": None,
            "target_stats": None
        }
    
    try:
        from sklearn.metrics import mutual_info_score
        
        return {
            "correlation": feature.corr(target),
            "rank_correlation": feature.corr(target, method='spearman'),
            "mutual_info": mutual_info_score(
                feature.rank(pct=True).round(2),  # Discretize to reduce computation
                target.rank(pct=True).round(2)
            ),
            "target_stats": {
                "mean": target.mean(),
                "std": target.std(),
                "min": target.min(),
                "max": target.max(),
                "null_count": target.isnull().sum()
            }
        }
    except Exception as e:
        logger.error(f"Error calculating feature-target metrics: {str(e)}")
        return None