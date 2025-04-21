from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from pydantic import BaseModel
from .middleware.feature_validation import FeatureComposition
from .middleware.feature_middleware import (
    cache_response,
    validate_data_requirements,
    FeatureCalculationError
)
from ..core.features.feature_backtester import FeatureBacktester
from ..core.models import get_default_model
from ..infrastructure.repositories.feature_repository import FeatureRepository
from ..infrastructure.database import get_db_session
from sqlalchemy.orm import Session

router = APIRouter()

class BacktestRequest(BaseModel):
    composition_id: str
    start_date: str
    end_date: str
    target_type: Optional[str] = 'returns'
    cv_folds: Optional[int] = 5
    horizon: Optional[int] = 5

@router.post("/api/features/backtest")
async def backtest_features(
    request: BacktestRequest,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """Run feature backtest analysis"""
    try:
        # Get feature composition from database
        feature_repo = FeatureRepository(db)
        composition = await feature_repo.get_composition(request.composition_id)
        
        if not composition:
            raise HTTPException(
                status_code=404,
                detail=f"Feature composition {request.composition_id} not found"
            )
        
        # Load historical data
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
        
        data = await feature_repo.get_historical_data(
            composition.symbol,
            start_date,
            end_date
        )
        
        validate_data_requirements(data)
        
        # Initialize backtester
        backtester = FeatureBacktester(
            data=data,
            target_type=request.target_type,
            cv_folds=request.cv_folds,
            horizon=request.horizon
        )
        
        # Get default model for CV
        model = get_default_model(request.target_type)
        
        # Run backtest
        results = backtester.backtest_features(composition, model)
        
        if 'error' in results:
            raise FeatureCalculationError(results['error'])
        
        # Add time series data for visualization
        time_series = {
            'dates': data['date'],
            'target': backtester.prepare_target().tolist(),
            'features': []
        }
        
        # Build features for time series
        builder = feature_repo.get_feature_builder(composition)
        features = builder.build()
        
        # Add each feature's time series
        for i in range(features.shape[1]):
            time_series['features'].append(features[:, i].tolist())
        
        results['time_series'] = time_series
        
        # Get feature importance ranking
        importance = backtester.get_feature_importance(composition.id)
        results['feature_importance'] = importance
        
        # Get recommendations
        recommendations = backtester.get_feature_recommendations(composition.id)
        results['recommendations'] = recommendations
        
        return results
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error running feature backtest: {str(e)}"
        )

@router.get("/api/features/backtest/{composition_id}/summary")
@cache_response(expire=3600)
async def get_backtest_summary(
    composition_id: str,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get cached summary of feature backtest results"""
    feature_repo = FeatureRepository(db)
    summary = await feature_repo.get_backtest_summary(composition_id)
    
    if not summary:
        raise HTTPException(
            status_code=404,
            detail=f"No backtest results found for composition {composition_id}"
        )
    
    return summary

@router.get("/api/features/backtest/performance")
async def get_performance_metrics(
    symbol: str,
    start_date: str,
    end_date: str,
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """Get performance metrics for all features over a time period"""
    try:
        feature_repo = FeatureRepository(db)
        
        # Get all feature compositions for symbol
        compositions = await feature_repo.get_compositions_by_symbol(symbol)
        
        performance_metrics = {}
        for composition in compositions:
            # Run quick performance evaluation
            metrics = await feature_repo.get_feature_performance(
                composition.id,
                start_date,
                end_date
            )
            
            performance_metrics[composition.id] = {
                'name': composition.name,
                'metrics': metrics
            }
        
        return performance_metrics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving performance metrics: {str(e)}"
        )