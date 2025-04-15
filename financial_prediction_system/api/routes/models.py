"""API routes for model management"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from financial_prediction_system.api.dependencies import get_db
from financial_prediction_system.api.schemas import (
    CreateModelRequest, CreateEnsembleRequest, ModelFilterRequest,
    ModelResponse, ModelListResponse, EnsembleResponse, ModelTrainingResponse
)
from financial_prediction_system.infrastructure.database.model_store import ModelRepository
from financial_prediction_system.core.models.factory import ModelFactory
from financial_prediction_system.core.models.training import ModelTrainer
from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.core.targets.target_builder import TargetBuilder
from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
from financial_prediction_system.core.evaluation.metrics import calculate_metrics

router = APIRouter(prefix="/models", tags=["models"])

@router.post("/train", response_model=ModelTrainingResponse, status_code=202)
async def train_model(request: CreateModelRequest, db: Session = Depends(get_db)):
    """Train a new prediction model"""
    # Create repository
    repo = ModelRepository(db)
    
    try:
        # 1. Load data
        loader = DataLoaderFactory.create_loader(request.symbol)
        data = loader.load_data(
            start_date=request.train_start_date,
            end_date=request.train_end_date
        )
        
        # 2. Prepare features
        feature_builder = FeatureBuilder(data)
        for feature_set in request.features:
            feature_builder.add_feature_set(feature_set.value)
        
        features_df = feature_builder.build()
        
        # 3. Prepare targets
        target_builder = TargetBuilder(data)
        target_type = request.target_type.value
        target_builder.add_target_set(target_type)
        
        targets_df = target_builder.build()
        
        # 4. Create model
        model_type = request.model_type.value
        model = ModelFactory.create_model(model_type, **request.hyperparameters or {})
        
        # 5. Save initial model entry to get an ID
        model_data = {
            "symbol": request.symbol,
            "model_type": model_type,
            "target_type": target_type,
            "feature_sets": [f.value for f in request.features],
            "hyperparameters": request.hyperparameters,
            "train_start_date": request.train_start_date,
            "train_end_date": request.train_end_date,
            "test_start_date": request.test_start_date,
            "test_end_date": request.test_end_date
        }
        model_record = repo.create_model(model_data)
        
        # 6. Train the model asynchronously (in a real implementation)
        # For simplicity, we'll train synchronously here
        target_column = targets_df.columns[0]  # Use the first target column
        model.train(features_df, targets_df[target_column])
        
        # 7. Evaluate the model
        # We would split the data for proper testing
        metrics = model.evaluate(features_df, targets_df[target_column])
        
        # 8. Update the model record with metrics
        updated_model_data = {
            "test_mse": metrics.get("mse"),
            "test_r2": metrics.get("r2"),
            "test_mae": metrics.get("mae"),
            "cv_mean_mse": metrics.get("cv_mean_mse"),
            "cv_std_mse": metrics.get("cv_std_mse"),
            "cv_mean_r2": metrics.get("cv_mean_r2"),
            "cv_std_r2": metrics.get("cv_std_r2"),
            "cv_mean_mae": metrics.get("cv_mean_mae"),
            "cv_std_mae": metrics.get("cv_std_mae"),
            "model_path": f"/models/{model_record.id}.pkl"  # We would save the model to this path
        }
        repo.update_model(model_record.id, updated_model_data)
        
        # 9. Add feature importance if available
        if hasattr(model, "feature_importance_"):
            feature_importance_data = [
                {"feature": feature, "importance": importance}
                for feature, importance in zip(features_df.columns, model.feature_importance_)
            ]
            repo.add_feature_importance(model_record.id, feature_importance_data)
        
        return {
            "success": True,
            "model_id": model_record.id,
            "status": "completed",
            "metrics": metrics
        }
    except Exception as e:
        # In a real implementation, we'd use proper error handling and logging
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")

@router.get("/", response_model=ModelListResponse)
async def list_models(
    symbol: Optional[str] = None,
    model_type: Optional[str] = None,
    target_type: Optional[str] = None,
    created_after: Optional[datetime] = None,
    min_r2: Optional[float] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List models with optional filtering"""
    repo = ModelRepository(db)
    
    filters = {
        "symbol": symbol,
        "model_type": model_type,
        "target_type": target_type,
        "created_after": created_after,
        "min_r2": min_r2
    }
    
    # Clean up filters by removing None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    # Calculate skip based on page and page_size
    skip = (page - 1) * page_size
    
    # Get models and total count
    models = repo.list_models(filters, skip, page_size)
    total = repo.count_models(filters)
    
    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "success": True,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "models": models
    }

@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    """Get details of a specific model"""
    repo = ModelRepository(db)
    model = repo.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    return {
        "success": True,
        "model": model,
        "feature_importance": model.feature_importance
    }

@router.delete("/{model_id}", response_model=dict)
async def delete_model(model_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    """Delete a model"""
    repo = ModelRepository(db)
    result = repo.delete_model(model_id)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    return {
        "success": True,
        "message": f"Model with ID {model_id} deleted"
    }

@router.post("/ensemble", response_model=EnsembleResponse)
async def create_ensemble(request: CreateEnsembleRequest, db: Session = Depends(get_db)):
    """Create an ensemble of models"""
    repo = ModelRepository(db)
    
    # Verify all models exist
    for model_id in request.model_ids:
        model = repo.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    try:
        # Create ensemble
        ensemble_data = {
            "name": request.name,
            "ensemble_method": request.ensemble_method.value,
            "hyperparameters": request.hyperparameters
        }
        
        ensemble = repo.create_ensemble(ensemble_data, request.model_ids)
        
        # In a real implementation, we would create the actual ensemble model here
        # and save it to disk, then update the model_path in the ensemble record
        
        return {
            "success": True,
            "ensemble": ensemble
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating ensemble: {str(e)}")

@router.get("/ensemble/{ensemble_id}", response_model=EnsembleResponse)
async def get_ensemble(ensemble_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    """Get details of a specific ensemble"""
    repo = ModelRepository(db)
    ensemble = repo.get_ensemble(ensemble_id)
    
    if not ensemble:
        raise HTTPException(status_code=404, detail=f"Ensemble with ID {ensemble_id} not found")
    
    return {
        "success": True,
        "ensemble": ensemble
    }

@router.get("/available-models", response_model=Dict[str, List[str]])
async def get_available_models():
    """Get lists of available model types, target types, and feature sets"""
    available_models = ModelFactory.get_available_models()
    
    # Get available targets and feature sets
    # In a real implementation, we would get these from the respective modules
    available_targets = [
        "price", "drawdown", "alpha", "volatility", "jump_risk", "tail_risk"
    ]
    
    available_features = [
        "technical", "volume", "date", "treasury_rate", 
        "treasury_rate_equity_reationship", "volatility", 
        "price_action_ranges", "price_action_gaps", "sector_behavior", 
        "market_index_relationship", "market_regime"
    ]
    
    return {
        "success": True,
        "model_types": available_models,
        "target_types": available_targets,
        "feature_sets": available_features
    }
