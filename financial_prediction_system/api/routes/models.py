"""API routes for model management"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import json

from financial_prediction_system.api.dependencies import get_db
from financial_prediction_system.api.schemas import (
    CreateModelRequest, CreateEnsembleRequest, ModelFilterRequest,
    ModelResponse, ModelListResponse, EnsembleResponse, ModelTrainingResponse,
    ModelExplanationRequest, ModelExplanationResponse
)
from financial_prediction_system.infrastructure.database.model_store import ModelRepository
from financial_prediction_system.core.models.factory import ModelFactory
from financial_prediction_system.core.models.training import ModelTrainer
from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.core.targets.target_builder import TargetBuilder
from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
from financial_prediction_system.core.evaluation.metrics import calculate_metrics
from financial_prediction_system.core.evaluation.model_explainer import ModelExplainer
from financial_prediction_system.data_loaders.data_providers import get_market_calendar_dates
import QuantLib as ql

router = APIRouter(prefix="/models", tags=["models"])

# Helper function to convert NumPy types to Python native types
def convert_numpy_to_python(value):
    """Convert NumPy data types to native Python types for database storage"""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: convert_numpy_to_python(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_numpy_to_python(item) for item in value]
    else:
        return value

@router.post("/train", response_model=ModelTrainingResponse, status_code=202)
async def train_model(request: CreateModelRequest, db: Session = Depends(get_db)):
    """Train a new prediction model"""
    # Create repository
    repo = ModelRepository(db)
    
    try:
        # 1. Load data
        # Determine loader type based on symbol - stock symbols like AAPL use the stock loader
        loader_type = "stock"  # Default to stock for equity symbols
        loader = DataLoaderFactory.get_loader(loader_type, db)
        
        # Convert datetime to date for database compatibility
        train_start = request.train_start_date.date() if hasattr(request.train_start_date, 'date') else request.train_start_date
        train_end = request.train_end_date.date() if hasattr(request.train_end_date, 'date') else request.train_end_date
        
        data = loader.load_data(
            symbol=request.symbol,
            start_date=train_start,
            end_date=train_end
        )
        
        # Load market index data if needed
        index_data = {}
        if any(feature.value in ["market_regime", "market_index_relationship", "sector_behavior"] for feature in request.features):
            # Load index data starting from 100 days before the training period to account for lookback windows
            extended_start = train_start - timedelta(days=100)
            index_data = DataLoaderFactory.get_index_data(db, extended_start, train_end)
        
        # Load treasury data if needed
        treasury_data = pd.DataFrame()
        if any(feature.value in ["treasury_rate", "treasury_rate_equity_relationship"] for feature in request.features):
            # Load treasury data with same extended window
            extended_start = train_start - timedelta(days=100)
            treasury_loader = DataLoaderFactory.get_loader("treasury", db)
            treasury_data = treasury_loader.load_data("", extended_start, train_end)
        
        # 2. Prepare features
        feature_builder = FeatureBuilder(data)
        for feature_set in request.features:
            if feature_set.value in ["market_regime", "market_index_relationship", "sector_behavior"]:
                # Pass index data for market-related features
                feature_builder.add_feature_set(feature_set.value, index_data=index_data)
            elif feature_set.value == "treasury_rate":
                # Pass treasury data for yield features
                feature_builder.add_feature_set(feature_set.value, yields_data=treasury_data)
            elif feature_set.value == "treasury_rate_equity_relationship":
                # Pass both treasury and index data for relationship features
                feature_builder.add_feature_set(feature_set.value, yields_data=treasury_data, index_data=index_data)
            else:
                feature_builder.add_feature_set(feature_set.value)
        
        features_df = feature_builder.build()
        
        # 3. Prepare targets
        target_builder = TargetBuilder(data)
        target_type = request.target_type.value
        target_builder.add_target_set(target_type)
        
        targets_df = target_builder.build()
        
        # FIX: Align features and targets dataframes to ensure consistent samples
        # Get common dates between features and targets
        common_dates = features_df.index.intersection(targets_df.index)
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between features and targets. Check feature and target generation.")
            
        features_df = features_df.loc[common_dates]
        targets_df = targets_df.loc[common_dates]
        
        # Verify alignment
        if features_df.shape[0] != targets_df.shape[0]:
            raise ValueError(f"Features and targets arrays still have inconsistent sizes after alignment: {features_df.shape[0]} vs {targets_df.shape[0]}")
        
        # 4. Create model
        model_type = request.model_type.value.lower()  # Convert to lowercase
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
        
        # 8. Update the model record with metrics - Convert NumPy types to Python native types
        metrics = convert_numpy_to_python(metrics)
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
            feature_importance = model.feature_importance_
            if hasattr(feature_importance, "tolist"):
                feature_importance = feature_importance.tolist()
                
            feature_importance_data = [
                {"feature": feature, "importance": convert_numpy_to_python(importance)}
                for feature, importance in zip(features_df.columns, feature_importance)
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

@router.get("/available-models", response_model=Dict[str, Any])
async def get_available_models():
    """Get lists of available model types, target types, and feature sets with descriptions"""
    available_models = ModelFactory.get_available_models()
    model_docs = ModelFactory.get_model_documentation()
    
    # Get available targets and feature sets
    # In a real implementation, we would get these from the respective modules
    available_targets = [
        "price", "drawdown", "alpha", "volatility", "jump_risk", "tail_risk"
    ]
    
    available_features = [
        "technical", "volume", "date", "treasury_rate", 
        "treasury_rate_equity_relationship", "volatility", 
        "price_action_ranges", "price_action_gaps", "sector_behavior", 
        "market_index_relationship", "market_regime"
    ]
    
    return {
        "success": True,
        "model_types": available_models,
        "model_documentation": model_docs,
        "target_types": available_targets,
        "feature_sets": available_features
    }

@router.post("/update-market-data", response_model=dict)
async def update_market_data(db: Session = Depends(get_db)):
    """
    Update the database with the latest market data
    """
    from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
    from datetime import date, timedelta, datetime
    
    try:
        results = {}
        
        # Update stock data
        stock_loader = DataLoaderFactory.get_loader("stock", db)
        stock_latest = stock_loader.get_latest_date()
        if stock_latest:
            stocks_updated = stock_loader.update_daily_data()
            results["stocks"] = {
                "latest_date_before": stock_latest.isoformat(),
                "records_updated": stocks_updated
            }
        else:
            # If no data exists, load a limited history (e.g., 1 year)
            one_year_ago = date.today() - timedelta(days=365)
            stocks_updated = stock_loader.load_historical_data(start_date=one_year_ago)
            results["stocks"] = {
                "message": "No existing data found - loaded initial data",
                "records_loaded": stocks_updated
            }
        
        # Update index data
        index_loader = DataLoaderFactory.get_loader("index", db)
        index_latest = index_loader.get_latest_date()
        if index_latest:
            indices_updated = index_loader.update_daily_data()
            results["indices"] = {
                "latest_date_before": index_latest.isoformat(),
                "records_updated": indices_updated
            }
        else:
            # If no data exists, load a limited history (e.g., 1 year)
            one_year_ago = date.today() - timedelta(days=365)
            indices_updated = index_loader.load_historical_data(start_date=one_year_ago)
            results["indices"] = {
                "message": "No existing data found - loaded initial data",
                "records_loaded": indices_updated
            }
        
        # Update treasury data
        treasury_loader = DataLoaderFactory.get_loader("treasury", db)
        treasury_latest = treasury_loader.get_latest_date()
        if treasury_latest:
            treasuries_updated = treasury_loader.update_daily_data()
            results["treasuries"] = {
                "latest_date_before": treasury_latest.isoformat(),
                "records_updated": treasuries_updated
            }
        else:
            # If no data exists, load a limited history (e.g., 1 year)
            one_year_ago = date.today() - timedelta(days=365)
            treasuries_updated = treasury_loader.load_historical_data(start_date=one_year_ago)
            results["treasuries"] = {
                "message": "No existing data found - loaded initial data",
                "records_loaded": treasuries_updated
            }
        
        # Commit all changes
        db.commit()
        
        return {
            "success": True,
            "message": "Market data updated successfully",
            "results": results
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating market data: {str(e)}")

@router.post("/fill-data-gap", response_model=dict)
async def fill_data_gap(
    lookback_days: int = 7,
    specific_date: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Fill market data gaps within the specified lookback period or for a specific date.
    
    Args:
        lookback_days: Number of days to look back for gaps (default: 7)
        specific_date: Optional specific date to fill (format: YYYY-MM-DD)
    """
    from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
    from financial_prediction_system.data_loaders.data_providers import get_market_calendar_dates
    from datetime import date, datetime, timedelta
    import QuantLib as ql
    
    try:
        results = {}
        
        # Handle specific date if provided
        if specific_date:
            try:
                target_date = date.fromisoformat(specific_date)
                
                # Verify it's a valid market date
                ql_date = ql.Date(target_date.day, target_date.month, target_date.year)
                nyse = ql.UnitedStates(ql.UnitedStates.NYSE)
                
                if not nyse.isBusinessDay(ql_date):
                    return {
                        "success": False,
                        "message": f"The date {specific_date} is not a valid NYSE market day"
                    }
                
                # Set date range to just this date
                start_date = target_date
                end_date = target_date
                
            except ValueError:
                return {
                    "success": False,
                    "message": f"Invalid date format: {specific_date}. Use YYYY-MM-DD format."
                }
        else:
            # Use lookback period
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=lookback_days)
            
        # Get list of market days in the date range
        market_dates = get_market_calendar_dates(start_date, end_date, "NYSE")
        
        if not market_dates:
            return {
                "success": True,
                "message": "No market days in the specified period"
            }
            
        # Update stock data for the period
        stock_loader = DataLoaderFactory.get_loader("stock", db)
        stock_records = stock_loader.load_historical_data(
            start_date=market_dates[0], 
            end_date=market_dates[-1]
        )
        
        results["stocks"] = {
            "date_range": f"{market_dates[0].isoformat()} to {market_dates[-1].isoformat()}",
            "market_days": len(market_dates),
            "records_filled": stock_records
        }
        
        # Update index data for the period
        index_loader = DataLoaderFactory.get_loader("index", db)
        index_records = index_loader.load_historical_data(
            start_date=market_dates[0], 
            end_date=market_dates[-1]
        )
        
        results["indices"] = {
            "date_range": f"{market_dates[0].isoformat()} to {market_dates[-1].isoformat()}",
            "market_days": len(market_dates),
            "records_filled": index_records
        }
        
        # Update treasury data for the period
        treasury_loader = DataLoaderFactory.get_loader("treasury", db)
        treasury_records = treasury_loader.load_historical_data(
            start_date=market_dates[0], 
            end_date=market_dates[-1]
        )
        
        results["treasuries"] = {
            "date_range": f"{market_dates[0].isoformat()} to {market_dates[-1].isoformat()}",
            "market_days": len(market_dates),
            "records_filled": treasury_records
        }
        
        # Force commit all changes
        db.commit()
        
        return {
            "success": True,
            "message": f"Filled data gaps for market days from {market_dates[0].isoformat()} to {market_dates[-1].isoformat()}",
            "results": results
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error filling data gap: {str(e)}")

@router.post("/explain/{model_id}", response_model=ModelExplanationResponse)
async def explain_model(
    model_id: int = Path(..., ge=1),
    request: ModelExplanationRequest = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Generate SHAP explanations for a trained model
    
    Args:
        model_id: ID of the model to explain
        request: Optional configuration for the explanation
        
    Returns:
        Dictionary with explanation results and links to visualizations
    """
    repo = ModelRepository(db)
    
    try:
        # Retrieve the model
        model_record = repo.get_model(model_id)
        if not model_record:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
            
        # Load the model
        model_type = model_record.model_type
        model = ModelFactory.create_model(model_type, **model_record.hyperparameters or {})
        model.load(model_record.model_path)
        
        # Get sample data for explanation
        sample_size = request.sample_size if request and request.sample_size else 100
        loader_type = "stock"  # Default to stock for equity symbols
        loader = DataLoaderFactory.get_loader(loader_type, db)
        
        # Get data for explanation (use test period from model record)
        data = loader.load_data(
            symbol=model_record.symbol,
            start_date=model_record.test_start_date,
            end_date=model_record.test_end_date
        )
        
        # Prepare features
        feature_builder = FeatureBuilder(data)
        feature_sets = model_record.feature_sets
        
        # Load additional data if needed for specific feature sets
        index_data = {}
        treasury_data = pd.DataFrame()
        
        # Check if market data is needed
        if any(fs in ["market_regime", "market_index_relationship", "sector_behavior"] for fs in feature_sets):
            # Load index data with extended window for lookback
            extended_start = model_record.test_start_date - timedelta(days=100)
            index_data = DataLoaderFactory.get_index_data(db, extended_start, model_record.test_end_date)
        
        # Check if treasury data is needed
        if any(fs in ["treasury_rate", "treasury_rate_equity_relationship"] for fs in feature_sets):
            # Load treasury data with extended window
            extended_start = model_record.test_start_date - timedelta(days=100)
            treasury_loader = DataLoaderFactory.get_loader("treasury", db)
            treasury_data = treasury_loader.load_data("", extended_start, model_record.test_end_date)
        
        # Add feature sets
        for feature_set in feature_sets:
            if feature_set in ["market_regime", "market_index_relationship", "sector_behavior"]:
                feature_builder.add_feature_set(feature_set, index_data=index_data)
            elif feature_set == "treasury_rate":
                feature_builder.add_feature_set(feature_set, yields_data=treasury_data)
            elif feature_set == "treasury_rate_equity_relationship":
                feature_builder.add_feature_set(feature_set, yields_data=treasury_data, index_data=index_data)
            else:
                feature_builder.add_feature_set(feature_set)
        
        # Build features
        features_df = feature_builder.build()
        
        # Sample features for explanation
        if features_df.shape[0] > sample_size:
            features_sample = features_df.sample(sample_size, random_state=42)
        else:
            features_sample = features_df
        
        # Generate SHAP explanation
        explanation = model.explain(features_sample)
        
        # Generate visualization
        visualizations = {}
        
        # If background tasks is available, generate plots asynchronously
        if background_tasks:
            # Create directory for visualizations
            os.makedirs(f"static/explanations/{model_id}", exist_ok=True)
            
            # Function to generate and save plots
            def generate_plots():
                explainer = ModelExplainer(model, model_type, list(features_sample.columns))
                
                # Summary plot
                summary_path = f"static/explanations/{model_id}/summary_plot.png"
                explainer.save_summary_plot(features_sample, summary_path)
                
                # Waterfall plot for a sample instance
                if features_sample.shape[0] > 0:
                    waterfall_path = f"static/explanations/{model_id}/waterfall_plot.png"
                    explainer.generate_waterfall_plot(features_sample, 0, waterfall_path)
                
                # Save feature importance as JSON
                importance_path = f"static/explanations/{model_id}/feature_importance.json"
                with open(importance_path, 'w') as f:
                    json.dump(explanation.get("feature_importance", {}), f, indent=2)
            
            # Add task to background tasks
            background_tasks.add_task(generate_plots)
            
            # Add paths to response
            visualizations = {
                "summary_plot": f"/static/explanations/{model_id}/summary_plot.png",
                "waterfall_plot": f"/static/explanations/{model_id}/waterfall_plot.png",
                "feature_importance_json": f"/static/explanations/{model_id}/feature_importance.json"
            }
        
        # Create response
        response = {
            "model_id": model_id,
            "explanation": explanation,
            "visualizations": visualizations
        }
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")
