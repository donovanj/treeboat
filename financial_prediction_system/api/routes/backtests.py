"""API routes for backtesting"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

from financial_prediction_system.api.dependencies import get_db
from financial_prediction_system.api.schemas import (
    BacktestRequest, EnsembleBacktestRequest, BacktestFilterRequest,
    BacktestResponse, BacktestListResponse
)
from financial_prediction_system.infrastructure.database.model_store import ModelRepository
from financial_prediction_system.core.evaluation.backtesting import Backtester
from financial_prediction_system.core.evaluation.metrics import calculate_backtest_metrics
from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory

router = APIRouter(prefix="/backtests", tags=["backtests"])

@router.post("/", response_model=BacktestResponse)
async def create_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """Create a new backtest for a model"""
    repo = ModelRepository(db)
    model_record = repo.get_model(request.model_id)
    
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {request.model_id} not found")
    
    if not model_record.model_path:
        raise HTTPException(status_code=400, detail=f"Model with ID {request.model_id} has no saved model file")
    
    try:
        # 1. Load the model
        from financial_prediction_system.api.routes.predictions import load_model
        model = load_model(model_record.model_path)
        
        # 2. Load data for the symbol
        loader_type = "stock"  # Default to stock for equity symbols
        loader = DataLoaderFactory.get_loader(loader_type, db)
        
        # Convert datetime to date for database compatibility
        start_date = request.start_date.date() if hasattr(request.start_date, 'date') else request.start_date
        end_date = request.end_date.date() if hasattr(request.end_date, 'date') else request.end_date
        
        data = loader.load_data(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Load market index data if needed
        index_data = {}
        if any(feature_set in ["market_regime", "market_index_relationship", "sector_behavior"] for feature_set in model_record.feature_sets):
            # Load index data for market-related features
            index_data = DataLoaderFactory.get_index_data(db, start_date, end_date)
            
        # Load treasury data if needed
        treasury_data = pd.DataFrame()
        if any(feature_set in ["treasury_rate", "treasury_rate_equity_relationship"] for feature_set in model_record.feature_sets):
            # Load treasury data
            treasury_loader = DataLoaderFactory.get_loader("treasury", db)
            treasury_data = treasury_loader.load_data("", start_date, end_date)
        
        # 3. Run backtest
        backtester = Backtester(
            model=model,
            data=data,
            features=model_record.feature_sets,
            parameters=request.parameters or {},
            index_data=index_data,  # Pass the index data to the backtester
            treasury_data=treasury_data  # Pass the treasury data to the backtester
        )
        
        backtest_results = backtester.run()
        
        # 4. Calculate performance metrics
        metrics = calculate_backtest_metrics(backtest_results)
        
        # 5. Save backtest results to database
        backtest_data = {
            "model_id": model_record.id,
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "performance_metrics": metrics,
            "selected_features": model_record.feature_sets,
            "parameters": request.parameters,
            "trades": backtest_results.get("trades"),
            "equity_curve": backtest_results.get("equity_curve")
        }
        
        backtest_record = repo.create_backtest(backtest_data)
        
        return {
            "success": True,
            "backtest": backtest_record,
            "model_id": model_record.id,
            "trades": backtest_record.trades,
            "equity_curve": backtest_record.equity_curve
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")

@router.post("/ensemble", response_model=BacktestResponse)
async def create_ensemble_backtest(request: EnsembleBacktestRequest, db: Session = Depends(get_db)):
    """Create a new backtest for an ensemble"""
    repo = ModelRepository(db)
    ensemble_record = repo.get_ensemble(request.ensemble_id)
    
    if not ensemble_record:
        raise HTTPException(status_code=404, detail=f"Ensemble with ID {request.ensemble_id} not found")
    
    if not ensemble_record.model_path:
        raise HTTPException(status_code=400, detail=f"Ensemble with ID {request.ensemble_id} has no saved model file")
    
    try:
        # 1. Load the ensemble model
        from financial_prediction_system.api.routes.predictions import load_model
        ensemble_model = load_model(ensemble_record.model_path)
        
        # 2. Load data for the symbol
        loader_type = "stock"  # Default to stock for equity symbols
        loader = DataLoaderFactory.get_loader(loader_type, db)
        
        # Convert datetime to date for database compatibility
        start_date = request.start_date.date() if hasattr(request.start_date, 'date') else request.start_date
        end_date = request.end_date.date() if hasattr(request.end_date, 'date') else request.end_date
        
        data = loader.load_data(
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Load market index data if needed
        index_data = {}
        if any(feature_set in ["market_regime", "market_index_relationship", "sector_behavior"] for feature_set in all_feature_sets):
            # Load index data for market-related features
            index_data = DataLoaderFactory.get_index_data(db, start_date, end_date)
            
        # Load treasury data if needed
        treasury_data = pd.DataFrame()
        if any(feature_set in ["treasury_rate", "treasury_rate_equity_relationship"] for feature_set in all_feature_sets):
            # Load treasury data
            treasury_loader = DataLoaderFactory.get_loader("treasury", db)
            treasury_data = treasury_loader.load_data("", start_date, end_date)
        
        # 3. Get feature sets from all component models
        all_feature_sets = set()
        for mapping in ensemble_record.model_mappings:
            model_record = repo.get_model(mapping.model_id)
            if model_record:
                all_feature_sets.update(model_record.feature_sets)
        
        # 4. Run backtest
        backtester = Backtester(
            model=ensemble_model,
            data=data,
            features=list(all_feature_sets),
            parameters=request.parameters or {},
            index_data=index_data,  # Pass the index data to the backtester
            treasury_data=treasury_data  # Pass the treasury data to the backtester
        )
        
        backtest_results = backtester.run()
        
        # 5. Calculate performance metrics
        metrics = calculate_backtest_metrics(backtest_results)
        
        # 6. Save backtest results to database
        backtest_data = {
            "ensemble_id": ensemble_record.id,
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "performance_metrics": metrics,
            "selected_features": list(all_feature_sets),
            "parameters": request.parameters,
            "trades": backtest_results.get("trades"),
            "equity_curve": backtest_results.get("equity_curve")
        }
        
        backtest_record = repo.create_backtest(backtest_data)
        
        return {
            "success": True,
            "backtest": backtest_record,
            "model_id": None,  # No single model ID for ensemble
            "ensemble_id": ensemble_record.id,
            "trades": backtest_record.trades,
            "equity_curve": backtest_record.equity_curve
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running ensemble backtest: {str(e)}")

@router.get("/", response_model=BacktestListResponse)
async def list_backtests(
    symbol: Optional[str] = None,
    model_id: Optional[int] = None,
    created_after: Optional[datetime] = None,
    min_return: Optional[float] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List backtests with optional filtering"""
    repo = ModelRepository(db)
    
    filters = {
        "symbol": symbol,
        "model_id": model_id,
        "created_after": created_after,
        "min_return": min_return
    }
    
    # Clean up filters by removing None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    # Calculate skip based on page and page_size
    skip = (page - 1) * page_size
    
    # Get backtests and total count
    backtests = repo.list_backtests(filters, skip, page_size)
    total = repo.count_backtests(filters)
    
    # Calculate total pages
    total_pages = (total + page_size - 1) // page_size
    
    return {
        "success": True,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "backtests": backtests
    }

@router.get("/{backtest_id}", response_model=BacktestResponse)
async def get_backtest(backtest_id: int = Path(..., ge=1), db: Session = Depends(get_db)):
    """Get details of a specific backtest"""
    repo = ModelRepository(db)
    backtest = repo.get_backtest(backtest_id)
    
    if not backtest:
        raise HTTPException(status_code=404, detail=f"Backtest with ID {backtest_id} not found")
    
    return {
        "success": True,
        "backtest": backtest,
        "model_id": backtest.model_id,
        "ensemble_id": backtest.ensemble_id,
        "trades": backtest.trades,
        "equity_curve": backtest.equity_curve
    }
