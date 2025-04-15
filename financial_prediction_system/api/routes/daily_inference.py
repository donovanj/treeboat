"""API routes for daily automated inference pipeline"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
import pandas as pd
import json
import os
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from financial_prediction_system.api.dependencies import get_db
from financial_prediction_system.api.schemas import (
    DailyPredictionRequest, 
    DailyPredictionResponse,
    TradingSignalResponse
)
from financial_prediction_system.infrastructure.database.model_store import ModelRepository
from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
from financial_prediction_system.data_loaders.update_manager import DataUpdateManager
from financial_prediction_system.core.models.base import PredictionModel
from financial_prediction_system.logging_config import logger

router = APIRouter(prefix="/daily", tags=["daily_inference"])

# Global scheduler for background tasks
scheduler = AsyncIOScheduler()

def load_model(model_path: str) -> PredictionModel:
    """Load a model from disk"""
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

async def update_all_data(db: Session):
    """Update all market data for today"""
    try:
        update_manager = DataUpdateManager(db)
        results = await update_manager.update_all_today()
        logger.info(f"Daily data update completed: {results}")
        return results
    except Exception as e:
        logger.error(f"Error updating daily data: {str(e)}", exc_info=True)
        return {"error": str(e)}

async def generate_daily_predictions(model_id: int, symbols: List[str], db: Session):
    """Generate predictions for the given model and symbols"""
    try:
        # 1. Get the model from repository
        repo = ModelRepository(db)
        model_record = repo.get_model(model_id)
        
        if not model_record:
            logger.error(f"Model with ID {model_id} not found")
            return {"error": f"Model with ID {model_id} not found"}
        
        if not model_record.model_path:
            logger.error(f"Model with ID {model_id} has no saved model file")
            return {"error": f"Model with ID {model_id} has no saved model file"}
        
        # 2. Load the model
        model = load_model(model_record.model_path)
        
        # 3. Set dates for data loading
        end_date = date.today()
        start_date = end_date - timedelta(days=100)  # 100 days of history for feature calculation
        
        results = []
        
        for symbol in symbols:
            try:
                # 4. Load data for the symbol
                loader_type = "stock"  # Default to stock for equity symbols
                loader = DataLoaderFactory.get_loader(loader_type, db)
                
                data = loader.load_data(symbol=symbol, start_date=start_date, end_date=end_date)
                
                # Skip if no data
                if data.empty:
                    logger.warning(f"No data available for {symbol}")
                    continue
                
                # 5. Load supporting data (market indices, treasury rates)
                index_data = {}
                if any(feature_set in ["market_regime", "market_index_relationship", "sector_behavior"] 
                       for feature_set in model_record.feature_sets):
                    index_data = DataLoaderFactory.get_index_data(db, start_date, end_date)
                
                treasury_data = pd.DataFrame()
                if any(feature_set in ["treasury_rate", "treasury_rate_equity_relationship"] 
                       for feature_set in model_record.feature_sets):
                    treasury_loader = DataLoaderFactory.get_loader("treasury", db)
                    treasury_data = treasury_loader.load_data("", start_date, end_date)
                
                # 6. Build features
                feature_builder = FeatureBuilder(data)
                for feature_set in model_record.feature_sets:
                    if feature_set in ["market_regime", "market_index_relationship", "sector_behavior"]:
                        feature_builder.add_feature_set(feature_set, index_data=index_data)
                    elif feature_set == "treasury_rate":
                        feature_builder.add_feature_set(feature_set, yields_data=treasury_data)
                    elif feature_set == "treasury_rate_equity_relationship":
                        feature_builder.add_feature_set(feature_set, yields_data=treasury_data, index_data=index_data)
                    else:
                        feature_builder.add_feature_set(feature_set)
                
                features_df = feature_builder.build()
                
                # Skip if insufficient features
                if features_df.empty:
                    logger.warning(f"Could not generate features for {symbol}")
                    continue
                
                # 7. Make prediction using the last row of features
                prediction = model.predict(features_df.iloc[-1:])
                
                # 8. Extract prediction value and supporting information
                prediction_value = prediction[0]  # Extract scalar value
                
                # For classification models, convert to probabilities if needed
                if model_record.model_type.endswith("classifier"):
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(features_df.iloc[-1:])[0]
                        prediction_value = {str(i): float(p) for i, p in enumerate(probabilities)}
                
                # Get feature contributions if available
                feature_contributions = None
                if hasattr(model, "feature_contributions"):
                    feature_contributions = model.feature_contributions(features_df.iloc[-1:])
                    feature_contributions = {str(k): float(v) for k, v in feature_contributions.items()}
                
                # Calculate confidence score if available
                confidence = None
                if hasattr(model, "predict_proba") and not isinstance(prediction_value, dict):
                    probas = model.predict_proba(features_df.iloc[-1:])[0]
                    confidence = float(max(probas))
                
                # 9. Get last price for trading signal calculation
                last_price = data.iloc[-1]['close'] if 'close' in data.columns else None
                
                # 10. Generate trading signal based on prediction
                trading_signal = generate_trading_signal(
                    symbol=symbol,
                    prediction=prediction_value,
                    last_price=last_price,
                    model_type=model_record.model_type
                )
                
                # 11. Add to results
                results.append({
                    "symbol": symbol,
                    "prediction_date": end_date.isoformat(),
                    "prediction": prediction_value,
                    "confidence": confidence,
                    "feature_contributions": feature_contributions,
                    "last_price": last_price,
                    "trading_signal": trading_signal
                })
                
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {str(e)}", exc_info=True)
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
        
        # 12. Save results to disk
        save_path = os.path.join("predictions", f"daily_{model_id}_{end_date.isoformat()}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump({
                "model_id": model_id,
                "model_type": model_record.model_type,
                "date": end_date.isoformat(),
                "results": results
            }, f, indent=2)
        
        return {
            "success": True,
            "model_id": model_id,
            "results": results,
            "date": end_date.isoformat(),
            "saved_to": save_path
        }
        
    except Exception as e:
        logger.error(f"Error in daily prediction generation: {str(e)}", exc_info=True)
        return {"error": str(e)}

def generate_trading_signal(symbol: str, prediction, last_price: float, model_type: str) -> Dict[str, Any]:
    """
    Generate trading signals from prediction values
    
    This is a simplified implementation - in a real system, you would have more
    sophisticated rules based on your trading strategy
    """
    if not last_price:
        return {"action": "HOLD", "reason": "Missing price data"}
    
    # Default response
    signal = {
        "action": "HOLD",
        "reason": "Default action",
        "target_price": None,
        "stop_loss": None
    }
    
    try:
        # For regression models predicting price
        if model_type == "regression" or model_type.endswith("regressor"):
            # If predicted price is a scalar
            if isinstance(prediction, (int, float)):
                predicted_change_pct = (prediction - last_price) / last_price * 100
                
                # Simple threshold rules
                if predicted_change_pct > 2.0:  # Strongly bullish
                    signal = {
                        "action": "BUY",
                        "reason": f"Predicted price increase of {predicted_change_pct:.2f}%",
                        "target_price": last_price * 1.03,  # 3% profit target
                        "stop_loss": last_price * 0.98    # 2% stop loss
                    }
                elif predicted_change_pct < -2.0:  # Strongly bearish
                    signal = {
                        "action": "SELL",
                        "reason": f"Predicted price decrease of {predicted_change_pct:.2f}%",
                        "target_price": last_price * 0.97,  # 3% profit target for short
                        "stop_loss": last_price * 1.02    # 2% stop loss for short
                    }
                else:  # Neutral
                    signal = {
                        "action": "HOLD",
                        "reason": f"Predicted change of {predicted_change_pct:.2f}% within neutral range",
                        "target_price": None,
                        "stop_loss": None
                    }
            
        # For classification models
        elif model_type.endswith("classifier"):
            # For binary classification (0=down, 1=up)
            if isinstance(prediction, (int, float)) and prediction in [0, 1]:
                if prediction == 1:
                    signal = {
                        "action": "BUY",
                        "reason": "Model predicts price increase",
                        "target_price": last_price * 1.03,
                        "stop_loss": last_price * 0.98
                    }
                else:
                    signal = {
                        "action": "SELL",
                        "reason": "Model predicts price decrease",
                        "target_price": last_price * 0.97,
                        "stop_loss": last_price * 1.02
                    }
            
            # For probabilistic output
            elif isinstance(prediction, dict):
                # Convert string keys to integers if needed
                prob_dict = {int(k) if k.isdigit() else k: v for k, v in prediction.items()}
                
                # Assuming 1 = up, 0 = down for binary classification
                if 1 in prob_dict and 0 in prob_dict:
                    up_prob = prob_dict[1]
                    down_prob = prob_dict[0]
                    
                    if up_prob > 0.65:  # Strong confidence in up movement
                        signal = {
                            "action": "BUY",
                            "reason": f"High confidence up prediction ({up_prob:.2f})",
                            "confidence": up_prob,
                            "target_price": last_price * 1.03,
                            "stop_loss": last_price * 0.98
                        }
                    elif down_prob > 0.65:  # Strong confidence in down movement
                        signal = {
                            "action": "SELL",
                            "reason": f"High confidence down prediction ({down_prob:.2f})",
                            "confidence": down_prob,
                            "target_price": last_price * 0.97,
                            "stop_loss": last_price * 1.02
                        }
                    else:
                        signal = {
                            "action": "HOLD",
                            "reason": "No strong directional signal",
                            "up_probability": up_prob,
                            "down_probability": down_prob
                        }
    except Exception as e:
        logger.error(f"Error generating trading signal for {symbol}: {str(e)}", exc_info=True)
        signal = {
            "action": "HOLD",
            "reason": f"Error generating signal: {str(e)}"
        }
    
    return signal

@router.post("/run", response_model=DailyPredictionResponse)
async def run_daily_pipeline(
    request: DailyPredictionRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Manually trigger the daily prediction pipeline
    
    This endpoint allows you to run the full daily pipeline:
    1. Update market data (optional)
    2. Generate predictions for specified model and symbols
    3. Generate trading signals
    
    The main processing happens in the background to avoid timeout issues.
    """
    try:
        # 1. Validate model exists
        repo = ModelRepository(db)
        model_record = repo.get_model(request.model_id)
        
        if not model_record:
            raise HTTPException(status_code=404, detail=f"Model with ID {request.model_id} not found")
        
        # 2. Update data if requested
        if request.update_data:
            logger.info("Starting data update process")
            background_tasks.add_task(update_all_data, db)
            data_update_status = "Data update scheduled in background"
        else:
            data_update_status = "Data update skipped"
        
        # 3. Schedule prediction generation
        logger.info(f"Scheduling prediction generation for model {request.model_id} and {len(request.symbols)} symbols")
        background_tasks.add_task(generate_daily_predictions, request.model_id, request.symbols, db)
        
        return {
            "success": True,
            "message": "Daily prediction pipeline started",
            "data_update_status": data_update_status,
            "model_id": request.model_id,
            "symbols": request.symbols,
            "scheduled_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting daily pipeline: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting daily pipeline: {str(e)}")

@router.get("/last-prediction/{model_id}")
async def get_last_prediction(model_id: int, db: Session = Depends(get_db)):
    """
    Get the most recent daily prediction for a specific model
    """
    try:
        # Find the latest prediction file
        prediction_dir = "predictions"
        if not os.path.exists(prediction_dir):
            return {"success": False, "message": "No predictions directory found"}
        
        # List all prediction files for this model
        model_files = [f for f in os.listdir(prediction_dir) 
                       if f.startswith(f"daily_{model_id}_") and f.endswith(".json")]
        
        if not model_files:
            return {"success": False, "message": f"No prediction files found for model {model_id}"}
        
        # Sort by date (which is part of the filename)
        model_files.sort(reverse=True)
        latest_file = os.path.join(prediction_dir, model_files[0])
        
        # Load the file
        with open(latest_file, 'r') as f:
            prediction_data = json.load(f)
        
        return {
            "success": True,
            "predictions": prediction_data,
            "file": latest_file
        }
        
    except Exception as e:
        logger.error(f"Error retrieving last prediction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving last prediction: {str(e)}")

@router.get("/signals/{model_id}", response_model=TradingSignalResponse)
async def get_trading_signals(model_id: int, db: Session = Depends(get_db)):
    """
    Get the current trading signals for a specific model
    
    This endpoint returns only the actionable trading signals from the most recent predictions.
    """
    try:
        # Get the latest prediction data
        prediction_dir = "predictions"
        if not os.path.exists(prediction_dir):
            return {"success": False, "message": "No predictions directory found"}
        
        # List all prediction files for this model
        model_files = [f for f in os.listdir(prediction_dir) 
                       if f.startswith(f"daily_{model_id}_") and f.endswith(".json")]
        
        if not model_files:
            return {"success": False, "message": f"No prediction files found for model {model_id}"}
        
        # Sort by date (which is part of the filename)
        model_files.sort(reverse=True)
        latest_file = os.path.join(prediction_dir, model_files[0])
        
        # Load the file
        with open(latest_file, 'r') as f:
            prediction_data = json.load(f)
        
        # Extract signals by action type
        buy_signals = []
        sell_signals = []
        hold_signals = []
        
        for result in prediction_data.get("results", []):
            if "trading_signal" not in result:
                continue
                
            signal = result["trading_signal"]
            symbol = result["symbol"]
            
            # Add symbol to the signal for easier reference
            signal["symbol"] = symbol
            
            # Categorize by action
            if signal["action"] == "BUY":
                buy_signals.append(signal)
            elif signal["action"] == "SELL":
                sell_signals.append(signal)
            else:
                hold_signals.append(signal)
        
        return {
            "success": True,
            "model_id": model_id,
            "prediction_date": prediction_data.get("date"),
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals
        }
        
    except Exception as e:
        logger.error(f"Error retrieving trading signals: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving trading signals: {str(e)}")

# Function to schedule the daily pipeline
async def schedule_daily_pipeline(model_id: int, symbols: List[str], time_str: str = "16:30"):
    """
    Schedule the daily pipeline to run at a specific time
    
    Args:
        model_id: ID of the model to use for predictions
        symbols: List of symbols to generate predictions for
        time_str: Time to run the pipeline, in HH:MM format (default 16:30 - after market close)
    """
    try:
        # Parse the time
        hour, minute = map(int, time_str.split(':'))
        
        # Get database session
        # In a real implementation, you'd handle this differently for background tasks
        from financial_prediction_system.api.dependencies import get_db_connection
        db = next(get_db_connection())
        
        async def daily_job():
            # 1. Update all data
            await update_all_data(db)
            # 2. Generate predictions
            await generate_daily_predictions(model_id, symbols, db)
            logger.info(f"Daily pipeline completed for model {model_id}")
        
        # Schedule the job
        scheduler.add_job(
            daily_job,
            'cron',
            hour=hour,
            minute=minute,
            id=f"daily_pipeline_{model_id}"
        )
        
        logger.info(f"Scheduled daily pipeline for model {model_id} at {time_str}")
        
        # Start the scheduler if not already running
        if not scheduler.running:
            scheduler.start()
            
    except Exception as e:
        logger.error(f"Error scheduling daily pipeline: {str(e)}", exc_info=True)

@router.post("/schedule")
async def create_daily_schedule(
    model_id: int, 
    symbols: List[str], 
    time: str = "16:30",
    db: Session = Depends(get_db)
):
    """
    Create a scheduled daily pipeline that runs automatically at the specified time
    """
    try:
        # Validate model exists
        repo = ModelRepository(db)
        model_record = repo.get_model(model_id)
        
        if not model_record:
            raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
        
        # Schedule the pipeline
        await schedule_daily_pipeline(model_id, symbols, time)
        
        return {
            "success": True,
            "message": f"Daily pipeline scheduled for model {model_id} at {time}",
            "model_id": model_id,
            "symbols": symbols,
            "time": time
        }
        
    except Exception as e:
        logger.error(f"Error creating daily schedule: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating daily schedule: {str(e)}")

@router.delete("/schedule/{model_id}")
async def remove_daily_schedule(model_id: int):
    """
    Remove a scheduled daily pipeline for a specific model
    """
    try:
        job_id = f"daily_pipeline_{model_id}"
        
        # Check if the job exists
        if job_id in [job.id for job in scheduler.get_jobs()]:
            scheduler.remove_job(job_id)
            return {
                "success": True,
                "message": f"Removed scheduled pipeline for model {model_id}"
            }
        else:
            return {
                "success": False,
                "message": f"No scheduled pipeline found for model {model_id}"
            }
            
    except Exception as e:
        logger.error(f"Error removing daily schedule: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error removing daily schedule: {str(e)}") 