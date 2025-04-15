"""API routes for predictions"""

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from financial_prediction_system.api.dependencies import get_db
from financial_prediction_system.api.schemas import (
    PredictionRequest, EnsemblePredictionRequest, BatchPredictionRequest,
    PredictionResponse, BatchPredictionResponse, EnsemblePredictionResponse
)
from financial_prediction_system.infrastructure.database.model_store import ModelRepository
from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
from financial_prediction_system.core.models.base import PredictionModel

router = APIRouter(prefix="/predictions", tags=["predictions"])

def load_model(model_path: str) -> PredictionModel:
    """Load a model from disk"""
    # This is a placeholder for the actual model loading logic
    # In a real implementation, we would deserialize the model from disk
    import pickle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@router.post("/", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest, db: Session = Depends(get_db)):
    """Make a prediction using a trained model"""
    repo = ModelRepository(db)
    model_record = repo.get_model(request.model_id)
    
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {request.model_id} not found")
    
    if not model_record.model_path:
        raise HTTPException(status_code=400, detail=f"Model with ID {request.model_id} has no saved model file")
    
    try:
        # 1. Load the model
        model = load_model(model_record.model_path)
        
        # 2. Load data for the symbol
        loader = DataLoaderFactory.create_loader(request.symbol)
        
        # Set end_date to prediction_date or current date if not specified
        end_date = request.prediction_date or datetime.now()
        
        # Load some historical data to create features
        # Adjust the start_date based on the feature requirements
        # For simplicity, we'll use 100 days of history
        start_date = datetime(end_date.year, end_date.month, end_date.day) - datetime.timedelta(days=100)
        
        data = loader.load_data(start_date=start_date, end_date=end_date)
        
        # 3. Prepare features
        feature_builder = FeatureBuilder(data)
        for feature_set in model_record.feature_sets:
            feature_builder.add_feature_set(feature_set)
        
        # Apply any custom features from the request
        if request.custom_features:
            for feature_name, feature_value in request.custom_features.items():
                # In a real implementation, we would have a more robust way to handle custom features
                data[feature_name] = feature_value
        
        features_df = feature_builder.build()
        
        # 4. Make prediction
        # Use the last row of features for the prediction
        prediction = model.predict(features_df.iloc[-1:])  
        
        # 5. Prepare response
        prediction_value = prediction[0]  # Extract the scalar value from the prediction array
        
        # For classification models, convert to probabilities if needed
        if model_record.model_type.endswith("classifier"):
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_df.iloc[-1:])[0]
                prediction_value = {str(i): float(p) for i, p in enumerate(probabilities)}
        
        # 6. Get feature contributions if available
        feature_contributions = None
        if hasattr(model, "feature_contributions"):
            feature_contributions = model.feature_contributions(features_df.iloc[-1:])
            feature_contributions = {str(k): float(v) for k, v in feature_contributions.items()}
        
        # 7. Confidence score (if available)
        confidence = None
        if hasattr(model, "predict_proba") and not isinstance(prediction_value, dict):
            probas = model.predict_proba(features_df.iloc[-1:])[0]
            confidence = float(max(probas))
        
        return {
            "success": True,
            "result": {
                "symbol": request.symbol,
                "prediction_date": end_date,
                "prediction": prediction_value,
                "confidence": confidence,
                "feature_contributions": feature_contributions
            },
            "model_id": model_record.id,
            "model_type": model_record.model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@router.post("/batch", response_model=BatchPredictionResponse)
async def make_batch_prediction(request: BatchPredictionRequest, db: Session = Depends(get_db)):
    """Make predictions for multiple symbols using the same model"""
    repo = ModelRepository(db)
    model_record = repo.get_model(request.model_id)
    
    if not model_record:
        raise HTTPException(status_code=404, detail=f"Model with ID {request.model_id} not found")
    
    if not model_record.model_path:
        raise HTTPException(status_code=400, detail=f"Model with ID {request.model_id} has no saved model file")
    
    try:
        # 1. Load the model
        model = load_model(model_record.model_path)
        
        # Set end_date to prediction_date or current date if not specified
        end_date = request.prediction_date or datetime.now()
        
        # Load some historical data to create features
        start_date = datetime(end_date.year, end_date.month, end_date.day) - datetime.timedelta(days=100)
        
        results = []
        
        for symbol in request.symbols:
            # 2. Load data for each symbol
            loader = DataLoaderFactory.create_loader(symbol)
            data = loader.load_data(start_date=start_date, end_date=end_date)
            
            # 3. Prepare features
            feature_builder = FeatureBuilder(data)
            for feature_set in model_record.feature_sets:
                feature_builder.add_feature_set(feature_set)
            
            # Apply any custom features from the request
            custom_features = request.custom_features.get(symbol, {}) if request.custom_features else {}
            for feature_name, feature_value in custom_features.items():
                data[feature_name] = feature_value
            
            features_df = feature_builder.build()
            
            # Skip symbols with insufficient data
            if features_df.empty:
                continue
            
            # 4. Make prediction
            prediction = model.predict(features_df.iloc[-1:])
            
            # 5. Prepare result
            prediction_value = prediction[0]  # Extract the scalar value
            
            # For classification models, convert to probabilities if needed
            if model_record.model_type.endswith("classifier"):
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_df.iloc[-1:])[0]
                    prediction_value = {str(i): float(p) for i, p in enumerate(probabilities)}
            
            # 6. Get feature contributions if available
            feature_contributions = None
            if hasattr(model, "feature_contributions"):
                feature_contributions = model.feature_contributions(features_df.iloc[-1:])
                feature_contributions = {str(k): float(v) for k, v in feature_contributions.items()}
            
            # 7. Confidence score (if available)
            confidence = None
            if hasattr(model, "predict_proba") and not isinstance(prediction_value, dict):
                probas = model.predict_proba(features_df.iloc[-1:])[0]
                confidence = float(max(probas))
            
            results.append({
                "symbol": symbol,
                "prediction_date": end_date,
                "prediction": prediction_value,
                "confidence": confidence,
                "feature_contributions": feature_contributions
            })
        
        return {
            "success": True,
            "results": results,
            "model_id": model_record.id,
            "model_type": model_record.model_type
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making batch prediction: {str(e)}")

@router.post("/ensemble", response_model=EnsemblePredictionResponse)
async def make_ensemble_prediction(request: EnsemblePredictionRequest, db: Session = Depends(get_db)):
    """Make a prediction using an ensemble of models"""
    repo = ModelRepository(db)
    ensemble_record = repo.get_ensemble(request.ensemble_id)
    
    if not ensemble_record:
        raise HTTPException(status_code=404, detail=f"Ensemble with ID {request.ensemble_id} not found")
    
    if not ensemble_record.model_path:
        raise HTTPException(status_code=400, detail=f"Ensemble with ID {request.ensemble_id} has no saved model file")
    
    try:
        # 1. Load the ensemble model
        ensemble_model = load_model(ensemble_record.model_path)
        
        # 2. Load models included in the ensemble
        model_ids = [mapping.model_id for mapping in ensemble_record.model_mappings]
        models = []
        
        for model_id in model_ids:
            model_record = repo.get_model(model_id)
            if model_record and model_record.model_path:
                model = load_model(model_record.model_path)
                models.append((model_id, model, model_record.model_type))
        
        # 3. Load data for the symbol
        loader = DataLoaderFactory.create_loader(request.symbol)
        
        # Set end_date to prediction_date or current date if not specified
        end_date = request.prediction_date or datetime.now()
        
        # Load some historical data to create features
        start_date = datetime(end_date.year, end_date.month, end_date.day) - datetime.timedelta(days=100)
        
        data = loader.load_data(start_date=start_date, end_date=end_date)
        
        # 4. Make predictions with each model
        base_predictions = []
        
        for model_id, model, model_type in models:
            # Create features for this model
            model_record = repo.get_model(model_id)
            feature_builder = FeatureBuilder(data)
            
            for feature_set in model_record.feature_sets:
                feature_builder.add_feature_set(feature_set)
            
            # Apply any custom features from the request
            if request.custom_features:
                for feature_name, feature_value in request.custom_features.items():
                    data[feature_name] = feature_value
            
            features_df = feature_builder.build()
            
            # Skip if insufficient data
            if features_df.empty:
                continue
            
            # Make prediction
            prediction = model.predict(features_df.iloc[-1:])
            
            # Prepare result
            prediction_value = prediction[0]  # Extract the scalar value
            
            # For classification models, convert to probabilities if needed
            if model_type.endswith("classifier"):
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_df.iloc[-1:])[0]
                    prediction_value = {str(i): float(p) for i, p in enumerate(probabilities)}
            
            base_predictions.append({
                "model_id": model_id,
                "model_type": model_type,
                "prediction": prediction_value
            })
        
        # 5. Make ensemble prediction
        # In a real implementation, we would use the ensemble model properly
        # For this simplification, we'll just use the first model's prediction
        if base_predictions:
            ensemble_prediction = base_predictions[0]["prediction"]
        else:
            raise HTTPException(status_code=500, detail="No valid base models available for ensemble prediction")
        
        return {
            "success": True,
            "result": {
                "symbol": request.symbol,
                "prediction_date": end_date,
                "prediction": ensemble_prediction,
                "confidence": None,  # Simplified example
                "feature_contributions": None  # Simplified example
            },
            "ensemble_id": ensemble_record.id,
            "base_predictions": base_predictions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making ensemble prediction: {str(e)}")
