"""Main FastAPI application"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from financial_prediction_system.api.routes import models, predictions, backtests, daily_inference
from financial_prediction_system.api.schemas import ErrorResponse
from financial_prediction_system.logging_config import logger

# Create FastAPI app
app = FastAPI(
    title="Financial Prediction API",
    description="API for financial predictions using machine learning models",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Log when the API starts"""
    logger.info("Financial Prediction API starting up")

@app.on_event("shutdown")
async def shutdown_event():
    """Log when the API shuts down"""
    logger.info("Financial Prediction API shutting down")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    error_response = ErrorResponse(
        success=False,
        error=str(exc),
        error_code=500
    )
    return JSONResponse(status_code=500, content=error_response.dict())

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    error_response = ErrorResponse(
        success=False,
        error=exc.detail,
        error_code=exc.status_code
    )
    return JSONResponse(status_code=exc.status_code, content=error_response.dict())

# Register routers
app.include_router(models.router)
app.include_router(predictions.router)
app.include_router(backtests.router)
app.include_router(daily_inference.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the Financial Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Add model endpoints for non-interactive API exploration
@app.get("/models-api")
async def models_api_info():
    """Information about the models API endpoints"""
    return {
        "endpoints": [
            {
                "path": "/models/train",
                "method": "POST",
                "description": "Train a new prediction model"
            },
            {
                "path": "/models/",
                "method": "GET",
                "description": "List models with optional filtering"
            },
            {
                "path": "/models/{model_id}",
                "method": "GET",
                "description": "Get details of a specific model"
            },
            {
                "path": "/models/{model_id}",
                "method": "DELETE",
                "description": "Delete a model"
            },
            {
                "path": "/models/ensemble",
                "method": "POST",
                "description": "Create an ensemble of models"
            },
            {
                "path": "/models/ensemble/{ensemble_id}",
                "method": "GET",
                "description": "Get details of a specific ensemble"
            },
            {
                "path": "/models/available-models",
                "method": "GET",
                "description": "Get lists of available model types, target types, and feature sets"
            }
        ]
    }

# Add prediction endpoints for non-interactive API exploration
@app.get("/predictions-api")
async def predictions_api_info():
    """Information about the predictions API endpoints"""
    return {
        "endpoints": [
            {
                "path": "/predictions/",
                "method": "POST",
                "description": "Make a prediction using a trained model"
            },
            {
                "path": "/predictions/batch",
                "method": "POST",
                "description": "Make predictions for multiple symbols using the same model"
            },
            {
                "path": "/predictions/ensemble",
                "method": "POST",
                "description": "Make a prediction using an ensemble of models"
            }
        ]
    }

# Add daily inference endpoints for non-interactive API exploration
@app.get("/daily-api")
async def daily_api_info():
    """Information about the daily inference API endpoints"""
    return {
        "endpoints": [
            {
                "path": "/daily/run",
                "method": "POST",
                "description": "Manually trigger the daily prediction pipeline"
            },
            {
                "path": "/daily/schedule",
                "method": "POST",
                "description": "Schedule a daily prediction pipeline to run automatically"
            },
            {
                "path": "/daily/schedule/{model_id}",
                "method": "DELETE",
                "description": "Remove a scheduled daily pipeline for a model"
            },
            {
                "path": "/daily/last-prediction/{model_id}",
                "method": "GET",
                "description": "Get the most recent daily prediction for a model"
            },
            {
                "path": "/daily/signals/{model_id}",
                "method": "GET",
                "description": "Get the current trading signals for a model"
            }
        ]
    }

# Add backtest endpoints for non-interactive API exploration
@app.get("/backtests-api")
async def backtests_api_info():
    """Information about the backtests API endpoints"""
    return {
        "endpoints": [
            {
                "path": "/backtests/",
                "method": "POST",
                "description": "Create a new backtest for a model"
            },
            {
                "path": "/backtests/ensemble",
                "method": "POST",
                "description": "Create a new backtest for an ensemble"
            },
            {
                "path": "/backtests/",
                "method": "GET",
                "description": "List backtests with optional filtering"
            },
            {
                "path": "/backtests/{backtest_id}",
                "method": "GET",
                "description": "Get details of a specific backtest"
            }
        ]
    }
