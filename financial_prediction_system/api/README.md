# Financial Prediction System API

This API provides access to the Financial Prediction System's machine learning models and prediction capabilities.

## Overview

The Financial Prediction System API is built with FastAPI and provides endpoints for:

1. **Models**: Training, retrieving, and managing machine learning models
2. **Predictions**: Making predictions using trained models and ensembles
3. **Backtests**: Evaluating model performance with historical backtests

## Logging

The API uses a centralized logging system defined in `financial_prediction_system/logging_config.py`. All logs are written to both:

1. Console output (INFO level and above)
2. Log files in the `logs` directory (DEBUG level and above)

Log files rotate daily with a 7-day retention period.

## API Endpoints

### Models API

- `POST /models/train`: Train a new prediction model
- `GET /models/`: List models with optional filtering
- `GET /models/{model_id}`: Get details of a specific model
- `DELETE /models/{model_id}`: Delete a model
- `POST /models/ensemble`: Create an ensemble of models
- `GET /models/ensemble/{ensemble_id}`: Get details of a specific ensemble
- `GET /models/available-models`: Get lists of available model types, targets, and features

### Predictions API

- `POST /predictions/`: Make a prediction using a trained model
- `POST /predictions/batch`: Make predictions for multiple symbols using the same model
- `POST /predictions/ensemble`: Make a prediction using an ensemble of models

### Backtests API

- `POST /backtests/`: Create a new backtest for a model
- `POST /backtests/ensemble`: Create a new backtest for an ensemble
- `GET /backtests/`: List backtests with optional filtering
- `GET /backtests/{backtest_id}`: Get details of a specific backtest

## Data Models

The API works with the following data models:

1. **Models**: ML models for prediction, stored in the database with metadata
2. **Ensembles**: Collections of models combined using ensemble methods
3. **Backtests**: Historical performance evaluations of models and ensembles

## Model Workflow

1. Select a target (price, volatility, etc.)
2. Choose feature sets to use
3. Select a model type (random forest, gradient boosting, etc.)
4. Train the model
5. Evaluate model performance with backtests
6. Use models for predictions
7. Combine models into ensembles

## Database Schema

The API uses a PostgreSQL database with the following tables:

- `ml_model_results`: Stores model metadata and performance metrics
- `ml_feature_importance`: Stores feature importance for models
- `ml_ensemble_models`: Stores ensemble metadata
- `ml_ensemble_model_mappings`: Maps models to ensembles
- `ml_backtest_results`: Stores backtest results for models and ensembles

## Usage Examples

### Training a Model

```json
POST /models/train

{
  "symbol": "SPX",
  "model_type": "random_forest_classifier",
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
```

### Making a Prediction

```json
POST /predictions/

{
  "model_id": 1,
  "symbol": "SPX",
  "prediction_date": "2023-04-15",
  "custom_features": {
    "market_regime": "bullish"
  }
}
```

### Creating a Backtest

```json
POST /backtests/

{
  "model_id": 1,
  "symbol": "SPX",
  "start_date": "2022-01-01",
  "end_date": "2022-12-31",
  "parameters": {
    "threshold": 0.75,
    "trade_size": 100000
  }
}
```

## Setup and Installation

1. Install dependencies from requirements.txt
2. Configure database connection in .env file
3. Run database migrations
4. Start the API server with `uvicorn financial_prediction_system.api.main:app --reload`

## Security

The API includes authentication and authorization mechanisms to ensure secure access to the financial prediction capabilities.

## Documentation

Full API documentation is available at the `/docs` endpoint when the server is running.