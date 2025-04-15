# Financial Prediction System

A robust, modular, and systematic machine learning system for financial predictions.

## Project Setup

### Environment Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```bash
     source venv/bin/activate
     ```
4. Install dependencies:
   ```bash
   pip install -e .
   ```
   For development:
   ```bash
   pip install -e ".[dev]"
   ```
5. Copy `.env.example` to `.env` and update the values:
   ```bash
   cp .env.example .env
   ```

### Running the API

```bash
uvicorn financial_prediction_system.api.main:app --reload
```

The API will be available at http://localhost:8000.

## Logging

The system uses a centralized logging configuration defined in `logging_config.py`. All logs are written to:

1. Console output (INFO level and above)
2. Log files in the `logs` directory (DEBUG level and above)

Log files rotate daily with a 7-day retention period.

To use the logger in your code:

```python
from financial_prediction_system.logging_config import logger

# Log at different levels
logger.debug("Detailed debug information")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error message", exc_info=True)  # With exception stacktrace
```

## Project Structure

The project follows a clean architecture approach with clear separation of concerns:

```
financial_prediction_system/
├── api/  # FastAPI application
│   ├── routes/  # API endpoints
│   │   ├── predictions.py  # Prediction endpoints
│   │   ├── models.py  # Model management endpoints
│   │   └── backtests.py  # Backtest endpoints
│   ├── schemas/  # Pydantic models (renamed from models to avoid confusion)
│   ├── dependencies.py  # Dependency injection
│   └── main.py  # Application entrypoint
├── core/  # Business logic
│   ├── models/  # ML model implementations
│   │   ├── base.py  # Base model interfaces
│   │   ├── factory.py  # Model factory pattern implementation
│   │   ├── registry.py  # Model registry & versioning
│   │   ├── regression/  # Regression models
│   │   │   ├── lstm.py  # LSTM implementation
│   │   │   ├── transformer.py  # Transformer implementation
│   │   │   └── neural_ode.py  # Neural ODE implementation
│   │   ├── classification/  # Classification models
│   │   │   ├── cnn.py  # CNN for pattern recognition
│   │   │   ├── xgboost_nn.py  # Hybrid XGBoost+NN model
│   │   │   └── transformer_classifier.py  # Transformer classifier
│   │   └── specialized/  # Specialized financial models
│   │       ├── deepar.py  # Probabilistic forecasting
│   │       ├── gnn.py  # Graph Neural Networks
│   │       └── reinforcement.py  # RL models
│   ├── features/  # Feature engineering
│   │   ├── technical.py  # Technical indicators
│   │   ├── fundamental.py  # Fundamental features
│   │   ├── sentiment.py  # Sentiment analysis
│   │   └── market.py  # Market/macro features
│   ├── targets/  # Target variable definitions
│   │   ├── price_targets.py  # Price-based targets
│   │   ├── volatility_targets.py  # Volatility-based targets
│   │   └── alpha_targets.py  # Alpha/relative performance targets
│   ├── ensemble/  # Model composition
│   │   ├── stacking.py  # Stacking ensemble
│   │   ├── boosting.py  # Boosting ensemble
│   │   ├── bagging.py  # Bagging ensemble
│   │   └── weighted.py  # Weighted ensemble
│   └── evaluation/  # Performance evaluation
│       ├── metrics.py  # Financial metrics
│       ├── backtesting.py  # Backtesting framework
│       └── cross_validation.py  # Time series cross-validation
├── infrastructure/  # External interfaces
│   ├── database/  # Database connections
│   │   ├── timeseries_db.py  # Time series database
│   │   └── model_store.py  # Model storage
│   ├── repositories/  # Data access
│   │   ├── market_data.py  # Market data repository
│   │   ├── features.py  # Feature repository
│   │   └── models.py  # Model repository
│   ├── services/  # External services
│   │   ├── data_providers.py  # Market data providers
│   │   ├── trading.py  # Trading API interfaces
│   │   └── notification.py  # Notification services
│   └── streaming/  # Real-time data processing
│       ├── kafka.py  # Kafka integration
│       └── websocket.py  # WebSocket handlers
├── utils/  # Utility functions
│   ├── preprocessing.py  # Data preprocessing
│   ├── visualization.py  # Data visualization
│   └── logging.py  # Logging helper functions
├── logging_config.py  # Centralized logging configuration
├── pipelines/  # Training and prediction pipelines
│   ├── training.py  # Model training pipeline
│   ├── prediction.py  # Prediction pipeline
│   └── optimization.py  # Hyperparameter optimization
├── config/  # Configuration files
│   ├── models.py  # Model configurations
│   └── app.py  # Application settings
├── tests/  # Test suite
│   ├── unit/  # Unit tests
│   ├── integration/  # Integration tests
│   └── conftest.py  # Test fixtures
├── .env  # Environment variables
└── pyproject.toml  # Project dependencies
```

## Testing

Run the tests with pytest:

```bash
pytest
```

With coverage:

```bash
pytest --cov=financial_prediction_system
``` 