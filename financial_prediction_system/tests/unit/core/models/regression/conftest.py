"""Fixtures for regression model tests."""

import pytest
import numpy as np
import torch

from financial_prediction_system.core.models.factory import ModelFactory


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    batch_size = 16
    seq_len = 10
    features = 8
    
    # Generate features and targets
    X = np.random.rand(batch_size, seq_len, features).astype(np.float32)
    y = np.random.rand(batch_size, 1).astype(np.float32)
    
    return X, y


@pytest.fixture
def model_factory_reset():
    """Reset the model factory registry before and after tests."""
    # Store original registry
    original_registry = ModelFactory._models.copy()
    
    # Reset registry for test
    ModelFactory._models = {}
    
    yield
    
    # Restore original registry
    ModelFactory._models = original_registry 