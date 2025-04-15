import os
import pytest
import importlib

def test_project_structure():
    """Test that the project structure is set up correctly."""
    # Define expected directories
    expected_directories = [
        "api",
        "api/routes",
        "api/schemas",
        "config",
        "core",
        "core/data",
        "core/ensemble",
        "core/evaluation",
        "core/features",
        "core/models",
        "core/targets",
        "infrastructure",
        "infrastructure/database",
        "infrastructure/repositories",
        "infrastructure/services",
        "models",
        "pipelines",
        "tests",
        "tests/unit",
        "tests/integration",
        "utils"
    ]
    
    # Check if all expected directories exist
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    for directory in expected_directories:
        dir_path = os.path.join(base_path, directory)
        assert os.path.isdir(dir_path), f"Directory not found: {directory}"

def test_main_module_imports():
    """Test that the main modules can be imported."""
    modules_to_test = [
        "financial_prediction_system.api.main",
        "financial_prediction_system.api.dependencies",
        "financial_prediction_system.infrastructure.database.connection"
    ]
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
        except ImportError as e:
            pytest.fail(f"Failed to import {module}: {e}")

def test_api_endpoints():
    """Test that the API endpoints are defined."""
    from financial_prediction_system.api.main import app
    
    # Check if root endpoint exists
    assert any(route.path == "/" for route in app.routes), "Root endpoint not found" 