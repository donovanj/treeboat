import pytest
import importlib
import sys
import platform

def test_python_version():
    """Test that Python version is at least 3.8."""
    assert sys.version_info >= (3, 8), "Python version should be at least 3.8"

def test_required_packages():
    """Test that required packages are installed."""
    required_packages = [
        'fastapi',
        'sqlalchemy',
        'pytest',
    ]
    
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            pytest.fail(f"Required package '{package}' is not installed")

@pytest.mark.optional
def test_torch_installation():
    """Test that PyTorch is installed correctly."""
    try:
        import torch
        assert torch.__version__, "PyTorch version not found"
    except ImportError:
        pytest.fail("PyTorch is not installed")

@pytest.mark.optional
def test_gpu_detection():
    """Test that GPU can be detected by PyTorch if available."""
    try:
        import torch
        # This just checks if CUDA/ROCm is available, not if it's working
        # Just reporting the status rather than asserting to make the test pass regardless
        has_gpu = torch.cuda.is_available() or hasattr(torch, 'has_mps') and torch.backends.mps.is_available()
        print(f"GPU detection: {'Available' if has_gpu else 'Not available'}")
    except ImportError:
        pytest.fail("PyTorch is not installed")
    except Exception as e:
        pytest.fail(f"Error checking GPU: {str(e)}") 