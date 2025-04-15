import os
import sys
import pytest

# Add the parent directory of financial_prediction_system to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # current test directory
project_root = os.path.abspath(os.path.join(current_dir, '../../../../..'))  # Go up to the parent dir of financial_prediction_system
sys.path.insert(0, project_root) 