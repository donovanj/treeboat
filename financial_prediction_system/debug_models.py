# Debug script to diagnose model registration issues
from financial_prediction_system.core.models import ModelFactory

# First, try importing all model modules explicitly
try:
    from financial_prediction_system.core.models.classification import RandomForestModel
    from financial_prediction_system.core.models.classification import SVMModel
    from financial_prediction_system.core.models.classification import GradientBoostingModel
    from financial_prediction_system.core.models.regression import LogisticRegressionModel
    from financial_prediction_system.core.models.regression import LSTMRegressor
    from financial_prediction_system.core.models.regression import TransformerRegressor
    print("All models imported successfully")
except Exception as e:
    print(f"Error importing models: {e}")

# Now check what's registered in the factory
print("\nRegistered models in ModelFactory:")
available_models = ModelFactory.get_available_models()
print(f"Number of models: {len(available_models)}")
for model_name in available_models:
    print(f"- {model_name}")

# Try to create each model type
for model_name in available_models:
    try:
        model = ModelFactory.create_model(model_name)
        print(f"Successfully created model of type: {model_name}, class: {model.__class__.__name__}")
    except Exception as e:
        print(f"Failed to create model of type: {model_name}, error: {e}")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from financial_prediction_system.data_loaders.loader_factory import DataLoaderFactory
from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.core.targets.target_builder import TargetBuilder

def debug_data_alignment_issue():
    """Debug function to identify why feature and target arrays have different lengths."""
    
    # Create a dummy database session (we're just testing data processing)
    db = None
    
    # Sample data parameters matching the problematic request
    symbol = "AAPL"
    train_start = datetime(2020, 1, 1).date()
    train_end = datetime(2021, 12, 31).date()
    
    print(f"Loading data for {symbol} from {train_start} to {train_end}")
    
    # Load data
    loader_type = "stock" 
    loader = DataLoaderFactory.get_loader(loader_type, db)
    
    try:
        data = loader.load_data(
            symbol=symbol,
            start_date=train_start,
            end_date=train_end
        )
        
        # Get extended date range for index data
        extended_start = train_start - timedelta(days=100)
        index_data = DataLoaderFactory.get_index_data(db, extended_start, train_end)
        
        print(f"Original data shape: {data.shape}")
        print(f"Original data date range: {data.index.min()} to {data.index.max()}")
        
        # 1. Build features - similar to the problematic request
        feature_builder = FeatureBuilder(data)
        feature_builder.add_feature_set("technical")
        feature_builder.add_feature_set("volume") 
        feature_builder.add_feature_set("market_regime", index_data=index_data)
        
        # 2. Build features
        features_df = feature_builder.build()
        print(f"Features shape: {features_df.shape}")
        print(f"Features date range: {features_df.index.min()} to {features_df.index.max()}")
        
        # 3. Build targets
        target_builder = TargetBuilder(data)
        target_builder.add_target_set("price")
        targets_df = target_builder.build()
        print(f"Targets shape: {targets_df.shape}")
        print(f"Targets date range: {targets_df.index.min()} to {targets_df.index.max()}")
        
        # 4. Check alignment
        print("\nChecking alignment:")
        features_dates = set(features_df.index)
        targets_dates = set(targets_df.index)
        
        only_in_features = features_dates - targets_dates
        only_in_targets = targets_dates - features_dates
        
        print(f"Dates only in features: {len(only_in_features)}")
        if only_in_features:
            print(f"Sample dates only in features: {sorted(list(only_in_features))[:5]}")
            
        print(f"Dates only in targets: {len(only_in_targets)}")
        if only_in_targets:
            print(f"Sample dates only in targets: {sorted(list(only_in_targets))[:5]}")
        
        # 5. Find solution: align data
        print("\nAligning data:")
        common_dates = features_dates.intersection(targets_dates)
        print(f"Common dates count: {len(common_dates)}")
        
        # Align both dataframes to common dates
        aligned_features = features_df.loc[list(common_dates)]
        aligned_targets = targets_df.loc[list(common_dates)]
        
        print(f"Aligned features shape: {aligned_features.shape}")
        print(f"Aligned targets shape: {aligned_targets.shape}")
        
        # 6. Report findings
        print("\nFindings:")
        if aligned_features.shape[0] == aligned_targets.shape[0]:
            print("✅ Successfully aligned features and targets")
            print(f"Aligned shape: {aligned_features.shape[0]} rows")
        else:
            print("❌ Failed to align features and targets")
        
        # 7. Check for potential causes
        print("\nPotential causes:")
        
        # NaN values in features
        nan_feature_count = features_df.isna().sum().sum()
        print(f"- NaN values in features: {nan_feature_count}")
        
        # NaN values in targets  
        nan_target_count = targets_df.isna().sum().sum()
        print(f"- NaN values in targets: {nan_target_count}")
        
        # Different lookback windows
        print(f"- Technical features may use lookback windows that create NaNs at the beginning")
        print(f"- Price target features may use forward-looking windows that create NaNs at the end")
        
    except Exception as e:
        print(f"Error during debugging: {str(e)}")

if __name__ == "__main__":
    print("Starting model data alignment debug")
    available_models = ModelFactory.get_available_models()
    print(f"Available models: {available_models}")
    
    debug_data_alignment_issue() 