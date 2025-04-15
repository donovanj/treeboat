"""
Example for using the target builder with the training pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.core.targets.target_builder import TargetBuilder, TargetDirector
from financial_prediction_system.core.targets.model_target_pipeline import ModelTargetPipeline
from financial_prediction_system.pipelines.training import train_model

def generate_sample_data(n_samples=500):
    """Generate sample price data for demonstration"""
    
    # Create a date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate price data with some trend and seasonality
    trend = np.linspace(0, 10, n_samples)
    seasonality = 2 * np.sin(np.linspace(0, 15, n_samples))
    noise = np.random.normal(0, 1, n_samples)
    
    # Create base price
    base_price = 100 + trend + seasonality + noise.cumsum() * 0.5
    
    # Generate OHLC data with some realistic properties
    high = base_price + np.random.normal(0, 1, n_samples).cumsum() * 0.2 + np.abs(noise) * 0.5
    low = base_price - np.random.normal(0, 1, n_samples).cumsum() * 0.2 - np.abs(noise) * 0.5
    open_price = base_price.copy()
    np.random.shuffle(open_price)
    close = base_price.copy()
    
    # Create volume
    volume = np.random.gamma(5, 100000, n_samples) * (1 + np.abs(noise) * 0.1)
    
    # Create a benchmark that's correlated but not identical
    benchmark = base_price * 0.8 + np.random.normal(0, 1, n_samples).cumsum() * 1.5 + 50
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
        'benchmark': benchmark
    })
    
    # Set date as index
    data.set_index('date', inplace=True)
    
    return data

def feature_builder_example(data):
    """Example of using the FeatureBuilder"""
    
    # Create a feature builder
    builder = FeatureBuilder(data)
    
    # Add technical indicators
    builder.add_feature_set('technical')
    
    # Add volume features
    builder.add_feature_set('volume')
    
    # Add date features
    builder.add_feature_set('date')
    
    # Build the features
    features = builder.build()
    
    print(f"Generated {features.shape[1]} features from the data.")
    print("First few feature names:", features.columns[:5].tolist())
    
    return features

def target_builder_example(data):
    """Example of using the TargetBuilder"""
    
    # Create a target builder
    builder = TargetBuilder(data)
    
    # Add price return targets
    builder.add_target_set('price', periods=[1, 5, 10, 20], return_type='log')
    
    # Add directional targets
    builder.add_target_set('price', periods=[1, 5], return_type='direction')
    
    # Add volatility targets
    builder.add_target_set('volatility', 
                          periods=[5, 10], 
                          volatility_type='realized',
                          classification=True)
    
    # Add alpha targets if benchmark exists
    if 'benchmark' in data.columns:
        builder.add_target_set('alpha',
                              periods=[5, 10],
                              alpha_type='information_ratio')
    
    # Build the targets
    targets = builder.build()
    
    print(f"Generated {targets.shape[1]} target variables.")
    print("Target variables:", targets.columns.tolist())
    
    return targets

def target_director_example(data):
    """Example of using the TargetDirector"""
    
    # Use the director to create a standard set of price targets
    price_targets = TargetDirector.create_price_return_targets(data)
    print(f"Generated {price_targets.shape[1]} price return targets.")
    
    # Use the director to create a comprehensive set of trading strategy targets
    trading_targets = TargetDirector.create_trading_strategy_targets(data)
    print(f"Generated {trading_targets.shape[1]} trading strategy targets.")
    
    # Use the director to create volatility prediction targets
    volatility_targets = TargetDirector.create_volatility_prediction_targets(data)
    print(f"Generated {volatility_targets.shape[1]} volatility prediction targets.")
    
    return {
        'price': price_targets,
        'trading': trading_targets,
        'volatility': volatility_targets
    }

def model_target_pipeline_example(data):
    """Example of using the ModelTargetPipeline"""
    
    # Configure the target pipeline for price targets
    price_config = {
        'target_type': 'price',
        'periods': [1, 5, 10],
        'return_type': 'log',
        'primary_target': 'log_return_5d'
    }
    
    # Create the pipeline
    price_pipeline = ModelTargetPipeline(config=price_config)
    price_targets = price_pipeline.prepare_targets(data)
    print(f"Generated {price_targets.shape[1]} price targets.")
    
    # Get the primary target
    primary_target = price_pipeline.get_specific_target()
    print(f"Primary target: {primary_target.name}, shape: {primary_target.shape}")
    
    # Get regression metrics for the primary target
    metrics = price_pipeline.get_regression_metrics()
    print("Target metrics:", {k: v for k, v in metrics.items() if k != 'is_regression'})
    
    # Configure the target pipeline for volatility targets
    volatility_config = {
        'target_type': 'volatility',
        'periods': [5, 10],
        'volatility_type': 'realized',
        'classification': True,
        'primary_target': 'realized_vol_5d_class'
    }
    
    # Create the pipeline
    vol_pipeline = ModelTargetPipeline(config=volatility_config)
    vol_targets = vol_pipeline.prepare_targets(data)
    
    # Get classification metrics
    class_metrics = vol_pipeline.get_classification_metrics('realized_vol_5d_class')
    if class_metrics['is_classification']:
        print("Class distribution:", class_metrics['class_proportions'])
    
    return {
        'price_pipeline': price_pipeline,
        'price_targets': price_targets,
        'vol_pipeline': vol_pipeline,
        'vol_targets': vol_targets
    }

def training_pipeline_example(data):
    """Example of using the training pipeline with the target builder"""
    
    # Define feature configuration
    feature_config = {
        'use_technical_features': True,
        'use_volume_features': True,
        'use_date_features': True,
        'window_sizes': [5, 10, 20]
    }
    
    # Define target configuration
    target_config = {
        'target_type': 'price',
        'periods': [5],
        'return_type': 'log',
        'primary_target': 'log_return_5d'
    }
    
    # Define model parameters for a simple model (could be any model registered with the factory)
    model_params = {
        'learning_rate': 0.01,
        'n_estimators': 100,
        'max_depth': 5
    }
    
    # Train a model using the pipeline
    # Note: Using 'xgboost' as an example - you would need to register this model type
    # with the ModelFactory for this to work
    try:
        results = train_model(
            data=data,
            model_type='xgboost',  # Example model type
            model_params=model_params,
            feature_config=feature_config,
            target_config=target_config
        )
        
        print("Model training completed successfully.")
        
        # Access the components
        model = results['model']
        feature_pipeline = results['feature_pipeline']
        target_pipeline = results['target_pipeline']
        
        # Example of getting feature importance
        if hasattr(model, 'feature_importances_'):
            feature_cols = feature_pipeline.feature_columns
            importances = model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            print("\nTop 10 most important features:")
            for i in range(min(10, len(feature_cols))):
                print(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
                
    except Exception as e:
        print(f"Note: Training example requires model registration. Error: {e}")
        print("To use this example, make sure to register model types with ModelFactory.")

def plot_targets(data, targets):
    """Plot some example target variables against price"""
    
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Price and log returns
    plt.subplot(3, 1, 1)
    ax1 = plt.gca()
    ax1.plot(data.index, data['close'], 'b-', label='Close Price')
    ax1.set_ylabel('Price', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    if 'log_return_5d' in targets.columns:
        ax2.plot(targets.index, targets['log_return_5d'], 'r-', label='5-day Log Return')
        ax2.set_ylabel('Log Return', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Price and 5-day Log Return')
    
    # Plot 2: Direction targets
    if 'direction_5d' in targets.columns:
        plt.subplot(3, 1, 2)
        plt.plot(targets.index, targets['direction_5d'], 'g-', drawstyle='steps-post')
        plt.ylabel('Direction (0=Down, 1=Up)')
        plt.title('5-day Price Direction')
    
    # Plot 3: Volatility targets
    if 'realized_vol_5d' in targets.columns:
        plt.subplot(3, 1, 3)
        plt.plot(targets.index, targets['realized_vol_5d'], 'purple')
        plt.ylabel('Realized Volatility')
        plt.title('5-day Realized Volatility')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the examples"""
    
    # Generate sample data
    print("Generating sample data...")
    data = generate_sample_data()
    print(f"Generated data with shape: {data.shape}")
    print(data.head())
    print("\n" + "="*80 + "\n")
    
    # Run feature builder example
    print("Running feature builder example...")
    features = feature_builder_example(data)
    print("\n" + "="*80 + "\n")
    
    # Run target builder example
    print("Running target builder example...")
    targets = target_builder_example(data)
    print("\n" + "="*80 + "\n")
    
    # Run target director example
    print("Running target director example...")
    director_targets = target_director_example(data)
    print("\n" + "="*80 + "\n")
    
    # Run model target pipeline example
    print("Running model target pipeline example...")
    pipeline_results = model_target_pipeline_example(data)
    print("\n" + "="*80 + "\n")
    
    # Run training pipeline example (note: requires model registration)
    print("Running training pipeline example...")
    training_pipeline_example(data)
    print("\n" + "="*80 + "\n")
    
    # Plot some targets
    print("Plotting target variables...")
    plot_targets(data, targets)

if __name__ == "__main__":
    main() 