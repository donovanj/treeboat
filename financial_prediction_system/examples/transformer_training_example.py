"""
Transformer Model Training Example

This example demonstrates how to use the training pipeline with the TransformerRegressor model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from financial_prediction_system.pipelines.training import train_model
from financial_prediction_system.core.features.feature_builder import FeatureBuilder
from financial_prediction_system.core.models.factory import ModelFactory


def load_sample_data():
    """Load or generate sample data for demonstration."""
    # For this example, we'll generate synthetic time series data
    # In a real scenario, you would load your financial data
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    
    # Generate price series with trend and seasonality
    t = np.arange(len(dates))
    trend = 0.01 * t
    seasonality = 0.1 * np.sin(2 * np.pi * t / 365)
    noise = 0.05 * np.random.randn(len(t))
    
    price = 100 + trend + seasonality + noise
    returns = np.diff(price) / price[:-1]
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates[1:],
        'price': price[1:],
        'returns': returns,
        'volume': np.random.randint(1000, 10000, size=len(dates)-1),
    })
    
    # Add some features to demonstrate the pipeline
    data['returns_lag1'] = data['returns'].shift(1)
    data['returns_lag2'] = data['returns'].shift(2)
    data['returns_lag3'] = data['returns'].shift(3)
    data['volume_lag1'] = data['volume'].shift(1)
    data['volume_lag2'] = data['volume'].shift(2)
    
    # Add target (next day return)
    data['target_next_return'] = data['returns'].shift(-1)
    
    # Drop NaN values
    data = data.dropna()
    
    return data


def main():
    """Run the transformer model training example."""
    # Load sample data
    data = load_sample_data()
    print(f"Loaded data with shape: {data.shape}")
    
    # Define feature configuration
    feature_config = {
        'use_technical_features': True,
        'window_sizes': [5, 10, 20],
        'use_volume_features': True,
        'use_date_features': True,
        'use_price_action_features': True,  # Added price action features
        'use_market_regime_features': False  # Not using market regime features for this example
    }
    
    # Define model parameters
    model_params = {
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'output_size': 1,
        'max_seq_len': 100,
        'learning_rate': 0.0001,
        'batch_size': 32,
        'num_epochs': 20,
        'sequence_length': 20,
        'prediction_length': 1
    }
    
    # Create save path
    save_dir = 'models'
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'transformer_model_{timestamp}.pt')
    
    # Train the model
    results = train_model(
        data=data,
        target_col='target_next_return',
        model_type='transformer',
        model_params=model_params,
        feature_config=feature_config,
        save_path=save_path
    )
    
    # Extract results
    model = results['model']
    feature_pipeline = results['feature_pipeline']
    training_history = results['training_history']
    
    # Print evaluation metrics
    train_losses = training_history.get('train_losses', [])
    val_losses = training_history.get('val_losses', [])
    
    print("\nTraining Results:")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    history_plot_path = os.path.join(save_dir, f'training_history_{timestamp}.png')
    plt.savefig(history_plot_path)
    print(f"Training history plot saved to: {history_plot_path}")
    
    # Make predictions on test data
    # In a real scenario, you would use separate test data
    test_data = data.iloc[-100:].copy()
    
    # Transform test data using the feature pipeline
    test_features = feature_pipeline.transform_new_data(test_data)
    
    # Use the pipeline to prepare sequences for prediction
    sequence_length = model_params['sequence_length']
    test_sequences, _ = feature_pipeline.prepare_sequence_data(
        test_features, 
        sequence_length=sequence_length
    )
    
    # Get predictions
    predictions = model.predict(test_sequences)
    
    # Compare with actual values
    actual = test_data['target_next_return'].iloc[sequence_length-1:].values[:len(predictions)]
    
    # Calculate metrics
    mse = np.mean((predictions.flatten() - actual)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions.flatten() - actual))
    direction_accuracy = np.mean((np.sign(predictions.flatten()) == np.sign(actual)).astype(int))
    
    print("\nTest Results:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predictions.flatten(), label='Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Return')
    plt.title('Transformer Model Predictions')
    plt.legend()
    plt.grid(True)
    
    predictions_plot_path = os.path.join(save_dir, f'predictions_{timestamp}.png')
    plt.savefig(predictions_plot_path)
    print(f"Predictions plot saved to: {predictions_plot_path}")
    
    print("\nExample completed successfully!")


if __name__ == '__main__':
    main() 