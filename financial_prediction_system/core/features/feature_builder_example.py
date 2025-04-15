"""
Example usage of FeatureBuilder with dimensionality reduction

This module demonstrates how to use the FeatureBuilder with dimensionality reduction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from feature_builder import (
    FeatureBuilder, 
    FeatureDirector, 
    FeatureSelector,
    PCAReducer,
    KMeansReducer
)

def create_sample_data(n_rows=500):
    """Create sample financial data for demonstration"""
    # Create date range
    dates = [datetime.today() - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()  # Oldest to newest
    
    # Generate price data with some pattern
    np.random.seed(42)  # For reproducibility
    base_price = 100
    trend = np.linspace(0, 30, n_rows)  # Upward trend
    noise = np.random.normal(0, 5, n_rows)  # Random noise
    sine_pattern = 10 * np.sin(np.linspace(0, 10, n_rows))  # Cyclical pattern
    
    price = base_price + trend + noise + sine_pattern
    
    # Generate volume data
    base_volume = 10000
    volume_noise = np.random.normal(0, 2000, n_rows)
    volume_trend = np.linspace(0, 5000, n_rows)
    volume = base_volume + volume_noise + volume_trend
    
    # Create additional features
    high = price + np.random.uniform(1, 3, n_rows)
    low = price - np.random.uniform(1, 3, n_rows)
    open_price = price - np.random.normal(0, 2, n_rows)
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': price,
        'volume': volume,
    }, index=dates)
    
    return df

def main():
    """Main function to demonstrate feature builder"""
    print("Creating sample data...")
    data = create_sample_data(500)
    print(f"Sample data shape: {data.shape}")
    print(data.head())
    
    # Basic feature building
    print("\n1. Basic Feature Building")
    builder = FeatureBuilder(data)
    features = (builder
                .add_technical_features()
                .add_volume_features()
                .add_date_features()
                .build())
    
    print(f"Generated features shape: {features.shape}")
    print("Feature columns:")
    print(features.columns.tolist())
    print("\nFeature sample:")
    print(features.head())
    
    # Using the director
    print("\n2. Using FeatureDirector")
    director = FeatureDirector(FeatureBuilder(data))
    technical_features = director.build_technical_features()
    print(f"Technical features shape: {technical_features.shape}")
    
    # PCA dimensionality reduction
    print("\n3. PCA Dimensionality Reduction")
    # Create a new builder to avoid using previously computed features
    builder = FeatureBuilder(data)
    pca_features = (builder
                    .add_technical_features()
                    .add_volume_features()
                    .add_pca_reduction(n_components=5)
                    .build())
    
    print(f"PCA reduced features shape: {pca_features.shape}")
    print("PCA feature columns:")
    print(pca_features.columns.tolist())
    
    # Get the PCA transformer
    pca_transformer = builder.get_transformer('pca')
    if pca_transformer:
        print("\nExplained variance ratio:")
        print(pca_transformer.get_explained_variance_ratio())
    
    # KMeans dimensionality reduction
    print("\n4. KMeans Dimensionality Reduction")
    # Create a new builder
    builder = FeatureBuilder(data)
    kmeans_features = (builder
                      .add_technical_features()
                      .add_volume_features()
                      .add_kmeans_reduction(n_clusters=3)
                      .build())
    
    print(f"KMeans reduced features shape: {kmeans_features.shape}")
    print("KMeans feature columns:")
    print(kmeans_features.columns.tolist())
    
    # Using feature selector
    print("\n5. Feature Selection with FeatureSelector")
    # Create some dummy features with different variances
    numeric_data = pd.DataFrame({
        'high_var': np.random.normal(0, 10, 100),
        'medium_var': np.random.normal(0, 1, 100),
        'low_var': np.random.normal(0, 0.1, 100),
        'very_low_var': np.random.normal(0, 0.01, 100),
        'zero_var': np.zeros(100)
    })
    
    # Select by variance
    selector = FeatureSelector(method='variance_threshold', threshold=0.05)
    selected = selector.select_features(numeric_data)
    print(f"Original features: {numeric_data.columns.tolist()}")
    print(f"Selected features: {selected.columns.tolist()}")
    
    # Select by PCA
    selector = FeatureSelector(method='pca', n_components=2)
    pca_reduced = selector.select_features(numeric_data)
    print(f"PCA reduced shape: {pca_reduced.shape}")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 