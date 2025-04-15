import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Union
from financial_prediction_system.core.features.dimensionality_reduction.dimensionality_reducer_base import DimensionalityReducer

class KMeansReducer(DimensionalityReducer):
    """KMeans dimensionality reduction"""
    
    def __init__(
        self, 
        n_clusters: int = 8,
        init: str = 'k-means++',
        n_init: Union[int, str] = 'auto',
        max_iter: int = 300,
        random_state: int = None,
        prefix: str = "cluster_"
    ):
        """
        Initialize the KMeans reducer
        
        Parameters
        ----------
        n_clusters : int, default=8
            Number of clusters to form
        init : str, default='k-means++'
            Method for initialization
        n_init : int or 'auto', default='auto'
            Number of time the k-means algorithm will be run with different seeds
        max_iter : int, default=300
            Maximum number of iterations for each run
        random_state : int, optional
            Random state for reproducibility
        prefix : str, default="cluster_"
            Prefix to add to feature names
        """
        super().__init__(n_components=n_clusters, prefix=prefix)
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self._create_model()
    
    def _create_model(self):
        """Create the KMeans model"""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
    
    def fit(self, data: pd.DataFrame) -> 'KMeansReducer':
        """
        Fit the KMeans model to the data
        
        Parameters
        ----------
        data : pd.DataFrame
            The input data
        
        Returns
        -------
        self : KMeansReducer
            The fitted reducer
        """
        self.model.fit(data)
        self.feature_names = [f"{self.prefix}{i}" for i in range(self.n_clusters)]
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using KMeans
        
        Parameters
        ----------
        data : pd.DataFrame
            The input data
        
        Returns
        -------
        pd.DataFrame
            The distances to cluster centers as features
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Calculate distances to each cluster center
        distances = []
        for i, center in enumerate(self.model.cluster_centers_):
            dist = np.sqrt(((data.values - center) ** 2).sum(axis=1))
            distances.append(dist)
        
        # Create DataFrame with distances to each cluster
        return pd.DataFrame(
            np.column_stack(distances),
            index=data.index,
            columns=self.feature_names
        )
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for data"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(data)
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get the cluster centers"""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.cluster_centers_