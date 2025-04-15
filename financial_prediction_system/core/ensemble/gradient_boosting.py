import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union, Callable, Dict, Any
from torch.utils.data import Dataset, DataLoader

class GradientBoostingEnsemble(nn.Module):
    """
    PyTorch implementation of Gradient Boosting ensemble method.
    
    This implementation builds sequential models where each model focuses on
    the residual errors of previous models.
    """
    
    def __init__(
        self, 
        base_model_factory: Callable[[], nn.Module],
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        loss: str = 'mse',
        device: Optional[torch.device] = None
    ):
        """
        Initialize Gradient Boosting ensemble.
        
        Args:
            base_model_factory: Function that returns a new instance of the base model
            n_estimators: Number of estimators to train
            learning_rate: Shrinkage applied to each tree's contribution
            subsample: Fraction of samples to use for fitting the individual base learners
            loss: Loss function to optimize ('mse' for regression, 'log_loss' for binary classification)
            device: Device to use for computation (CPU/GPU)
        """
        super(GradientBoostingEnsemble, self).__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.loss = loss
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize list to store models
        self.models = nn.ModuleList()
        self.base_model_factory = base_model_factory
        
        # Initial prediction (prior)
        self.initial_prediction = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction.
        
        Args:
            x: Input tensor of shape [batch_size, *features]
            
        Returns:
            Ensemble prediction by summing all model outputs
        """
        if self.initial_prediction is None:
            raise RuntimeError("Model has not been fitted yet.")
            
        # Start with initial prediction
        if isinstance(self.initial_prediction, torch.Tensor):
            # Use broadcast to match batch size
            batch_size = x.size(0)
            ensemble_preds = self.initial_prediction.expand(batch_size, -1).to(self.device)
        else:
            # For scalar initial prediction
            ensemble_preds = torch.full((x.size(0), 1), self.initial_prediction, 
                                        device=self.device, dtype=torch.float32)
            
        # Add contributions from each model
        for model in self.models:
            model_pred = model(x)
            ensemble_preds = ensemble_preds + self.learning_rate * model_pred
            
        # For classification with log loss, apply sigmoid
        if self.loss == 'log_loss':
            ensemble_preds = torch.sigmoid(ensemble_preds)
            
        return ensemble_preds
    
    def fit(
        self, 
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        epochs_per_model: int = 10,
        early_stopping: int = 5,
        optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fit the Gradient Boosting ensemble.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Optional validation dataset
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            epochs_per_model: Number of epochs to train each model
            early_stopping: Number of rounds with no improvement to wait before early stopping
            optimizer_factory: Function that creates an optimizer for a model
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Extract all data from dataset for computing initial prediction and residuals
        X, y = self._extract_dataset(train_dataset)
        X, y = X.to(self.device), y.to(self.device)
        
        # Set initial prediction based on loss function
        self._initialize_prediction(y)
        
        # Default optimizer factory
        if optimizer_factory is None:
            optimizer_factory = lambda model: torch.optim.Adam(model.parameters(), lr=0.01)
        
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Current predictions (will be updated iteratively)
        if isinstance(self.initial_prediction, torch.Tensor):
            # Handle multi-dimensional targets
            current_preds = self.initial_prediction.expand(X.size(0), -1).to(self.device)
        else:
            # For scalar initial prediction
            current_preds = torch.full((X.size(0), 1), self.initial_prediction, 
                                      device=self.device, dtype=torch.float32)
        
        best_val_loss = float('inf')
        rounds_no_improve = 0
        
        # Train each model sequentially
        for m in range(self.n_estimators):
            # Calculate negative gradient (residuals)
            negative_gradient = self._calculate_negative_gradient(y, current_preds)
            
            # Create new model for this iteration
            model = self.base_model_factory().to(self.device)
            
            # Create dataset with current features and negative gradients as targets
            residual_dataset = self._create_residual_dataset(train_dataset, negative_gradient)
            
            # Create DataLoader with subsampling if needed
            if self.subsample < 1.0:
                n_samples = len(residual_dataset)
                indices = torch.randperm(n_samples)[:int(n_samples * self.subsample)]
                subset = torch.utils.data.Subset(residual_dataset, indices)
                train_loader = DataLoader(
                    subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
                )
            else:
                train_loader = DataLoader(
                    residual_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
                )
            
            # Train the model on residuals
            optimizer = optimizer_factory(model)
            
            # Use MSE loss for residual fitting regardless of the ensemble's loss function
            criterion = nn.MSELoss()
            
            # Train model
            model_history = self._train_single_model(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs_per_model,
                verbose=verbose
            )
            
            # Update history
            history['train_loss'].extend(model_history['train_loss'])
            
            # Add trained model to ensemble
            self.models.append(model)
            
            # Update current predictions by adding contribution of the new model
            with torch.no_grad():
                new_contrib = model(X)
                current_preds = current_preds + self.learning_rate * new_contrib
            
            # Evaluate on validation set if provided
            if validation_dataset is not None:
                val_X, val_y = self._extract_dataset(validation_dataset)
                val_X, val_y = val_X.to(self.device), val_y.to(self.device)
                
                # Forward pass with current ensemble
                with torch.no_grad():
                    val_preds = self(val_X)
                
                # Calculate validation loss
                val_loss = self._calculate_loss(val_y, val_preds)
                history['val_loss'].append(val_loss.item())
                
                if verbose:
                    print(f"Model {m+1}/{self.n_estimators}: "
                          f"train_loss={model_history['train_loss'][-1]:.4f}, "
                          f"val_loss={val_loss.item():.4f}")
                
                # Check for early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1
                    if rounds_no_improve >= early_stopping:
                        if verbose:
                            print(f"Early stopping at model {m+1}/{self.n_estimators}")
                        break
            else:
                if verbose:
                    print(f"Model {m+1}/{self.n_estimators}: "
                          f"train_loss={model_history['train_loss'][-1]:.4f}")
        
        return history
    
    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train a single model in the ensemble.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        history = {'train_loss': []}
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_loss:.4f}")
                
        return history
    
    def _initialize_prediction(self, y: torch.Tensor):
        """
        Initialize the base prediction for the ensemble based on loss function.
        
        Args:
            y: Target values
        """
        if self.loss == 'mse':
            # For MSE, initial prediction is the mean of targets
            if y.dim() > 1 and y.shape[1] > 1:
                # Multi-dimensional targets, compute mean for each dimension
                self.initial_prediction = y.mean(dim=0, keepdim=True)
            else:
                # Scalar targets
                self.initial_prediction = y.mean().item()
        
        elif self.loss == 'log_loss':
            # For log loss, initial prediction is log-odds of positive class
            if y.dim() > 1 and y.shape[1] > 1:
                # Multi-label classification
                pos_ratio = y.mean(dim=0, keepdim=True)
                # Clip to avoid numerical issues
                pos_ratio = torch.clamp(pos_ratio, 1e-5, 1-1e-5)
                self.initial_prediction = torch.log(pos_ratio / (1 - pos_ratio))
            else:
                # Binary classification
                pos_ratio = y.mean().item()
                # Clip to avoid numerical issues
                pos_ratio = max(1e-5, min(1-1e-5, pos_ratio))
                self.initial_prediction = np.log(pos_ratio / (1 - pos_ratio))
    
    def _calculate_negative_gradient(self, y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative gradient of the loss function.
        
        Args:
            y: True target values
            pred: Current predictions
            
        Returns:
            Negative gradient (residuals)
        """
        if self.loss == 'mse':
            # For MSE, negative gradient is simply (y - pred)
            return y - pred
        
        elif self.loss == 'log_loss':
            # For log loss, negative gradient is (y - sigmoid(pred))
            sigmoid_pred = torch.sigmoid(pred)
            return y - sigmoid_pred
    
    def _calculate_loss(self, y: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss based on the loss function.
        
        Args:
            y: True target values
            pred: Predictions
            
        Returns:
            Loss value
        """
        if self.loss == 'mse':
            return torch.mean((y - pred) ** 2)
        
        elif self.loss == 'log_loss':
            # Binary cross-entropy loss
            return torch.mean(
                -y * torch.log(pred + 1e-10) - (1 - y) * torch.log(1 - pred + 1e-10)
            )
    
    def _extract_dataset(self, dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract all data from a dataset into tensors.
        
        Args:
            dataset: Dataset to extract from
            
        Returns:
            Tuple of (features, targets)
        """
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        # Get a single batch containing all samples
        data_iter = iter(dataloader)
        features, targets = next(data_iter)
        return features, targets
    
    def _create_residual_dataset(
        self, original_dataset: Dataset, residuals: torch.Tensor
    ) -> Dataset:
        """
        Create a new dataset where targets are replaced with residuals.
        
        Args:
            original_dataset: Original dataset with features and targets
            residuals: Residual values to use as new targets
            
        Returns:
            New dataset with same features but residuals as targets
        """
        class ResidualDataset(Dataset):
            def __init__(self, original_dataset, residuals):
                self.original_dataset = original_dataset
                self.residuals = residuals.cpu()
                
            def __len__(self):
                return len(self.original_dataset)
                
            def __getitem__(self, idx):
                features, _ = self.original_dataset[idx]
                return features, self.residuals[idx]
                
        return ResidualDataset(original_dataset, residuals)
    
    def save(self, path: str):
        """
        Save the ensemble model to a file.
        
        Args:
            path: Path to save the model
        """
        state_dict = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'loss': self.loss,
            'initial_prediction': self.initial_prediction,
            'state_dict': self.state_dict()
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(cls, path: str, base_model_factory: Callable[[], nn.Module], device: Optional[torch.device] = None):
        """
        Load an ensemble model from a file.
        
        Args:
            path: Path to load the model from
            base_model_factory: Function that returns a new instance of the base model
            device: Device to load the model on
            
        Returns:
            Loaded GradientBoostingEnsemble instance
        """
        state_dict = torch.load(path, map_location=device)
        ensemble = cls(
            base_model_factory=base_model_factory,
            n_estimators=state_dict['n_estimators'],
            learning_rate=state_dict['learning_rate'],
            subsample=state_dict['subsample'],
            loss=state_dict['loss'],
            device=device
        )
        ensemble.load_state_dict(state_dict['state_dict'])
        ensemble.initial_prediction = state_dict['initial_prediction']
        return ensemble
