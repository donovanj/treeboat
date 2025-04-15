import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
from torch.utils.data import Dataset, DataLoader, TensorDataset
import copy

class StackingEnsemble(nn.Module):
    """
    PyTorch implementation of a Stacking ensemble.
    
    This implementation trains multiple base models and uses their predictions
    as input features for a meta-model, which makes the final prediction.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_model: nn.Module,
        task_type: str = 'regression',
        use_features: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Stacking ensemble.
        
        Args:
            base_models: List of base models to use in the ensemble
            meta_model: Meta-model that combines predictions from base models
            task_type: 'regression' or 'classification'
            use_features: Whether to include original features in meta-model input
            device: Device to use for computation (CPU/GPU)
        """
        super(StackingEnsemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model
        self.task_type = task_type
        self.use_features = use_features
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to the specified device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction.
        
        Args:
            x: Input tensor of shape [batch_size, n_features]
            
        Returns:
            Meta-model predictions
        """
        # Get predictions from base models
        base_preds = self._get_base_predictions(x)
        
        # Create meta-features by combining base predictions with original features if required
        if self.use_features:
            meta_features = torch.cat([base_preds, x], dim=1)
        else:
            meta_features = base_preds
        
        # Get predictions from meta-model
        return self.meta_model(meta_features)
    
    def _get_base_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from all base models.
        
        Args:
            x: Input tensor of shape [batch_size, n_features]
            
        Returns:
            Concatenated predictions from all base models
        """
        base_preds = []
        
        for model in self.base_models:
            with torch.no_grad():
                model.eval()
                pred = model(x)
                
                # Handle different output shapes for different task types
                if self.task_type == 'classification':
                    if pred.size(-1) > 1:
                        # Multi-class: get probabilities for each class
                        pred = torch.softmax(pred, dim=-1)
                    else:
                        # Binary: get probability of positive class
                        pred = torch.sigmoid(pred)
                
                base_preds.append(pred)
        
        # Concatenate all predictions along the feature dimension
        return torch.cat(base_preds, dim=1)
    
    def fit(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        base_model_optimizers: Optional[List[torch.optim.Optimizer]] = None,
        meta_model_optimizer: Optional[torch.optim.Optimizer] = None,
        base_criterion: Optional[nn.Module] = None,
        meta_criterion: Optional[nn.Module] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        epochs_base: int = 10,
        epochs_meta: int = 10,
        n_folds: int = 5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fit the Stacking ensemble.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Optional validation dataset
            base_model_optimizers: Optimizers for base models (if None, defaults to Adam)
            meta_model_optimizer: Optimizer for meta model (if None, defaults to Adam)
            base_criterion: Loss function for base models (defaults based on task_type)
            meta_criterion: Loss function for meta model (defaults based on task_type)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            epochs_base: Number of epochs to train base models
            epochs_meta: Number of epochs to train meta model
            n_folds: Number of folds for cross-validation when generating meta-features
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Set default loss functions based on task type
        if base_criterion is None:
            if self.task_type == 'regression':
                base_criterion = nn.MSELoss()
            else:  # classification
                if hasattr(self, 'n_classes') and self.n_classes > 2:
                    base_criterion = nn.CrossEntropyLoss()
                else:
                    base_criterion = nn.BCEWithLogitsLoss()
        
        if meta_criterion is None:
            if self.task_type == 'regression':
                meta_criterion = nn.MSELoss()
            else:  # classification
                if hasattr(self, 'n_classes') and self.n_classes > 2:
                    meta_criterion = nn.CrossEntropyLoss()
                else:
                    meta_criterion = nn.BCEWithLogitsLoss()
        
        # Set default optimizers
        if base_model_optimizers is None:
            base_model_optimizers = [
                torch.optim.Adam(model.parameters(), lr=0.001)
                for model in self.base_models
            ]
        
        if meta_model_optimizer is None:
            meta_model_optimizer = torch.optim.Adam(self.meta_model.parameters(), lr=0.001)
        
        # Extract all data from dataset
        X, y = self._extract_dataset(train_dataset)
        
        # Extract validation data if provided
        if validation_dataset is not None:
            val_X, val_y = self._extract_dataset(validation_dataset)
        
        history = {
            'base_models_train_loss': [[] for _ in range(len(self.base_models))],
            'base_models_val_loss': [[] for _ in range(len(self.base_models))],
            'meta_model_train_loss': [],
            'meta_model_val_loss': []
        }
        
        # Step 1: Train base models
        if verbose:
            print("Training base models...")
        
        # Create a full dataset loader for training base models
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # Optionally create a validation loader
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        
        # Train each base model
        for i, (model, optimizer) in enumerate(zip(self.base_models, base_model_optimizers)):
            if verbose:
                print(f"Training base model {i+1}/{len(self.base_models)}")
            
            # Train the model
            model_history = self._train_model(
                model, optimizer, base_criterion,
                train_loader, 
                val_loader if validation_dataset is not None else None,
                epochs_base, verbose
            )
            
            # Store the training history
            history['base_models_train_loss'][i] = model_history['train_loss']
            if validation_dataset is not None:
                history['base_models_val_loss'][i] = model_history['val_loss']
        
        # Step 2: Generate meta-features using k-fold cross-validation
        if verbose:
            print("Generating meta-features using cross-validation...")
        
        meta_features = self._generate_meta_features(X, y, n_folds, batch_size, verbose)
        
        # Create meta-dataset
        if self.use_features:
            meta_X = torch.cat([meta_features, X], dim=1)
        else:
            meta_X = meta_features
        
        meta_dataset = TensorDataset(meta_X, y)
        meta_train_loader = DataLoader(
            meta_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # Generate meta-features for validation set if provided
        if validation_dataset is not None:
            val_meta_features = self._get_base_predictions(val_X)
            
            if self.use_features:
                val_meta_X = torch.cat([val_meta_features, val_X], dim=1)
            else:
                val_meta_X = val_meta_features
            
            meta_val_dataset = TensorDataset(val_meta_X, val_y)
            meta_val_loader = DataLoader(
                meta_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        else:
            meta_val_loader = None
        
        # Step 3: Train meta-model
        if verbose:
            print("Training meta-model...")
        
        meta_history = self._train_model(
            self.meta_model, meta_model_optimizer, meta_criterion,
            meta_train_loader, meta_val_loader, epochs_meta, verbose
        )
        
        # Store meta-model training history
        history['meta_model_train_loss'] = meta_history['train_loss']
        if validation_dataset is not None:
            history['meta_model_val_loss'] = meta_history['val_loss']
        
        return history
    
    def _train_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train a single model.
        
        Args:
            model: Model to train
            optimizer: Optimizer to use
            criterion: Loss function
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of epochs to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                
                # Handle different output and target shapes
                if self.task_type == 'classification':
                    if target.dim() == 1 and output.shape[1] > 1:
                        # Multi-class with class indices
                        loss = criterion(output, target.long())
                    elif output.shape[1] == 1 and target.dim() == 1:
                        # Binary with single output
                        loss = criterion(output.squeeze(), target.float())
                    else:
                        # Shape should match
                        loss = criterion(output, target)
                else:
                    loss = criterion(output, target)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase if validation loader is provided
            if val_loader is not None:
                val_loss = self._evaluate_model(model, val_loader, criterion)
                history['val_loss'].append(val_loss)
                
                if verbose and (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch+1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}, "
                          f"val_loss={val_loss:.4f}")
            elif verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}")
        
        return history
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            criterion: Loss function
            
        Returns:
            Average loss on the dataset
        """
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Handle different output and target shapes
                if self.task_type == 'classification':
                    if target.dim() == 1 and output.shape[1] > 1:
                        # Multi-class with class indices
                        loss = criterion(output, target.long())
                    elif output.shape[1] == 1 and target.dim() == 1:
                        # Binary with single output
                        loss = criterion(output.squeeze(), target.float())
                    else:
                        # Shape should match
                        loss = criterion(output, target)
                else:
                    loss = criterion(output, target)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _generate_meta_features(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_folds: int,
        batch_size: int,
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate meta-features using cross-validation to avoid data leakage.
        
        Args:
            X: Input features
            y: Target values
            n_folds: Number of cross-validation folds
            batch_size: Batch size for processing
            verbose: Whether to print progress
            
        Returns:
            Meta-features (predictions from base models)
        """
        n_samples = X.size(0)
        fold_size = n_samples // n_folds
        
        # Clone base models to avoid modifying the original models
        cloned_models = [copy.deepcopy(model) for model in self.base_models]
        
        # Prepare a tensor to hold all meta-features
        all_meta_features = []
        
        # Split data into folds
        indices = torch.randperm(n_samples)
        
        for fold in range(n_folds):
            if verbose:
                print(f"Processing fold {fold+1}/{n_folds}")
            
            # Create train and validation indices for this fold
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            val_indices = indices[start_idx:end_idx]
            train_indices = torch.cat([indices[:start_idx], indices[end_idx:]])
            
            # Create datasets for this fold
            train_X, train_y = X[train_indices], y[train_indices]
            val_X, val_y = X[val_indices], y[val_indices]
            
            train_dataset = TensorDataset(train_X, train_y)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
            )
            
            # Train each base model on the training set for this fold
            for i, model in enumerate(cloned_models):
                # Reset model parameters by re-initializing with the same architecture
                # This ensures we train a fresh model for each fold
                new_model = copy.deepcopy(self.base_models[i])
                model.load_state_dict(new_model.state_dict())
                
                # Train the model
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.MSELoss() if self.task_type == 'regression' else nn.BCEWithLogitsLoss()
                
                # Simple training loop for this fold
                model.to(self.device)
                model.train()
                
                for epoch in range(5):  # Fewer epochs for fold training
                    for data, target in train_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        optimizer.zero_grad()
                        output = model(data)
                        
                        # Handle different output and target shapes
                        if self.task_type == 'classification':
                            if target.dim() == 1 and output.shape[1] > 1:
                                loss = criterion(output, target.long())
                            elif output.shape[1] == 1 and target.dim() == 1:
                                loss = criterion(output.squeeze(), target.float())
                            else:
                                loss = criterion(output, target)
                        else:
                            loss = criterion(output, target)
                        
                        loss.backward()
                        optimizer.step()
            
            # Generate predictions on the validation set for this fold
            fold_meta_features = []
            
            for model in cloned_models:
                model.eval()
                with torch.no_grad():
                    # Process in batches to avoid memory issues
                    val_loader = DataLoader(
                        TensorDataset(val_X, val_y),
                        batch_size=batch_size,
                        shuffle=False
                    )
                    
                    fold_preds = []
                    for data, _ in val_loader:
                        data = data.to(self.device)
                        pred = model(data)
                        
                        # Handle different output shapes for different task types
                        if self.task_type == 'classification':
                            if pred.size(-1) > 1:
                                # Multi-class: get probabilities for each class
                                pred = torch.softmax(pred, dim=-1)
                            else:
                                # Binary: get probability of positive class
                                pred = torch.sigmoid(pred)
                        
                        fold_preds.append(pred.cpu())
                    
                    # Concatenate batch predictions
                    model_preds = torch.cat(fold_preds, dim=0)
                    fold_meta_features.append(model_preds)
            
            # Concatenate all model predictions for this fold
            fold_meta_features = torch.cat(fold_meta_features, dim=1)
            
            # Store the meta-features for this fold with their original indices
            for i, idx in enumerate(val_indices):
                while len(all_meta_features) <= idx:
                    all_meta_features.append(None)
                all_meta_features[idx] = fold_meta_features[i]
        
        # Stack all meta-features in the original order
        return torch.stack(all_meta_features)
    
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
    
    def save(self, path: str):
        """
        Save the ensemble model to a file.
        
        Args:
            path: Path to save the model
        """
        state_dict = {
            'task_type': self.task_type,
            'use_features': self.use_features,
            'base_models_state_dict': [model.state_dict() for model in self.base_models],
            'meta_model_state_dict': self.meta_model.state_dict()
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(
        cls,
        path: str,
        base_models: List[nn.Module],
        meta_model: nn.Module,
        device: Optional[torch.device] = None
    ):
        """
        Load an ensemble model from a file.
        
        Args:
            path: Path to load the model from
            base_models: List of base model instances (architecture only, weights will be loaded)
            meta_model: Meta-model instance (architecture only, weights will be loaded)
            device: Device to load the model on
            
        Returns:
            Loaded StackingEnsemble instance
        """
        state_dict = torch.load(path, map_location=device)
        
        # Create ensemble instance
        ensemble = cls(
            base_models=base_models,
            meta_model=meta_model,
            task_type=state_dict['task_type'],
            use_features=state_dict['use_features'],
            device=device
        )
        
        # Load model weights
        for i, model in enumerate(ensemble.base_models):
            model.load_state_dict(state_dict['base_models_state_dict'][i])
        
        ensemble.meta_model.load_state_dict(state_dict['meta_model_state_dict'])
        
        return ensemble
