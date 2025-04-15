import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Union, Callable
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class AdaBoostEnsemble(nn.Module):
    """
    PyTorch implementation of AdaBoost ensemble method.
    
    This implementation uses sample reweighting to focus on previously misclassified samples
    in subsequent weak learners, following the AdaBoost algorithm.
    """
    
    def __init__(
        self, 
        base_model_factory: Callable[[], nn.Module],
        n_estimators: int = 50, 
        learning_rate: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize AdaBoost ensemble.
        
        Args:
            base_model_factory: Function that returns a new instance of the base model
            n_estimators: Number of estimators to train
            learning_rate: Weight applied to each classifier at each iteration
            device: Device to use for computation (CPU/GPU)
        """
        super(AdaBoostEnsemble, self).__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize lists to store models and their weights
        self.models = nn.ModuleList()
        self.model_weights = []
        self.base_model_factory = base_model_factory
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction.
        
        Args:
            x: Input tensor of shape [batch_size, *features]
            
        Returns:
            Weighted sum of predictions from all models
        """
        if len(self.models) == 0:
            raise RuntimeError("No models in the ensemble. Call fit() first.")
        
        # Get predictions from each model and apply weights
        ensemble_preds = None
        
        for i, model in enumerate(self.models):
            model_pred = model(x)
            
            # For first prediction, initialize the ensemble prediction
            if ensemble_preds is None:
                ensemble_preds = torch.zeros_like(model_pred)
            
            # Add weighted prediction
            model_weight = self.model_weights[i]
            ensemble_preds += model_weight * model_pred
            
        # For binary classification, apply sign function
        if ensemble_preds.shape[1] == 1:
            return torch.sign(ensemble_preds)
        
        # For multi-class, return the maximum probability class
        return ensemble_preds
    
    def fit(
        self, 
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        epochs_per_model: int = 10,
        criterion: Optional[nn.Module] = None,
        optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None
    ) -> List[dict]:
        """
        Fit the AdaBoost ensemble.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Optional validation dataset
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            epochs_per_model: Number of epochs to train each model
            criterion: Loss function (defaults to BCEWithLogitsLoss for binary, CrossEntropyLoss for multiclass)
            optimizer_factory: Function that creates an optimizer for a model
            
        Returns:
            List of training history dictionaries for each model
        """
        X, y = self._extract_dataset(train_dataset)
        
        # Initialize sample weights uniformly
        n_samples = len(train_dataset)
        sample_weights = torch.ones(n_samples) / n_samples
        
        # Default loss function based on task type
        if criterion is None:
            if y.dim() == 1 or y.shape[1] == 1:
                criterion = nn.BCEWithLogitsLoss(reduction='none')
            else:
                criterion = nn.CrossEntropyLoss(reduction='none')
        
        # Default optimizer factory
        if optimizer_factory is None:
            optimizer_factory = lambda model: torch.optim.Adam(model.parameters(), lr=0.001)
        
        training_histories = []
        
        # Train each model sequentially
        for m in range(self.n_estimators):
            # Create a new base model
            model = self.base_model_factory().to(self.device)
            
            # Create a weighted sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights.numpy(),
                num_samples=n_samples,
                replacement=True
            )
            
            # Create DataLoader with the weighted sampler
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )
            
            # Create optimizer
            optimizer = optimizer_factory(model)
            
            # Train the model
            history = self._train_single_model(
                model=model,
                train_loader=train_loader,
                validation_dataset=validation_dataset,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs_per_model
            )
            
            # Make predictions with the trained model
            model.eval()
            with torch.no_grad():
                predictions = model(X.to(self.device))
                
            # Convert to class predictions if needed
            if predictions.shape[1] > 1:  # Multi-class
                pred_labels = torch.argmax(predictions, dim=1).cpu()
                actual_labels = torch.argmax(y, dim=1) if y.dim() > 1 else y
                incorrect = (pred_labels != actual_labels).float()
            else:  # Binary class
                pred_labels = (torch.sigmoid(predictions) > 0.5).cpu().float()
                incorrect = (pred_labels.squeeze() != y.squeeze()).float()
            
            # Calculate error rate (weighted)
            epsilon = torch.sum(sample_weights * incorrect) / torch.sum(sample_weights)
            epsilon = torch.clamp(epsilon, 1e-10, 1-1e-10)  # Prevent division by zero
            
            # Calculate model weight
            model_weight = self.learning_rate * torch.log((1 - epsilon) / epsilon)
            
            # Update sample weights
            sample_weights = sample_weights * torch.exp(model_weight * incorrect)
            
            # Normalize sample weights
            sample_weights = sample_weights / torch.sum(sample_weights)
            
            # Save model and its weight
            self.models.append(model)
            self.model_weights.append(model_weight.item())
            
            training_histories.append(history)
            
            print(f"Model {m+1}/{self.n_estimators} trained. "
                  f"Error rate: {epsilon.item():.4f}, "
                  f"Model weight: {model_weight.item():.4f}")
            
        return training_histories
    
    def _train_single_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_dataset: Optional[Dataset],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int
    ) -> dict:
        """
        Train a single model in the ensemble.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            validation_dataset: Optional validation dataset
            criterion: Loss function
            optimizer: Optimizer
            epochs: Number of epochs to train
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
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
                if target.dim() == 1 and output.shape[1] > 1:
                    # Multi-class case with class indices
                    loss = criterion(output, target)
                elif output.shape[1] == 1 and target.dim() == 1:
                    # Binary case with single output
                    loss = criterion(output.squeeze(), target.float())
                else:
                    # Shape should match
                    loss = criterion(output, target)
                
                # If loss is per-sample, take the mean
                if loss.dim() > 0:
                    loss = loss.mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase if validation set is provided
            if validation_dataset is not None:
                val_loss, val_acc = self._evaluate_model(
                    model, validation_dataset, criterion, batch_size=train_loader.batch_size
                )
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"train_loss={avg_train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_acc={val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}")
                
        return history
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataset: Dataset,
        criterion: nn.Module,
        batch_size: int = 32
    ) -> Tuple[float, float]:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            criterion: Loss function
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                
                # Calculate loss
                if target.dim() == 1 and output.shape[1] > 1:
                    # Multi-class case with class indices
                    loss = criterion(output, target)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                elif output.shape[1] == 1 and target.dim() == 1:
                    # Binary case with single output
                    loss = criterion(output.squeeze(), target.float())
                    pred = (torch.sigmoid(output) > 0.5).squeeze().float()
                    correct += (pred == target).sum().item()
                else:
                    # Shape should match
                    loss = criterion(output, target)
                    # For multi-label, use different accuracy calculation
                    if target.dim() > 1 and target.shape[1] > 1:
                        pred = (torch.sigmoid(output) > 0.5).float()
                        correct += (pred == target).sum().item() / target.shape[1]
                    
                # If loss is per-sample, take the mean
                if loss.dim() > 0:
                    loss = loss.mean()
                    
                total_loss += loss.item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
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
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'model_weights': self.model_weights,
            'models_state_dict': [model.state_dict() for model in self.models]
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
            Loaded AdaBoostEnsemble instance
        """
        state_dict = torch.load(path, map_location=device)
        ensemble = cls(
            base_model_factory=base_model_factory,
            n_estimators=state_dict['n_estimators'],
            learning_rate=state_dict['learning_rate'],
            device=device
        )
        
        # Create and load models
        for model_state in state_dict['models_state_dict']:
            model = base_model_factory().to(device)
            model.load_state_dict(model_state)
            ensemble.models.append(model)
            
        ensemble.model_weights = state_dict['model_weights']
        
        return ensemble
