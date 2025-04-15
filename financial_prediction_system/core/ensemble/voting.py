import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
from torch.utils.data import Dataset, DataLoader

class VotingEnsemble(nn.Module):
    """
    PyTorch implementation of a Voting ensemble.
    
    This implementation supports both hard voting (majority voting) for classification
    and soft voting (weighted average) for both classification and regression.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        task_type: str = 'classification',
        voting: str = 'soft',
        device: Optional[torch.device] = None
    ):
        """
        Initialize Voting ensemble.
        
        Args:
            models: List of models to ensemble
            weights: Optional list of weights for each model (default: equal weights)
            task_type: 'classification' or 'regression'
            voting: 'hard' (only for classification) or 'soft'
            device: Device to use for computation (CPU/GPU)
        """
        super(VotingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Validate inputs
        if voting == 'hard' and task_type != 'classification':
            raise ValueError("Hard voting is only available for classification tasks")
        
        # Set weights to equal if not provided
        if weights is None:
            self.weights = torch.ones(len(models))
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            self.weights = torch.tensor(weights)
            
        # Normalize weights
        self.weights = self.weights / self.weights.sum()
        
        self.task_type = task_type
        self.voting = voting
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to the specified device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction.
        
        Args:
            x: Input tensor of shape [batch_size, n_features]
            
        Returns:
            Ensemble prediction based on voting strategy
        """
        # Get predictions from all models
        predictions = []
        
        for model in self.models:
            with torch.no_grad():
                model.eval()
                pred = model(x)
                predictions.append(pred)
        
        # Perform voting based on strategy and task type
        if self.task_type == 'classification':
            if self.voting == 'hard':
                return self._hard_voting_classification(predictions)
            else:  # soft voting
                return self._soft_voting_classification(predictions)
        else:  # regression
            return self._soft_voting_regression(predictions)
    
    def _hard_voting_classification(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform hard voting for classification by taking the most common prediction.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Most common class prediction
        """
        # Get class predictions from each model
        if predictions[0].size(-1) > 1:
            # For multi-class, convert logits to class indices
            class_predictions = [pred.argmax(dim=-1) for pred in predictions]
        else:
            # For binary, threshold at 0.5
            class_predictions = [(torch.sigmoid(pred) > 0.5).long() for pred in predictions]
        
        # Stack predictions along a new dimension [n_models, batch_size]
        stacked_preds = torch.stack(class_predictions)
        
        # For each sample, count the votes for each class
        n_classes = max(pred.max().item() for pred in class_predictions) + 1
        batch_size = stacked_preds.size(1)
        
        # One-hot encode the predictions to count votes for each class
        votes = torch.zeros(batch_size, n_classes, device=self.device)
        
        for model_idx, weight in enumerate(self.weights):
            # For each model, add its weighted vote
            for sample_idx in range(batch_size):
                class_idx = stacked_preds[model_idx, sample_idx].item()
                votes[sample_idx, int(class_idx)] += weight
        
        # Return the class with the most votes for each sample
        return votes.argmax(dim=1)
    
    def _soft_voting_classification(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform soft voting for classification by averaging class probabilities.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Average of class probabilities
        """
        # Convert logits to probabilities if needed
        probs = []
        
        for pred in predictions:
            if pred.size(-1) > 1:
                # Multi-class: apply softmax
                prob = torch.softmax(pred, dim=-1)
                probs.append(prob)
            else:
                # Binary: apply sigmoid and create [1-p, p] probabilities
                p = torch.sigmoid(pred)
                # Reshape to [batch_size, 2]
                prob = torch.cat([1 - p, p], dim=-1)
                probs.append(prob)
        
        # Compute weighted average of probabilities
        weighted_probs = torch.zeros_like(probs[0])
        
        for i, prob in enumerate(probs):
            weighted_probs += self.weights[i] * prob
        
        return weighted_probs
    
    def _soft_voting_regression(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform soft voting for regression by averaging predictions.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Weighted average of predictions
        """
        # Stack predictions along a new dimension [n_models, batch_size, *]
        stacked_preds = torch.stack(predictions)
        
        # Compute weighted average along the model dimension
        weights = self.weights.view(-1, *([1] * (len(stacked_preds.shape) - 1))).to(self.device)
        weighted_pred = (stacked_preds * weights).sum(dim=0)
        
        return weighted_pred
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions for input data.
        
        For classification, returns class indices.
        For regression, returns the predicted values.
        
        Args:
            x: Input tensor of shape [batch_size, n_features]
            
        Returns:
            Predictions
        """
        outputs = self(x)
        
        if self.task_type == 'classification':
            if outputs.size(-1) > 1:
                # Multi-class: return most likely class
                return outputs.argmax(dim=-1)
            else:
                # Binary: threshold at 0.5
                return (outputs > 0.5).float()
        
        # For regression, return as is
        return outputs
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for classification tasks.
        
        Args:
            x: Input tensor of shape [batch_size, n_features]
            
        Returns:
            Class probabilities
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        outputs = self(x)
        
        # Ensure output is probabilities
        if self.voting == 'hard':
            # Convert one-hot to probabilities
            if outputs.dim() == 1:
                # Convert class indices to one-hot
                n_classes = outputs.max().item() + 1
                one_hot = torch.zeros(outputs.size(0), n_classes, device=self.device)
                one_hot.scatter_(1, outputs.unsqueeze(1).long(), 1)
                return one_hot
            return outputs
        
        # For soft voting, output is already probabilities
        return outputs
    
    def fit(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        criterion: Optional[nn.Module] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Fit all models in the ensemble.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Optional validation dataset
            optimizers: List of optimizers for each model (if None, defaults to Adam)
            criterion: Loss function (defaults based on task_type)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            epochs: Number of epochs to train
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Set default criterion based on task type
        if criterion is None:
            if self.task_type == 'regression':
                criterion = nn.MSELoss()
            else:  # classification
                criterion = nn.BCEWithLogitsLoss() if not hasattr(self, 'n_classes') or self.n_classes <= 2 else nn.CrossEntropyLoss()
        
        # Set default optimizers if not provided
        if optimizers is None:
            optimizers = [
                torch.optim.Adam(model.parameters(), lr=0.001)
                for model in self.models
            ]
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        
        # History to track training progress
        history = {
            'train_loss': [[] for _ in range(len(self.models))],
            'val_loss': [[] for _ in range(len(self.models))]
        }
        
        # Determine if we're doing multiclass classification
        is_multiclass = False
        for batch in train_loader:
            _, targets = batch
            if len(targets.shape) == 1 and targets.dtype == torch.int64:
                is_multiclass = True
            break
        
        # Train each model independently
        for model_idx, (model, optimizer) in enumerate(zip(self.models, optimizers)):
            if verbose:
                print(f"Training model {model_idx+1}/{len(self.models)}")
            
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
                        if is_multiclass and output.shape[1] > 1:
                            # Multi-class with class indices (output: [batch_size, n_classes], target: [batch_size])
                            loss = criterion(output, target.long())
                        elif output.shape[1] == 1 and target.dim() == 1:
                            # Binary with single output (output: [batch_size, 1], target: [batch_size])
                            loss = criterion(output.squeeze(), target.float())
                        else:
                            # Shape should match
                            loss = criterion(output, target)
                    else:
                        # For regression
                        if output.shape != target.shape:
                            # Ensure shapes match
                            if output.dim() > target.dim():
                                # If output has more dimensions, squeeze it
                                output = output.squeeze()
                            elif target.dim() > output.dim():
                                # If target has more dimensions, squeeze it
                                target = target.squeeze()
                        loss = criterion(output, target)
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                history['train_loss'][model_idx].append(avg_train_loss)
                
                # Validation phase if validation set is provided
                if validation_dataset is not None:
                    val_loss = self._evaluate_model(model, val_loader, criterion, is_multiclass)
                    history['val_loss'][model_idx].append(val_loss)
                    
                    if verbose and (epoch + 1) % 5 == 0:
                        print(f"  Model {model_idx+1}, Epoch {epoch+1}/{epochs}: "
                              f"train_loss={avg_train_loss:.4f}, "
                              f"val_loss={val_loss:.4f}")
                elif verbose and (epoch + 1) % 5 == 0:
                    print(f"  Model {model_idx+1}, Epoch {epoch+1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}")
        
        return history
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        is_multiclass: bool = False
    ) -> float:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            criterion: Loss function
            is_multiclass: Whether this is a multiclass classification task with class indices
            
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
                    if is_multiclass and output.shape[1] > 1:
                        # Multi-class with class indices
                        loss = criterion(output, target.long())
                    elif output.shape[1] == 1 and target.dim() == 1:
                        # Binary with single output
                        loss = criterion(output.squeeze(), target.float())
                    else:
                        # Shape should match
                        loss = criterion(output, target)
                else:
                    # For regression
                    if output.shape != target.shape:
                        # Ensure shapes match
                        if output.dim() > target.dim():
                            # If output has more dimensions, squeeze it
                            output = output.squeeze()
                        elif target.dim() > output.dim():
                            # If target has more dimensions, squeeze it
                            target = target.squeeze()
                    loss = criterion(output, target)
                
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(
        self,
        dataset: Dataset,
        criterion: Optional[nn.Module] = None,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, float]:
        """
        Evaluate the ensemble on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            criterion: Loss function (defaults based on task_type)
            batch_size: Batch size for evaluation
            num_workers: Number of workers for data loading
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Set default criterion based on task type
        if criterion is None:
            if self.task_type == 'regression':
                criterion = nn.MSELoss()
            else:  # classification
                criterion = nn.BCEWithLogitsLoss() if not hasattr(self, 'n_classes') or self.n_classes <= 2 else nn.CrossEntropyLoss()
        
        # Create data loader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        # Metrics to compute
        metrics = {
            'loss': 0.0
        }
        
        if self.task_type == 'classification':
            metrics['accuracy'] = 0.0
        
        # Evaluate
        self.eval()
        total_samples = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                
                # Calculate loss
                if self.task_type == 'classification':
                    if target.dim() == 1 and output.shape[1] > 1:
                        # Multi-class
                        loss = criterion(output, target.long())
                        pred = output.argmax(dim=1)
                        correct += (pred == target).sum().item()
                    elif output.shape[1] == 1 and target.dim() == 1:
                        # Binary
                        loss = criterion(output.squeeze(), target.float())
                        pred = (output > 0.5).float().squeeze()
                        correct += (pred == target).sum().item()
                    else:
                        # Multi-label
                        loss = criterion(output, target)
                        pred = (output > 0.5).float()
                        correct += (pred == target).sum().item() / target.shape[1]
                else:
                    # Regression
                    loss = criterion(output, target)
                
                metrics['loss'] += loss.item() * data.size(0)
                total_samples += data.size(0)
        
        # Compute average metrics
        metrics['loss'] /= total_samples
        
        if self.task_type == 'classification':
            metrics['accuracy'] = correct / total_samples
        
        return metrics
    
    def save(self, path: str):
        """
        Save the ensemble model to a file.
        
        Args:
            path: Path to save the model
        """
        state_dict = {
            'task_type': self.task_type,
            'voting': self.voting,
            'weights': self.weights,
            'models_state_dict': [model.state_dict() for model in self.models]
        }
        torch.save(state_dict, path)
    
    @classmethod
    def load(
        cls,
        path: str,
        models: List[nn.Module],
        device: Optional[torch.device] = None
    ):
        """
        Load an ensemble model from a file.
        
        Args:
            path: Path to load the model from
            models: List of model instances (architecture only, weights will be loaded)
            device: Device to load the model on
            
        Returns:
            Loaded VotingEnsemble instance
        """
        state_dict = torch.load(path, map_location=device)
        
        ensemble = cls(
            models=models,
            weights=state_dict['weights'],
            task_type=state_dict['task_type'],
            voting=state_dict['voting'],
            device=device
        )
        
        # Load model weights
        for i, model in enumerate(ensemble.models):
            model.load_state_dict(state_dict['models_state_dict'][i])
        
        return ensemble

    def _soft_voting_multiclass(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Perform soft voting for multi-class classification by averaging probabilities.
        
        Args:
            predictions: List of model predictions (logits)
            
        Returns:
            Average of class probabilities
        """
        # Apply softmax to convert logits to probabilities
        probs = [torch.softmax(pred, dim=-1) for pred in predictions]
        
        # Apply weights to each model's probabilities
        weighted_probs = torch.zeros_like(probs[0])
        for i, prob in enumerate(probs):
            weighted_probs += self.weights[i] * prob
        
        return weighted_probs
    
    def fit_multiclass(
        self,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        epochs: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Special fit method for multiclass classification that handles CrossEntropyLoss correctly.
        
        Args:
            train_dataset: Training dataset
            validation_dataset: Optional validation dataset
            optimizers: List of optimizers for each model (if None, defaults to Adam)
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            epochs: Number of epochs to train
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        # Use CrossEntropyLoss which expects class indices (not one-hot encoded)
        criterion = nn.CrossEntropyLoss()
        
        # Set default optimizers if not provided
        if optimizers is None:
            optimizers = [
                torch.optim.Adam(model.parameters(), lr=0.001)
                for model in self.models
            ]
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        
        # History to track training progress
        history = {
            'train_loss': [[] for _ in range(len(self.models))],
            'val_loss': [[] for _ in range(len(self.models))]
        }
        
        # Train each model independently
        for model_idx, (model, optimizer) in enumerate(zip(self.models, optimizers)):
            if verbose:
                print(f"Training model {model_idx+1}/{len(self.models)}")
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                epoch_loss = 0.0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    
                    # CrossEntropyLoss expects [batch_size, n_classes] logits and [batch_size] class indices
                    loss = criterion(output, target.long())
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                history['train_loss'][model_idx].append(avg_train_loss)
                
                # Validation phase if validation set is provided
                if validation_dataset is not None:
                    val_loss = 0.0
                    model.eval()
                    
                    with torch.no_grad():
                        for data, target in val_loader:
                            data, target = data.to(self.device), target.to(self.device)
                            output = model(data)
                            loss = criterion(output, target.long())
                            val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(val_loader)
                    history['val_loss'][model_idx].append(avg_val_loss)
                    
                    if verbose and (epoch + 1) % 5 == 0:
                        print(f"  Model {model_idx+1}, Epoch {epoch+1}/{epochs}: "
                              f"train_loss={avg_train_loss:.4f}, "
                              f"val_loss={avg_val_loss:.4f}")
                elif verbose and (epoch + 1) % 5 == 0:
                    print(f"  Model {model_idx+1}, Epoch {epoch+1}/{epochs}: "
                          f"train_loss={avg_train_loss:.4f}")
        
        return history
