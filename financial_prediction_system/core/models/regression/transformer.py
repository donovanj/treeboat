"""
Transformer Regression Model

This module implements a Transformer model for time series regression in financial applications.
It provides a wrapper around PyTorch's Transformer implementation with additional functionality
for training, evaluation, and prediction in the context of financial forecasting.

Based on PyTorch's Transformer implementation:
torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                    dim_feedforward=2048, dropout=0.1, activation=relu, 
                    custom_encoder=None, custom_decoder=None, layer_norm_eps=1e-05,
                    batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
import numpy as np
from typing import Dict, Tuple, Optional, Union, List, Callable, Any

from ..base import PredictionModel
from ..factory import ModelFactory


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model.
    
    This adds positional information to the input embeddings as
    transformers don't inherently understand sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter, but part of module state)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model) or 
               (batch_size, seq_len, d_model) if batch_first=True
                
        Returns:
            Tensor with positional encoding added
        """
        if x.dim() == 3 and x.size(1) != self.pe.size(1):
            # If batch dimension is first (batch_first=True)
            x = x.transpose(0, 1)
            x = x + self.pe[:x.size(0), :]
            x = self.dropout(x)
            x = x.transpose(0, 1)
        else:
            # Standard case (seq_len, batch, features)
            x = x + self.pe[:x.size(0), :]
            x = self.dropout(x)
            
        return x


class TimeSeriesTransformer(nn.Module):
    """Transformer neural network for time series forecasting."""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 1,
        max_seq_len: int = 100,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        batch_first: bool = True
    ):
        """Initialize time series transformer model.
        
        Args:
            input_size: Number of input features
            d_model: Dimension of the model (embedding dimension)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            output_size: Size of the output (prediction horizon)
            max_seq_len: Maximum sequence length for positional encoding
            activation: Activation function
            batch_first: If True, batch dimension is first in input tensor
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.batch_first = batch_first
        
        # Input projection to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=batch_first
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate a square causal attention mask.
        
        Args:
            sz: Size of the square matrix
            
        Returns:
            Tensor of shape (sz, sz) with elements set to -inf where future info is masked
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
        
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the transformer network.
        
        Args:
            src: Source sequence (input time series)
                Shape: (batch_size, src_seq_len, input_size) if batch_first=True
                or (src_seq_len, batch_size, input_size) otherwise
            tgt: Target sequence for the decoder
                If None, a shifted version of src is used (for autoregressive prediction)
                Shape: (batch_size, tgt_seq_len, input_size) if batch_first=True
                or (tgt_seq_len, batch_size, input_size) otherwise
            src_mask: Mask for source sequence
            tgt_mask: Mask for target sequence
            
        Returns:
            Predicted output tensor
        """
        # Project inputs to d_model dimension
        if self.batch_first:
            batch_size, src_seq_len, _ = src.shape
        else:
            src_seq_len, batch_size, _ = src.shape
        
        # Project inputs to d_model dimensions
        src = self.input_projection(src)
        
        # If no target is provided, use a shifted version of the source for autoregressive prediction
        if tgt is None:
            if self.batch_first:
                # Create shifted input as target (for autoregressive forecasting)
                # We use zeros for the last positions that we want to predict
                tgt = torch.zeros_like(src)
                tgt[:, :-1, :] = src[:, 1:, :]  # Shift by 1 position
            else:
                tgt = torch.zeros_like(src)
                tgt[:-1, :, :] = src[1:, :, :]  # Shift by 1 position
        else:
            # Project target to d_model dimensions
            tgt = self.input_projection(tgt)
        
        # Add positional encoding
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        
        # Create masks if not provided
        if src_mask is None:
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)
        
        if tgt_mask is None:
            tgt_seq_len = tgt.size(1) if self.batch_first else tgt.size(0)
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # Apply transformer
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to output size
        output = self.output_projection(output)
        
        return output


class TransformerRegressor(PredictionModel):
    """Transformer model for time series regression tasks.
    
    This class wraps a Transformer neural network with the PredictionModel interface
    for seamless integration with the prediction system pipeline.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_size: int = 1,
        max_seq_len: int = 100,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        num_epochs: int = 100,
        sequence_length: int = 20,
        prediction_length: int = 1,
        device: str = None
    ):
        """Initialize transformer regressor.
        
        Args:
            input_size: Number of input features
            d_model: Dimension of the model (embedding dimension)
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            output_size: Size of the output (prediction horizon)
            max_seq_len: Maximum sequence length for positional encoding
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            sequence_length: Length of input sequences
            prediction_length: Length of prediction window
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.output_size = output_size
        self.max_seq_len = max_seq_len
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create model
        self.model = TimeSeriesTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_size=output_size,
            max_seq_len=max_seq_len,
            activation=F.gelu,  # Using GELU activation (often better for transformers)
            batch_first=True    # Always use batch_first=True for easier handling
        ).to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def _prepare_data(self, features: np.ndarray, targets: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepare data for transformer model.
        
        Args:
            features: Input features
            targets: Optional target values
            
        Returns:
            Tuple of (features_tensor, targets_tensor)
        """
        # Convert to torch tensor
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        
        # Ensure we have the right dimensions (batch, seq, features)
        if features.dim() == 2:
            # Add sequence dimension if only (batch, features)
            features = features.unsqueeze(1)
        
        features = features.to(self.device)
        
        # Process targets if provided
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, dtype=torch.float32)
            targets = targets.to(self.device)
            return features, targets
        
        return features, None
    
    def train(self, features: np.ndarray, targets: np.ndarray, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, callback=None, **params):
        """Train the transformer model.
        
        Args:
            features: Training features of shape (n_samples, seq_len, n_features)
            targets: Target values of shape (n_samples, output_size)
            validation_data: Optional tuple of (val_features, val_targets) for validation
            callback: Optional callback function to report progress (epoch, loss)
            **params: Additional parameters to override default training settings
                - batch_size: Size of training batches
                - num_epochs: Number of training epochs
                - learning_rate: Learning rate for optimizer
                
        Returns:
            Training history (loss values)
        """
        # Update parameters if provided
        batch_size = params.get('batch_size', self.batch_size)
        num_epochs = params.get('num_epochs', self.num_epochs)
        learning_rate = params.get('learning_rate', self.learning_rate)
        
        # Update optimizer if learning rate changed
        if learning_rate != self.learning_rate:
            self.learning_rate = learning_rate
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=False
            )
        
        # Prepare data
        X, y = self._prepare_data(features, targets)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare validation data if provided
        val_dataloader = None
        if validation_data is not None:
            val_X, val_y = self._prepare_data(validation_data[0], validation_data[1])
            val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Forward pass
                self.optimizer.zero_grad()
                
                # Create target input for decoder (shifted version of source)
                # For teacher forcing during training
                outputs = self.model(batch_X)
                
                # Get predictions for the target time steps
                if outputs.dim() == 3:
                    # Get the last time step (for next step prediction)
                    # or specific time steps depending on the task
                    outputs = outputs[:, -1, :]
                
                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients (common for transformers)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # Record average training loss
            avg_train_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = None
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                self.val_losses.append(val_loss)
                
                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)
            
            # Call the callback if provided
            if callback is not None:
                callback(epoch, avg_train_loss if val_loss is None else val_loss)
        
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses}
    
    def _validate(self, val_dataloader):
        """Run validation and return validation loss."""
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                outputs = self.model(batch_X)
                
                # Get predictions for the target time steps
                if outputs.dim() == 3:
                    outputs = outputs[:, -1, :]
                
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
        
        self.model.train()
        return val_loss / len(val_dataloader)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.
        
        Args:
            features: Input features of shape (n_samples, seq_len, n_features)
            
        Returns:
            Predicted values as numpy array
        """
        # Prepare input data
        X, _ = self._prepare_data(features)
        
        # Generate predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            
            # Get predictions for the target time steps
            if predictions.dim() == 3:
                predictions = predictions[:, -1, :]
        
        # Convert to numpy
        return predictions.cpu().numpy()
    
    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            features: Input features
            targets: Ground truth values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        predictions = self.predict(features)
        targets = np.array(targets)
        
        # Calculate metrics
        mse = np.mean((predictions - targets)**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))
        
        # Additional metrics for financial time series
        direction_accuracy = np.mean((np.sign(predictions) == np.sign(targets)).astype(int))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'direction_accuracy': direction_accuracy
        }
    
    def save(self, path: str):
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'hyperparameters': {
                'input_size': self.input_size,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'num_decoder_layers': self.num_decoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'output_size': self.output_size,
                'max_seq_len': self.max_seq_len,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'num_epochs': self.num_epochs,
                'sequence_length': self.sequence_length,
                'prediction_length': self.prediction_length
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load(self, path: str):
        """Load model from disk.
        
        Args:
            path: Path to saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Recreate model with saved hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.input_size = hyperparams['input_size']
        self.d_model = hyperparams['d_model']
        self.nhead = hyperparams['nhead']
        self.num_encoder_layers = hyperparams['num_encoder_layers']
        self.num_decoder_layers = hyperparams['num_decoder_layers'] 
        self.dim_feedforward = hyperparams['dim_feedforward']
        self.dropout = hyperparams['dropout']
        self.output_size = hyperparams['output_size']
        self.max_seq_len = hyperparams['max_seq_len']
        self.learning_rate = hyperparams['learning_rate']
        self.batch_size = hyperparams['batch_size']
        self.num_epochs = hyperparams['num_epochs']
        self.sequence_length = hyperparams['sequence_length']
        self.prediction_length = hyperparams['prediction_length']
        
        # Recreate model
        self.model = TimeSeriesTransformer(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=self.output_size,
            max_seq_len=self.max_seq_len,
            batch_first=True
        ).to(self.device)
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=False
        )
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


# Register the model with the factory
ModelFactory.register('transformer', TransformerRegressor)
