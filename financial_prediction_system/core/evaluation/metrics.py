"""Metrics module for evaluating model and backtest performance"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

def calculate_metrics(y_true, y_pred, task_type='regression', cv_results=None):
    """Calculate performance metrics for a model
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        task_type: Type of task ('regression' or 'classification')
        cv_results: Cross-validation results (optional)
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    if task_type == 'regression':
        # Regression metrics
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Calculate mean absolute percentage error (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
        metrics['mape'] = mape
        
    else:  # classification
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        try:
            # These can fail for multiclass or if classes are missing
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        except Exception:
            pass
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC AUC for binary classification
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
        except Exception:
            pass
    
    # Add cross-validation metrics if provided
    if cv_results is not None:
        for metric, values in cv_results.items():
            metrics[f'cv_mean_{metric}'] = np.mean(values)
            metrics[f'cv_std_{metric}'] = np.std(values)
    
    return metrics

def calculate_backtest_metrics(backtest_results):
    """Calculate performance metrics for backtest results
    
    Args:
        backtest_results: Dictionary with backtest results
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Extract trades and equity curve
    trades = backtest_results.get('trades', [])
    equity_curve_data = backtest_results.get('equity_curve', [])
    
    # If no trades or equity curve, return empty metrics
    if not trades or not equity_curve_data:
        return metrics
    
    # Convert equity curve to pandas if it's a list of dicts
    if isinstance(equity_curve_data, list):
        equity_curve = pd.DataFrame(equity_curve_data)
    else:
        equity_curve = equity_curve_data
    
    # Total return
    if len(equity_curve) > 1:
        total_return = (equity_curve['equity'].iloc[-1] / equity_curve['equity'].iloc[0] - 1) * 100
        metrics['total_return'] = total_return
    else:
        metrics['total_return'] = 0.0
    
    # Extract trade statistics
    if trades:
        # Win rate
        profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
        win_rate = len(profitable_trades) / len(trades) * 100 if trades else 0
        metrics['win_rate'] = win_rate
        
        # Profit factor
        gross_profit = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        metrics['profit_factor'] = profit_factor
        
        # Average trade
        avg_profit = np.mean([t.get('pnl', 0) for t in profitable_trades]) if profitable_trades else 0
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        avg_loss = np.mean([abs(t.get('pnl', 0)) for t in losing_trades]) if losing_trades else 0
        metrics['average_profit'] = avg_profit
        metrics['average_loss'] = avg_loss
    else:
        metrics['win_rate'] = 0.0
        metrics['profit_factor'] = 0.0
        metrics['average_profit'] = 0.0
        metrics['average_loss'] = 0.0
    
    # Calculate returns and drawdowns from equity curve
    if len(equity_curve) > 1:
        # Daily returns
        if 'equity' in equity_curve.columns and len(equity_curve) > 1:
            equity_curve['daily_return'] = equity_curve['equity'].pct_change().fillna(0)
            
            # Annualized return (assuming 252 trading days per year)
            # Using the formula: (1 + total_return)^(252/days) - 1
            days = len(equity_curve)
            annualized_return = ((1 + total_return/100) ** (252/days) - 1) * 100
            metrics['annualized_return'] = annualized_return
            
            # Volatility (annualized)
            daily_std = equity_curve['daily_return'].std()
            annualized_std = daily_std * np.sqrt(252)
            metrics['annualized_volatility'] = annualized_std * 100
            
            # Sharpe ratio (assuming 0% risk-free rate for simplicity)
            sharpe_ratio = annualized_return / (metrics['annualized_volatility'] + 1e-10)
            metrics['sharpe_ratio'] = sharpe_ratio
            
            # Maximum drawdown
            equity_curve['cummax'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['equity'] / equity_curve['cummax'] - 1) * 100
            max_drawdown = abs(equity_curve['drawdown'].min())
            metrics['max_drawdown'] = max_drawdown
            
            # Calmar ratio (annualized return / maximum drawdown)
            calmar_ratio = annualized_return / (max_drawdown + 1e-10)
            metrics['calmar_ratio'] = calmar_ratio
            
            # Sortino ratio (using downside deviation instead of standard deviation)
            negative_returns = equity_curve['daily_return'][equity_curve['daily_return'] < 0]
            downside_std = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 1e-10
            sortino_ratio = annualized_return / (downside_std * 100 + 1e-10)
            metrics['sortino_ratio'] = sortino_ratio
    else:
        metrics['annualized_return'] = 0.0
        metrics['annualized_volatility'] = 0.0
        metrics['sharpe_ratio'] = 0.0
        metrics['max_drawdown'] = 0.0
        metrics['calmar_ratio'] = 0.0
        metrics['sortino_ratio'] = 0.0
    
    return metrics