"""Schemas for backtest endpoints"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

from .base import BaseResponse, PaginatedResponse

class BacktestRequest(BaseModel):
    """Request model for creating a new backtest"""
    model_id: int = Field(..., description="ID of the model to backtest")
    symbol: str = Field(..., description="Symbol to backtest")
    start_date: datetime = Field(..., description="Start date for backtest")
    end_date: datetime = Field(..., description="End date for backtest")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Backtest parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_id": 1,
                "symbol": "SPX",
                "start_date": "2022-01-01T00:00:00",
                "end_date": "2022-12-31T00:00:00",
                "parameters": {
                    "threshold": 0.75,
                    "trade_size": 100000
                }
            }
        }

class EnsembleBacktestRequest(BaseModel):
    """Request model for creating an ensemble backtest"""
    ensemble_id: int = Field(..., description="ID of the ensemble to backtest")
    symbol: str = Field(..., description="Symbol to backtest")
    start_date: datetime = Field(..., description="Start date for backtest")
    end_date: datetime = Field(..., description="End date for backtest")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Backtest parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ensemble_id": 1,
                "symbol": "SPX",
                "start_date": "2022-01-01T00:00:00",
                "end_date": "2022-12-31T00:00:00",
                "parameters": {
                    "threshold": 0.75,
                    "trade_size": 100000
                }
            }
        }

class BacktestFilterRequest(BaseModel):
    """Request model for filtering backtests"""
    symbol: Optional[str] = Field(None, description="Filter by symbol")
    model_id: Optional[int] = Field(None, description="Filter by model ID")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    min_return: Optional[float] = Field(None, description="Minimum return")
    page: int = Field(1, description="Page number")
    page_size: int = Field(20, description="Page size")

class BacktestMetrics(BaseModel):
    """Model for backtest performance metrics"""
    total_return: float = Field(..., description="Total return percentage")
    annualized_return: float = Field(..., description="Annualized return percentage")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    calmar_ratio: Optional[float] = Field(None, description="Calmar ratio")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    average_profit: float = Field(..., description="Average profit per trade")
    average_loss: float = Field(..., description="Average loss per trade")
    alpha: Optional[float] = Field(None, description="Alpha")
    beta: Optional[float] = Field(None, description="Beta")
    information_ratio: Optional[float] = Field(None, description="Information Ratio")

class BacktestResult(BaseModel):
    """Model for backtest results"""
    id: int = Field(..., description="Backtest ID")
    symbol: str = Field(..., description="Symbol")
    start_date: datetime = Field(..., description="Start date")
    end_date: datetime = Field(..., description="End date")
    performance_metrics: BacktestMetrics = Field(..., description="Performance metrics")
    selected_features: Optional[List[str]] = Field(None, description="Features used in the model")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

class BacktestResponse(BaseResponse):
    """Response model for backtest details"""
    backtest: BacktestResult = Field(..., description="Backtest details")
    model_id: int = Field(..., description="ID of the model used")
    trades: Optional[List[Dict[str, Any]]] = Field(None, description="Trade history")
    equity_curve: Optional[List[Dict[str, Any]]] = Field(None, description="Equity curve")

class BacktestListResponse(PaginatedResponse):
    """Response model for listing backtests"""
    backtests: List[BacktestResult] = Field(..., description="List of backtests")
