"""
Core Market Data Module

Provides a unified interface for retrieving market data from various sources.
Implements the Strategy Pattern for different data retrieval methods.

Base data types:
- OHLCV: open, high, low, close, volume
- Returns: daily_return, log_return, cumulative_return
- Price Relatives: open-close, high-low, close/prev_close
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text

from financial_prediction_system.infrastructure.database.connection import DataSource, DataSourceFactory, DatabaseConnection


class MarketDataProvider(ABC):
    """Abstract base class defining the interface for all market data providers"""
    
    @abstractmethod
    def get_ohlcv(self, symbol: str, start_date: Union[str, datetime], 
                 end_date: Union[str, datetime], interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve OHLCV data for a specific symbol
        
        Args:
            symbol: The ticker or identifier for the asset
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval (e.g., "1d", "1h", "5m")
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols
        
        Returns:
            List of symbol strings
        """
        pass


class DatabaseMarketDataProvider(MarketDataProvider):
    """Implementation of MarketDataProvider that retrieves data from a database"""
    
    def __init__(self, data_source: Optional[DataSource] = None):
        """
        Initialize with a data source
        
        Args:
            data_source: DataSource implementation, uses default if None
        """
        self._data_source = data_source or DataSourceFactory.create_database_connection()
    
    def get_ohlcv(self, symbol: str, start_date: Union[str, datetime], 
                 end_date: Union[str, datetime], interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve OHLCV data from database
        
        Args:
            symbol: The ticker or identifier for the asset
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval (e.g., "1d", "1h", "5m")
            
        Returns:
            DataFrame with OHLCV data
        """
        # Convert date strings to datetime objects if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        # Get a database session
        session = self._data_source.get_session()
        
        try:
            # Example SQL query using SQLAlchemy
            # This would need to be adjusted based on your actual database schema
            query = text("""
                SELECT date, open, high, low, close, volume
                FROM market_data
                WHERE symbol = :symbol
                AND date BETWEEN :start_date AND :end_date
                AND interval = :interval
                ORDER BY date ASC
            """)
            
            result = session.execute(query, {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date,
                "interval": interval
            })
            
            # Convert to DataFrame
            df = pd.DataFrame(result.fetchall(), 
                              columns=["date", "open", "high", "low", "close", "volume"])
            
            # Set date as index
            if not df.empty:
                df.set_index("date", inplace=True)
                
            return df
            
        finally:
            session.close()
    
    def get_symbols(self) -> List[str]:
        """
        Get list of available symbols from the database
        
        Returns:
            List of symbol strings
        """
        session = self._data_source.get_session()
        
        try:
            # Example query to get distinct symbols
            query = text("SELECT DISTINCT symbol FROM market_data ORDER BY symbol")
            result = session.execute(query)
            return [row[0] for row in result.fetchall()]
        finally:
            session.close()


class MarketDataProviderFactory:
    """Factory for creating different types of market data providers"""
    
    @staticmethod
    def create_database_provider(data_source: Optional[DataSource] = None) -> DatabaseMarketDataProvider:
        """
        Create and return a database market data provider
        
        Args:
            data_source: Optional custom data source
            
        Returns:
            Configured DatabaseMarketDataProvider
        """
        return DatabaseMarketDataProvider(data_source)


class MarketDataService:
    """
    Service class that provides market data and derived financial features
    
    This class implements the Strategy Pattern by allowing different
    MarketDataProvider strategies to be used.
    """
    
    def __init__(self, provider: Optional[MarketDataProvider] = None):
        """
        Initialize the service with a data provider
        
        Args:
            provider: MarketDataProvider implementation, uses database provider if None
        """
        self._provider = provider or MarketDataProviderFactory.create_database_provider()
    
    def get_ohlcv(self, symbol: str, start_date: Union[str, datetime], 
                 end_date: Union[str, datetime], interval: str = "1d") -> pd.DataFrame:
        """
        Get raw OHLCV data
        
        Args:
            symbol: The ticker or identifier for the asset
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            interval: Data interval (e.g., "1d", "1h", "5m")
            
        Returns:
            DataFrame with OHLCV data
        """
        return self._provider.get_ohlcv(symbol, start_date, end_date, interval)
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols
        
        Returns:
            List of symbol strings
        """
        return self._provider.get_symbols()
    
    def calculate_returns(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with at least a 'close' column
            
        Returns:
            DataFrame with added return columns
        """
        df = ohlcv_data.copy()
        
        # Daily returns (percentage change)
        df['daily_return'] = df['close'].pct_change()
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        return df
    
    def calculate_price_relatives(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price relative metrics from OHLCV data
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added price relative columns
        """
        df = ohlcv_data.copy()
        
        # Open-Close (intraday change)
        df['open_close'] = df['close'] - df['open']
        
        # High-Low (volatility)
        df['high_low'] = df['high'] - df['low']
        
        # Close/Prev Close (day-to-day ratio)
        df['close_prev_close'] = df['close'] / df['close'].shift(1)
        
        return df