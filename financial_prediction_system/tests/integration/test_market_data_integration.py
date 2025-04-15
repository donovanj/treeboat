import pytest
import pandas as pd
import os
from datetime import datetime, timedelta
from sqlalchemy import text

from financial_prediction_system.infrastructure.database.connection import (
    DatabaseConnection, DataSourceFactory
)
from financial_prediction_system.core.features.core_market_data.core import (
    DatabaseMarketDataProvider, MarketDataService
)


@pytest.fixture
def setup_test_data(test_db):
    """Create test market data table and populate with sample data"""
    # Create market_data table
    test_db.execute(text("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TIMESTAMP NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            interval TEXT NOT NULL,
            UNIQUE(symbol, date, interval)
        )
    """))
    
    # First clean existing data to avoid conflicts
    test_db.execute(text("DELETE FROM market_data"))
    
    # Insert sample data
    today = datetime.now().date()
    
    symbols = ['AAPL', 'MSFT', 'GOOG']
    intervals = ['1d', '1h']
    
    for symbol in symbols:
        for i in range(10):  # 10 days of data
            date = today - timedelta(days=i)
            for interval in intervals:
                base_price = 100 + (i % 5)
                
                test_db.execute(text("""
                    INSERT INTO market_data 
                    (symbol, date, open, high, low, close, volume, interval)
                    VALUES 
                    (:symbol, :date, :open, :high, :low, :close, :volume, :interval)
                """), {
                    'symbol': symbol,
                    'date': date,
                    'open': base_price,
                    'high': base_price + 2,
                    'low': base_price - 1,
                    'close': base_price + 1,
                    'volume': 1000 + (i * 100),
                    'interval': interval
                })
    
    test_db.commit()
    
    return test_db


class TestMarketDataIntegration:
    """Integration tests for market data system with database"""
    
    def test_retrieve_data_from_database(self, setup_test_data, monkeypatch):
        """Test retrieving market data from database"""
        # Mock the engine creation to use the test database
        test_db = setup_test_data
        
        # Create database connection pointing to test database
        db_conn = DatabaseConnection()
        monkeypatch.setattr(db_conn, '_engine', test_db.get_bind())
        monkeypatch.setattr(db_conn, '_session_factory', lambda: test_db)
        
        # Create market data provider
        provider = DatabaseMarketDataProvider(db_conn)
        
        # Create service
        service = MarketDataService(provider)
        
        # Get available symbols
        symbols = service.get_available_symbols()
        assert len(symbols) == 3
        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert 'GOOG' in symbols
        
        # Get OHLCV data
        today = datetime.now().date()
        five_days_ago = today - timedelta(days=5)
        
        data = service.get_ohlcv('AAPL', five_days_ago, today)
        
        # Verify data
        assert not data.empty
        assert len(data) <= 6  # Should have at most 6 days (including today)
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_calculate_derived_metrics(self, setup_test_data, monkeypatch):
        """Test calculating derived metrics from database data"""
        # Mock the engine creation to use the test database
        test_db = setup_test_data
        
        # Create database connection pointing to test database
        db_conn = DatabaseConnection()
        monkeypatch.setattr(db_conn, '_engine', test_db.get_bind())
        monkeypatch.setattr(db_conn, '_session_factory', lambda: test_db)
        
        # Create market data provider
        provider = DatabaseMarketDataProvider(db_conn)
        
        # Create service
        service = MarketDataService(provider)
        
        # Get OHLCV data
        today = datetime.now().date()
        five_days_ago = today - timedelta(days=5)
        
        data = service.get_ohlcv('AAPL', five_days_ago, today)
        
        # Calculate returns
        returns_data = service.calculate_returns(data)
        
        # Verify return calculations
        assert 'daily_return' in returns_data.columns
        assert 'log_return' in returns_data.columns
        assert 'cumulative_return' in returns_data.columns
        
        # Calculate price relatives
        price_relatives = service.calculate_price_relatives(returns_data)
        
        # Verify price relative calculations
        assert 'open_close' in price_relatives.columns
        assert 'high_low' in price_relatives.columns
        assert 'close_prev_close' in price_relatives.columns
        
        # Verify specific calculations (first row)
        first_row = price_relatives.iloc[0]
        assert first_row['open_close'] == first_row['close'] - first_row['open']
        assert first_row['high_low'] == first_row['high'] - first_row['low']
    
    def test_full_data_pipeline(self, setup_test_data, monkeypatch):
        """Test the full data pipeline from retrieval to analysis"""
        # Mock the engine creation to use the test database
        test_db = setup_test_data
        
        # Create database connection pointing to test database
        db_conn = DatabaseConnection()
        monkeypatch.setattr(db_conn, '_engine', test_db.get_bind())
        monkeypatch.setattr(db_conn, '_session_factory', lambda: test_db)
        
        # Create market data provider using factory
        monkeypatch.setattr(DataSourceFactory, 'create_database_connection', lambda _=None: db_conn)
        
        # Create service with default factory
        service = MarketDataService()
        
        # Get available symbols
        symbols = service.get_available_symbols()
        
        # Process each symbol
        for symbol in symbols:
            # Get data for last 10 days
            today = datetime.now().date()
            ten_days_ago = today - timedelta(days=10)
            
            # Get raw data
            data = service.get_ohlcv(symbol, ten_days_ago, today)
            
            # Calculate metrics
            data = service.calculate_returns(data)
            data = service.calculate_price_relatives(data)
            
            # Verify we have all expected columns
            assert 'open' in data.columns
            assert 'high' in data.columns
            assert 'low' in data.columns
            assert 'close' in data.columns
            assert 'volume' in data.columns
            assert 'daily_return' in data.columns
            assert 'log_return' in data.columns
            assert 'cumulative_return' in data.columns
            assert 'open_close' in data.columns
            assert 'high_low' in data.columns
            assert 'close_prev_close' in data.columns
            
            # Calculate some aggregate statistics
            mean_daily_return = data['daily_return'].mean()
            mean_high_low = data['high_low'].mean()
            
            # Just verify these don't error
            assert isinstance(mean_daily_return, float) or pd.isna(mean_daily_return)
            assert isinstance(mean_high_low, float) 