import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock

from financial_prediction_system.core.features.core_market_data.core import (
    MarketDataProvider, DatabaseMarketDataProvider, 
    MarketDataProviderFactory, MarketDataService
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    return pd.DataFrame({
        'date': pd.date_range(start='2022-01-01', periods=5),
        'open': [100.0, 101.0, 103.0, 102.0, 105.0],
        'high': [102.0, 103.0, 105.0, 106.0, 107.0],
        'low': [99.0, 100.0, 101.0, 101.0, 103.0],
        'close': [101.0, 102.0, 104.0, 105.0, 106.0],
        'volume': [1000, 1100, 900, 1200, 1000]
    }).set_index('date')


class MockMarketDataProvider(MarketDataProvider):
    """Mock implementation of MarketDataProvider for testing"""
    
    def __init__(self, ohlcv_data=None):
        """Initialize with optional predefined data"""
        self.ohlcv_data = pd.DataFrame() if ohlcv_data is None else ohlcv_data
        self.symbols = ["AAPL", "MSFT", "GOOG"]
    
    def get_ohlcv(self, symbol, start_date, end_date, interval="1d"):
        """Return predefined data"""
        return self.ohlcv_data
    
    def get_symbols(self):
        """Return predefined symbols"""
        return self.symbols


class TestDatabaseMarketDataProvider:
    """Test suite for DatabaseMarketDataProvider"""
    
    @patch('financial_prediction_system.infrastructure.database.connection.DatabaseConnection')
    def test_initialization(self, mock_db_connection):
        """Test provider initialization with data source"""
        # Setup
        mock_data_source = MagicMock()
        
        # Create provider
        provider = DatabaseMarketDataProvider(mock_data_source)
        
        # Verify data source was set
        assert provider._data_source is mock_data_source
    
    @patch('financial_prediction_system.core.features.core_market_data.core.DataSourceFactory')
    def test_default_data_source(self, mock_factory):
        """Test provider uses factory when no data source is provided"""
        # Setup
        mock_data_source = MagicMock()
        mock_factory.create_database_connection.return_value = mock_data_source
        
        # Create provider without explicit data source
        provider = DatabaseMarketDataProvider()
        
        # Verify factory was used
        mock_factory.create_database_connection.assert_called_once()
        
        # Verify data source from factory was set
        assert provider._data_source is mock_data_source
    
    def test_get_ohlcv(self):
        """Test get_ohlcv method executes correct database query"""
        # Setup
        mock_data_source = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Configure mocks
        mock_data_source.get_session.return_value = mock_session
        mock_session.execute.return_value = mock_result
        mock_result.fetchall.return_value = [
            ('2022-01-01', 100.0, 102.0, 99.0, 101.0, 1000),
            ('2022-01-02', 101.0, 103.0, 100.0, 102.0, 1100)
        ]
        
        # Create provider
        provider = DatabaseMarketDataProvider(mock_data_source)
        
        # Call method
        df = provider.get_ohlcv('AAPL', '2022-01-01', '2022-01-31')
        
        # Verify session was requested
        mock_data_source.get_session.assert_called_once()
        
        # Verify execute was called with correct parameters
        mock_session.execute.assert_called_once()
        
        # Verify result is a proper DataFrame with the expected structure
        assert isinstance(df, pd.DataFrame)
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert len(df) == 2
    
    def test_get_symbols(self):
        """Test get_symbols method returns correct symbols list"""
        # Setup
        mock_data_source = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        
        # Configure mocks
        mock_data_source.get_session.return_value = mock_session
        mock_session.execute.return_value = mock_result
        mock_result.fetchall.return_value = [('AAPL',), ('MSFT',), ('GOOG',)]
        
        # Create provider
        provider = DatabaseMarketDataProvider(mock_data_source)
        
        # Call method
        symbols = provider.get_symbols()
        
        # Verify session was requested
        mock_data_source.get_session.assert_called_once()
        
        # Verify execute was called
        mock_session.execute.assert_called_once()
        
        # Verify result
        assert symbols == ['AAPL', 'MSFT', 'GOOG']


class TestMarketDataProviderFactory:
    """Test suite for MarketDataProviderFactory"""
    
    @patch('financial_prediction_system.core.features.core_market_data.core.DatabaseMarketDataProvider')
    def test_create_database_provider(self, mock_provider_class):
        """Test factory creates database provider correctly"""
        # Setup
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        mock_data_source = MagicMock()
        
        # Call factory method
        provider = MarketDataProviderFactory.create_database_provider(mock_data_source)
        
        # Verify provider class was instantiated correctly
        mock_provider_class.assert_called_once_with(mock_data_source)
        
        # Verify correct provider was returned
        assert provider is mock_provider


class TestMarketDataService:
    """Test suite for MarketDataService"""
    
    def test_initialization_with_provider(self):
        """Test service initialization with explicit provider"""
        # Setup
        mock_provider = MagicMock()
        
        # Create service
        service = MarketDataService(mock_provider)
        
        # Verify provider was set
        assert service._provider is mock_provider
    
    @patch('financial_prediction_system.core.features.core_market_data.core.MarketDataProviderFactory')
    def test_initialization_default_provider(self, mock_factory):
        """Test service uses factory when no provider is specified"""
        # Setup
        mock_provider = MagicMock()
        mock_factory.create_database_provider.return_value = mock_provider
        
        # Create service without explicit provider
        service = MarketDataService()
        
        # Verify factory was used
        mock_factory.create_database_provider.assert_called_once()
        
        # Verify provider from factory was set
        assert service._provider is mock_provider
    
    def test_get_ohlcv_delegates_to_provider(self):
        """Test get_ohlcv method delegates to provider"""
        # Setup
        mock_provider = MagicMock()
        expected_df = pd.DataFrame({'close': [101, 102]})
        mock_provider.get_ohlcv.return_value = expected_df
        
        # Create service
        service = MarketDataService(mock_provider)
        
        # Call method
        result = service.get_ohlcv('AAPL', '2022-01-01', '2022-01-31', '1d')
        
        # Verify provider method was called with correct parameters
        mock_provider.get_ohlcv.assert_called_once_with('AAPL', '2022-01-01', '2022-01-31', '1d')
        
        # Verify result
        assert result is expected_df
    
    def test_get_available_symbols_delegates_to_provider(self):
        """Test get_available_symbols method delegates to provider"""
        # Setup
        mock_provider = MagicMock()
        expected_symbols = ['AAPL', 'MSFT']
        mock_provider.get_symbols.return_value = expected_symbols
        
        # Create service
        service = MarketDataService(mock_provider)
        
        # Call method
        result = service.get_available_symbols()
        
        # Verify provider method was called
        mock_provider.get_symbols.assert_called_once()
        
        # Verify result
        assert result is expected_symbols
    
    def test_calculate_returns(self, sample_ohlcv_data):
        """Test calculate_returns method computes return metrics correctly"""
        # Setup
        service = MarketDataService(MockMarketDataProvider())
        
        # Call method
        result = service.calculate_returns(sample_ohlcv_data)
        
        # Verify result has expected columns
        assert 'daily_return' in result.columns
        assert 'log_return' in result.columns
        assert 'cumulative_return' in result.columns
        
        # Verify calculations
        # First daily return should be NaN (no previous day)
        assert np.isnan(result['daily_return'].iloc[0])
        
        # Calculate expected values for the second row
        expected_daily_return = (102.0 - 101.0) / 101.0
        expected_log_return = np.log(102.0 / 101.0)
        
        # Check values (using almost equal for floating point comparison)
        assert result['daily_return'].iloc[1] == pytest.approx(expected_daily_return)
        assert result['log_return'].iloc[1] == pytest.approx(expected_log_return)
    
    def test_calculate_price_relatives(self, sample_ohlcv_data):
        """Test calculate_price_relatives method computes price metrics correctly"""
        # Setup
        service = MarketDataService(MockMarketDataProvider())
        
        # Call method
        result = service.calculate_price_relatives(sample_ohlcv_data)
        
        # Verify result has expected columns
        assert 'open_close' in result.columns
        assert 'high_low' in result.columns
        assert 'close_prev_close' in result.columns
        
        # Verify calculations for the first row
        assert result['open_close'].iloc[0] == 101.0 - 100.0
        assert result['high_low'].iloc[0] == 102.0 - 99.0
        
        # First close/prev_close should be NaN (no previous day)
        assert np.isnan(result['close_prev_close'].iloc[0])
        
        # Check second row close/prev_close
        assert result['close_prev_close'].iloc[1] == 102.0 / 101.0


def test_integration(sample_ohlcv_data):
    """Test the full data pipeline from provider to calculated metrics"""
    # Setup
    mock_provider = MockMarketDataProvider(sample_ohlcv_data)
    service = MarketDataService(mock_provider)
    
    # Get data and calculate metrics
    data = service.get_ohlcv('AAPL', '2022-01-01', '2022-01-31')
    data_with_returns = service.calculate_returns(data)
    final_data = service.calculate_price_relatives(data_with_returns)
    
    # Verify all metrics are present
    assert 'open' in final_data.columns
    assert 'high' in final_data.columns
    assert 'low' in final_data.columns
    assert 'close' in final_data.columns
    assert 'volume' in final_data.columns
    assert 'daily_return' in final_data.columns
    assert 'log_return' in final_data.columns
    assert 'cumulative_return' in final_data.columns
    assert 'open_close' in final_data.columns
    assert 'high_low' in final_data.columns
    assert 'close_prev_close' in final_data.columns 