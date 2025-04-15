from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Protocol, Set, Callable
from sqlalchemy.orm import Session
from financial_prediction_system.logging_config import logger

class DataProviderStrategy(Protocol):
    """Strategy for data retrieval from different sources"""
    
    def fetch_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch data from source for given date range and symbol"""
        ...

class DataQualityObserver(Protocol):
    """Observer for data quality events"""
    
    def on_data_loaded(self, loader_type: str, data: List[Dict[str, Any]], start_date: Optional[date], end_date: Optional[date], symbol: Optional[str]) -> Dict[str, Any]:
        """Called when data is loaded by a loader"""
        ...

class BaseDataLoader(ABC):
    def __init__(self, db: Session, data_provider: Optional[DataProviderStrategy] = None):
        self.db = db
        self.data_provider = data_provider
        self.quality_observers: Set[DataQualityObserver] = set()
        logger.debug(f"Initialized {self.__class__.__name__}")
        self.logger = logger

    def register_quality_observer(self, observer: DataQualityObserver) -> None:
        """Register an observer for data quality events"""
        self.quality_observers.add(observer)
        logger.debug(f"Registered quality observer: {observer.__class__.__name__}")
        
    def unregister_quality_observer(self, observer: DataQualityObserver) -> None:
        """Unregister an observer for data quality events"""
        self.quality_observers.discard(observer)
        logger.debug(f"Unregistered quality observer: {observer.__class__.__name__}")
        
    def notify_data_loaded(self, data: List[Dict[str, Any]], start_date: Optional[date], end_date: Optional[date], symbol: Optional[str]) -> Dict[str, Dict[str, Any]]:
        """Notify all observers that data has been loaded"""
        quality_results = {}
        loader_type = self.__class__.__name__.replace('DataLoader', '').lower()
        
        for observer in self.quality_observers:
            try:
                result = observer.on_data_loaded(loader_type, data, start_date, end_date, symbol)
                quality_results[observer.__class__.__name__] = result
            except Exception as e:
                self._log_error(f"Error in quality observer: {str(e)}", e)
                
        return quality_results

    @abstractmethod
    def load_historical_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> int:
        """Load historical data for the given date range"""
        pass

    @abstractmethod
    def update_daily_data(self, symbol: Optional[str] = None) -> int:
        """Update data with the latest available information"""
        pass

    @abstractmethod
    def get_latest_date(self, symbol: Optional[str] = None) -> Optional[date]:
        """Get the latest date for which data exists"""
        pass

    @abstractmethod
    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean the data before insertion"""
        pass

    def _log_progress(self, message: str, level: str = "info"):
        """Helper method for consistent logging"""
        log_method = getattr(self.logger, level.lower())
        log_method(message)

    def _handle_error(self, error: Exception, context: str = ""):
        """Helper method for consistent error handling"""
        self._log_progress(f"Error in {context}: {str(error)}", "error")
        raise error

    def _log_error(self, message: str, exception: Exception = None):
        """Helper method for consistent error logging"""
        if exception:
            self.logger.error(f"{message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(message)

    def _log_info(self, message: str):
        """Helper method for consistent info logging"""
        self.logger.info(message)

    def _log_debug(self, message: str):
        """Helper method for consistent debug logging"""
        self.logger.debug(message) 