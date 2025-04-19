from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from financial_prediction_system.infrastructure.database.models_and_schemas.models import Stock, StockPrice
from .base_loader import BaseDataLoader
from .data_providers import AlpacaProvider, YahooFinanceProvider, check_missing_market_dates, get_market_date_ranges
from .cache_decorator import cacheable, invalidate_cache
from .cache import RedisCache
import pandas as pd
from financial_prediction_system.logging_config import logger

class StockDataLoader(BaseDataLoader):
    def __init__(self, db: Session, cache: Optional[RedisCache] = None):
        # Use AlpacaProvider as default strategy, but fall back to YahooFinanceProvider if it fails
        try:
            data_provider = AlpacaProvider()
            logger.info("Using Alpaca as the data provider")
        except Exception as e:
            logger.warning(f"Failed to initialize Alpaca provider: {type(e).__name__}. Falling back to Yahoo Finance")
            data_provider = YahooFinanceProvider()
            
        super().__init__(db, data_provider)
        self.cache = cache
        self.provider_name = data_provider.__class__.__name__

    def load_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Load data for a specific symbol between start_date and end_date.
        This is a facade method that calls the appropriate internal methods.
        
        Parameters
        ----------
        symbol : str
            The stock symbol to load data for
        start_date : date
            Start date for the data
        end_date : date
            End date for the data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the stock data
        """
        self._log_progress(f"Loading data for {symbol} from {start_date} to {end_date}")
        
        # For existing systems, we won't try to load new data on demand
        # This way we ensure we only use what's in the database
        try:
            # Query the database directly
            query = self.db.query(StockPrice).filter(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).order_by(StockPrice.date)
            
            data = [{
                'date': record.date,
                'open': float(record.open) if record.open is not None else None,
                'high': float(record.high) if record.high is not None else None,
                'low': float(record.low) if record.low is not None else None,
                'close': float(record.close) if record.close is not None else None,
                'volume': int(record.volume) if record.volume is not None else None
            } for record in query]
            
            if not data:
                self._log_progress(f"No data found for {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            self._handle_error(e, f"querying database for {symbol}")
            return pd.DataFrame()

    @cacheable("stock_historical")
    def load_historical_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> int:
        if not start_date:
            start_date = date(2000, 1, 1)  # Default start date
        if not end_date:
            end_date = datetime.now().date()

        total_records = 0
        all_validated_data = []
        symbols = [symbol] if symbol else self._get_eligible_stocks()

        for sym in symbols:
            try:
                validated_data, records = self._load_symbol_data(sym, start_date, end_date)
                total_records += records
                all_validated_data.extend(validated_data)
                self._log_progress(f"Loaded {records} records for {sym}")
            except Exception as e:
                self._handle_error(e, f"loading historical data for {sym}")

        # Notify observers about data quality
        if all_validated_data and self.quality_observers:
            quality_results = self.notify_data_loaded(all_validated_data, start_date, end_date, symbol)
            self._log_debug(f"Data quality results: {quality_results}")

        return total_records

    @cacheable("stock_daily")
    def update_daily_data(self, symbol: Optional[str] = None) -> int:
        latest_date = self.get_latest_date(symbol=symbol)
        if not latest_date:
            return self.load_historical_data(symbol=symbol)

        # Check for gaps in the data
        all_dates = self._check_for_data_gaps(latest_date, symbol)
        if not all_dates:
            self._log_progress("Data is up to date")
            return 0
            
        total_records = 0
        for date_range in all_dates:
            records = self.load_historical_data(date_range[0], date_range[1], symbol=symbol)
            total_records += records
        
        # Invalidate historical cache if we updated data
        if self.cache and total_records > 0:
            invalidate_cache("stock_historical", symbol)(self.cache)
            
        return total_records

    def _check_for_data_gaps(self, latest_date: date, symbol: Optional[str] = None) -> List[tuple]:
        """
        Check for gaps in data from latest_date to today using QuantLib's NYSE calendar.
        If today is Tuesday and we have Monday's data, don't try to update today's data.
        
        Args:
            latest_date: The latest date for which we have data
            symbol: Symbol to check (optional)
            
        Returns:
            List of tuples (start_date, end_date) for each gap that needs to be filled
        """
        today = datetime.now().date()
        
        # If we already have today's data, no need to update
        if latest_date >= today:
            self._log_progress(f"Data is already up to date with latest record on {latest_date}", "info")
            return []
        
        # Get existing dates from the database
        existing_dates = set()
        if symbol:
            query = self.db.query(StockPrice.date).filter(
                StockPrice.symbol == symbol,
                StockPrice.date > latest_date,
                StockPrice.date <= today
            ).distinct()
        else:
            query = self.db.query(StockPrice.date).filter(
                StockPrice.date > latest_date,
                StockPrice.date <= today
            ).distinct()
        
        existing_dates = set(date for (date,) in query)
        
        # Use the market calendar to check for missing dates
        date_ranges = get_market_date_ranges(existing_dates, latest_date, today, "NYSE")
        
        # Log the findings
        for start, end in date_ranges:
            if start == end:
                self._log_progress(f"Will fetch data for market date {start.strftime('%Y-%m-%d')}", "info")
            else:
                self._log_progress(f"Will fetch data for market dates from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", "info")
        
        return date_ranges

    def get_latest_date(self, symbol: Optional[str] = None) -> Optional[date]:
        try:
            query = self.db.query(StockPrice.date).order_by(StockPrice.date.desc())
            if symbol:
                query = query.filter(StockPrice.symbol == symbol)
            result = query.first()
            return result[0] if result else None
        except Exception as e:
            self._handle_error(e, "getting latest date")

    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated_data = []
        for record in data:
            try:
                # Basic validation
                if not all(key in record for key in ['symbol', 'date', 'open', 'high', 'low', 'close']):
                    continue
                
                # Type conversion and cleaning
                validated_record = {
                    'symbol': str(record['symbol']),
                    'date': record['date'],
                    'open': float(record['open']) if record['open'] else None,
                    'high': float(record['high']) if record['high'] else None,
                    'low': float(record['low']) if record['low'] else None,
                    'close': float(record['close']) if record['close'] else None,
                    'volume': int(record['volume']) if record['volume'] else 0  # Default to 0 instead of None for low/no volume stocks
                }
                
                # Additional validation
                if validated_record['close'] is not None and validated_record['close'] > 0:
                    validated_data.append(validated_record)
            except (ValueError, TypeError) as e:
                self._log_progress(f"Invalid record: {str(e)}", "warning")
                continue

        return validated_data

    def _get_eligible_stocks(self) -> List[str]:
        try:
            stocks = self.db.query(Stock.symbol).filter(
                and_(
                    Stock.quotetype == 'EQUITY',
                    Stock.sector.isnot(None),
                    Stock.is_active == True
                )
            ).all()
            return [stock[0] for stock in stocks]
        except Exception as e:
            self._handle_error(e, "getting eligible stocks")

    def _load_symbol_data(self, symbol: str, start_date: date, end_date: date) -> tuple:
        try:
            # First check if we already have data for these dates
            existing_dates = set()
            query = self.db.query(StockPrice.date).filter(
                StockPrice.symbol == symbol,
                StockPrice.date >= start_date,
                StockPrice.date <= end_date
            ).all()
            existing_dates = set(date for (date,) in query)
            
            # If we already have all the data for all the days in the range, return early
            if existing_dates:
                weekday_dates = set()
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Only weekdays
                        weekday_dates.add(current_date)
                    current_date += timedelta(days=1)
                
                # Check if we have data for all weekdays
                missing_dates = weekday_dates - existing_dates
                if not missing_dates:
                    self._log_progress(f"Already have data for {symbol} from {start_date} to {end_date}", "debug")
                    return [], 0
                
                # Adjust start_date and end_date to only fetch missing dates
                # But only if we're not looking for today's data, which might be updated throughout the day
                today = datetime.now().date()
                if not any(d == today for d in missing_dates):
                    # Create a new range covering just the missing dates
                    date_ranges = get_market_date_ranges(existing_dates, start_date, end_date, "NYSE")
                    if not date_ranges:
                        return [], 0
                    
                    # Load each range individually
                    all_validated_records = []
                    total_records = 0
                    for range_start, range_end in date_ranges:
                        self._log_progress(f"Fetching missing data for {symbol} from {range_start} to {range_end}", "debug")
                        validated_data, records = self._fetch_and_process_data(symbol, range_start, range_end)
                        all_validated_records.extend(validated_data)
                        total_records += records
                    
                    return all_validated_records, total_records
            
            # Use the data provider to fetch data for the full range
            return self._fetch_and_process_data(symbol, start_date, end_date)
        except Exception as e:
            self._handle_error(e, f"loading data for {symbol}")
            return [], 0
    
    def _fetch_and_process_data(self, symbol: str, start_date: date, end_date: date) -> tuple:
        """Helper to fetch and process data from the provider"""
        self._log_progress(f"Fetching data for {symbol} from {start_date} to {end_date}", "debug")
        records = self.data_provider.fetch_data(start_date, end_date, symbol)
        
        if not records:
            self._log_progress(f"No data returned from provider for {symbol} between {start_date} and {end_date}", "warning")
            # In case of no data, check if this is a special case for certain dates
            if (end_date - start_date).days <= 5:  # If we're fetching a small date range
                # Try to fill in with the most recent data available (appropriate for low volume stocks)
                self._log_progress(f"Checking if we need to fill in missing data for low volume stock {symbol}", "debug")
                last_record = self.db.query(StockPrice).filter(
                    StockPrice.symbol == symbol,
                    StockPrice.date < start_date
                ).order_by(StockPrice.date.desc()).first()
                
                if last_record:
                    self._log_progress(f"Found previous data for {symbol}, will use for filling", "debug")
                    # Create records for each day using the last available price
                    current_date = start_date
                    filled_records = []
                    while current_date <= end_date:
                        # Skip weekends
                        if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                            filled_records.append({
                                'symbol': symbol,
                                'date': current_date,
                                'open': last_record.close,
                                'high': last_record.close,
                                'low': last_record.close,
                                'close': last_record.close,
                                'volume': 0  # Zero volume indicates no trading
                            })
                            self._log_progress(f"Filled missing data for {symbol} on {current_date} with previous close", "info")
                        current_date += timedelta(days=1)
                    
                    # Validate and save these records
                    if filled_records:
                        validated_records = self.validate_data(filled_records)
                        self._save_records(validated_records)
                        return validated_records, len(validated_records)
            
            return [], 0
            
        self._log_progress(f"Received {len(records)} records for {symbol}", "debug")
        
        # Check for specific dates in the returned data
        received_dates = {r['date'] for r in records}
        expected_dates = set()
        check_date = start_date
        while check_date <= end_date:
            if check_date.weekday() < 5:  # Skip weekends
                expected_dates.add(check_date)
            check_date += timedelta(days=1)
            
        missing_dates = expected_dates - received_dates
        if missing_dates:
            self._log_progress(f"Provider missing data for {symbol} on dates: {sorted(missing_dates)}", "warning")
            
        validated_records = self.validate_data(records)
        self._save_records(validated_records)
        return validated_records, len(validated_records)

    def _save_records(self, records: List[Dict[str, Any]]):
        try:
            for record in records:
                # Check if this record already exists
                existing = self.db.query(StockPrice).filter(
                    StockPrice.symbol == record['symbol'],
                    StockPrice.date == record['date']
                ).first()
                
                # Only save if it doesn't exist or we have a more complete record
                # (existing record has 0 volume but new record has real data)
                if not existing or (
                    existing and (
                        existing.volume == 0 and record['volume'] > 0
                    )
                ):
                    price = StockPrice(**record)
                    self.db.merge(price)  # Use merge for upsert functionality
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self._handle_error(e, "saving records") 