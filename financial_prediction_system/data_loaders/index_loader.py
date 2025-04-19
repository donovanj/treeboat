from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_
import yfinance as yf
from .base_loader import BaseDataLoader
import pandas as pd
from sqlalchemy import text, func
from financial_prediction_system.logging_config import logger
from financial_prediction_system.infrastructure.database.models_and_schemas.models import SPXPrice, NDXPrice, DJIPrice, RUTPrice, VIXPrice, SOXPrice, OSXPrice
from .data_providers import YahooFinanceProvider, check_missing_market_dates, get_market_date_ranges
from .cache_decorator import cacheable, invalidate_cache
from .cache import RedisCache

class IndexDataLoader(BaseDataLoader):
    def __init__(self, db: Session, cache: Optional[RedisCache] = None):
        # Use YahooFinanceProvider as default strategy
        data_provider = YahooFinanceProvider()
        super().__init__(db, data_provider)
        self.cache = cache
        
        self.index_configs = [
            {"yahoo_symbol": "^SPX", "db_symbol": "SPX", "table_name": "spx_prices"},
            {"yahoo_symbol": "^NDX", "db_symbol": "NDX", "table_name": "ndx_prices"},
            {"yahoo_symbol": "^DJI", "db_symbol": "DJI", "table_name": "dji_prices"},
            {"yahoo_symbol": "^RUT", "db_symbol": "RUT", "table_name": "rut_prices"},
            {"yahoo_symbol": "^VIX", "db_symbol": "VIX", "table_name": "vix_prices"},
            {"yahoo_symbol": "^SOX", "db_symbol": "SOX", "table_name": "sox_prices"},
            {"yahoo_symbol": "^OSX", "db_symbol": "OSX", "table_name": "osx_prices"}
        ]
        self.symbol_map = {config['db_symbol']: config['yahoo_symbol'] for config in self.index_configs}
        self.model_map = {
            "spx_prices": SPXPrice,
            "ndx_prices": NDXPrice,
            "dji_prices": DJIPrice,
            "rut_prices": RUTPrice,
            "vix_prices": VIXPrice,
            "sox_prices": SOXPrice,
            "osx_prices": OSXPrice
        }

    def load_data(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Load data for a specific index symbol between start_date and end_date.
        This is a facade method that calls the appropriate internal methods.
        
        Parameters
        ----------
        symbol : str
            The index symbol to load data for (e.g., 'SPX', 'VIX')
        start_date : date
            Start date for the data
        end_date : date
            End date for the data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing the index data
        """
        self._log_progress(f"Loading data for index {symbol} from {start_date} to {end_date}")
        
        # Find the config for this symbol
        config = None
        for cfg in self.index_configs:
            if cfg['db_symbol'] == symbol:
                config = cfg
                break
                
        if not config:
            self._log_progress(f"Unknown index symbol: {symbol}", "error")
            return pd.DataFrame()
        
        # Don't automatically load data - only use what's in the database
        try:
            # Query the database directly
            model_class = self.model_map.get(config['table_name'])
            if not model_class:
                raise ValueError(f"Unknown table name: {config['table_name']}")
                
            query = self.db.query(model_class).filter(
                model_class.symbol == symbol,
                model_class.date >= start_date,
                model_class.date <= end_date
            ).order_by(model_class.date)
            
            data = [{
                'date': record.date,
                'open': float(record.open) if record.open is not None else None,
                'high': float(record.high) if record.high is not None else None,
                'low': float(record.low) if record.low is not None else None,
                'close': float(record.close) if record.close is not None else None,
                'volume': int(record.volume) if record.volume is not None else None
            } for record in query]
            
            if not data:
                self._log_progress(f"No data found for index {symbol} from {start_date} to {end_date}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df
            
        except Exception as e:
            self._handle_error(e, f"querying database for index {symbol}")
            return pd.DataFrame()

    @cacheable("index_historical")
    def load_historical_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> int:
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()

        total_records = 0
        all_validated_data = []
        configs = [c for c in self.index_configs if not symbol or c['db_symbol'] == symbol]

        for config in configs:
            try:
                validated_data, records = self._load_index_data(config, start_date, end_date)
                total_records += records
                all_validated_data.extend(validated_data)
                self._log_progress(f"Loaded {records} records for {config['db_symbol']}")
            except Exception as e:
                self._handle_error(e, f"loading historical data for {config['db_symbol']}")

        # Notify observers about data quality
        if all_validated_data and self.quality_observers:
            quality_results = self.notify_data_loaded(all_validated_data, start_date, end_date, symbol)
            self._log_debug(f"Data quality results: {quality_results}")

        return total_records

    @cacheable("index_daily")
    def update_daily_data(self, symbol: Optional[str] = None) -> int:
        total_records = 0
        all_validated_data = []
        today = datetime.now().date()
        configs = [c for c in self.index_configs if not symbol or c['db_symbol'] == symbol]

        for config in configs:
            try:
                latest_date = self._get_latest_date_for_index(config['table_name'], symbol=config['db_symbol'])
                if not latest_date:
                    validated_data, records = self._load_index_data(config)
                else:
                    # Check for gaps in the data
                    date_ranges = self._check_for_data_gaps(latest_date, config['table_name'], config['db_symbol'])
                    if not date_ranges:
                        self._log_progress(f"{config['db_symbol']} is up to date")
                        continue
                    
                    # Process each date range to fill gaps
                    records_for_symbol = 0
                    validated_data_for_symbol = []
                    for start_date, end_date in date_ranges:
                        data, rec = self._load_index_data(config, start_date, end_date)
                        validated_data_for_symbol.extend(data)
                        records_for_symbol += rec
                    
                    validated_data = validated_data_for_symbol
                    records = records_for_symbol
                
                total_records += records
                all_validated_data.extend(validated_data)
                
                # Invalidate cache for this symbol
                if self.cache and records > 0:
                    invalidate_cache("index_historical", config['db_symbol'])(self.cache)
            except Exception as e:
                self._handle_error(e, f"updating daily data for {config['db_symbol']}")

        # Notify observers about data quality
        if all_validated_data and self.quality_observers:
            # Use today's date for daily updates
            quality_results = self.notify_data_loaded(all_validated_data, today - timedelta(days=7), today, symbol)
            self._log_debug(f"Data quality results: {quality_results}")

        return total_records

    def get_latest_date(self, symbol: Optional[str] = None) -> Optional[date]:
        latest_dates = []
        configs = [c for c in self.index_configs if not symbol or c['db_symbol'] == symbol]

        for config in configs:
            try:
                date = self._get_latest_date_for_index(config['table_name'], symbol=config['db_symbol'])
                if date:
                    latest_dates.append(date)
            except Exception as e:
                self._log_progress(f"Error getting latest date for {config['db_symbol']}: {str(e)}", "warning")

        return min(latest_dates) if latest_dates else None

    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated_data = []
        for record in data:
            try:
                if not all(key in record for key in ['symbol', 'date', 'open', 'high', 'low', 'close']):
                    continue

                validated_record = {
                    'symbol': str(record['symbol']),
                    'date': record['date'],
                    'open': float(record['open']) if pd.notna(record['open']) else None,
                    'high': float(record['high']) if pd.notna(record['high']) else None,
                    'low': float(record['low']) if pd.notna(record['low']) else None,
                    'close': float(record['close']) if pd.notna(record['close']) else None,
                    'volume': int(record['volume']) if pd.notna(record['volume']) else 0  # Default to 0 for null volume
                }

                if validated_record['close'] is not None and validated_record['close'] > 0:
                    validated_data.append(validated_record)
            except (ValueError, TypeError) as e:
                self._log_progress(f"Invalid record: {str(e)}", "warning")
                continue

        return validated_data

    def _load_index_data(self, config: Dict[str, str], start_date: Optional[date] = None, end_date: Optional[date] = None) -> tuple:
        """
        Load data for a specific index using the data provider.
        
        Args:
            config: Configuration dictionary for the index
            start_date: Start date for the data
            end_date: End date for the data
            
        Returns:
            tuple of (validated_records, num_records)
        """
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()

        try:
            # Check for existing data to avoid reloading
            model_class = self.model_map.get(config['table_name'])
            existing_query = self.db.query(model_class.date).filter(
                model_class.symbol == config['db_symbol'],
                model_class.date >= start_date,
                model_class.date <= end_date
            ).all()
            existing_dates = set(d[0] for d in existing_query)
            
            # Skip loading if we already have all the needed dates
            # Except for today's date which might need refreshing
            today = datetime.now().date()
            if existing_dates and (end_date < today or today in existing_dates):
                weekday_dates = set()
                current_date = start_date
                while current_date <= end_date:
                    if current_date.weekday() < 5:  # Only weekdays
                        weekday_dates.add(current_date)
                    current_date += timedelta(days=1)
                
                # Only missing dates (excluding today which might need updating)
                missing_dates = weekday_dates - existing_dates
                if not missing_dates or (len(missing_dates) == 1 and today in missing_dates and today.weekday() == 1):
                    self._log_progress(f"Already have data for {config['db_symbol']} from {start_date} to {end_date}", "debug")
                    return [], 0
            
            # Use the data provider strategy to fetch data
            # We pass the DB symbol to fetch_data, the provider will handle the Yahoo symbol mapping
            self._log_progress(f"Fetching data for {config['db_symbol']} from {start_date} to {end_date}", "debug")
            raw_data = self.data_provider.fetch_data(start_date, end_date, config['db_symbol'])
            
            if not raw_data:
                self._log_progress(f"No data returned from provider for {config['db_symbol']} between {start_date} and {end_date}", "warning")
                # In case of no data, check if this is a special case for specific dates
                if (end_date - start_date).days <= 5:
                    # Try to fill in with the most recent data available
                    self._log_progress(f"Checking if we can fill in missing data for index {config['db_symbol']}", "debug")
                    last_record = self.db.query(model_class).filter(
                        model_class.symbol == config['db_symbol'],
                        model_class.date < start_date
                    ).order_by(model_class.date.desc()).first()
                    
                    if last_record:
                        self._log_progress(f"Found previous data for {config['db_symbol']}, will use for filling", "debug")
                        # Create records for each day in the missing range
                        current_date = start_date
                        filled_records = []
                        while current_date <= end_date:
                            # Skip weekends
                            if current_date.weekday() < 5:  # 0-4 are Monday to Friday
                                # Skip if we already have this date
                                if current_date not in existing_dates:
                                    filled_records.append({
                                        'symbol': config['db_symbol'],
                                        'date': current_date,
                                        'open': last_record.close,
                                        'high': last_record.close,
                                        'low': last_record.close,
                                        'close': last_record.close,
                                        'volume': 0  # Zero volume indicates trading was limited
                                    })
                                    self._log_progress(f"Filled missing data for {config['db_symbol']} on {current_date}", "info")
                            current_date += timedelta(days=1)
                        
                        if filled_records:
                            validated_records = self.validate_data(filled_records)
                            self._save_records(validated_records, config['table_name'])
                            return validated_records, len(validated_records)
                
                return [], 0

            # Log received data
            self._log_progress(f"Received {len(raw_data)} records for {config['db_symbol']}", "debug")
            
            # Check for specific dates in the returned data
            received_dates = {r['date'] for r in raw_data}
            expected_dates = set()
            check_date = start_date
            while check_date <= end_date:
                if check_date.weekday() < 5:  # Skip weekends
                    expected_dates.add(check_date)
                check_date += timedelta(days=1)
                
            missing_dates = expected_dates - received_dates
            if missing_dates:
                self._log_progress(f"Provider missing data for {config['db_symbol']} on dates: {sorted(missing_dates)}", "warning")

            # Make sure we're using the db_symbol
            records = []
            for record in raw_data:
                # Skip dates we already have (except today which might need updating)
                if record['date'] in existing_dates and record['date'] != today:
                    continue
                    
                record['symbol'] = config['db_symbol']  # Ensure correct symbol
                records.append(record)
                
            validated_records = self.validate_data(records)
            if validated_records:
                self._save_records(validated_records, config['table_name'])
            return validated_records, len(validated_records)
        except Exception as e:
            self._handle_error(e, f"loading data for {config['db_symbol']}")
            return [], 0

    def _save_records(self, records: List[Dict[str, Any]], table_name: str):
        """
        Save records to the database.
        
        Args:
            records: List of records to save
            table_name: Name of the table to save to
        """
        if not records:
            return
            
        model_class = self.model_map.get(table_name)
        if not model_class:
            self._log_progress(f"Unknown table name: {table_name}", "error")
            return
            
        try:
            for record in records:
                # Check if this record already exists
                existing = self.db.query(model_class).filter(
                    model_class.symbol == record['symbol'],
                    model_class.date == record['date']
                ).first()
                
                # Only save if it doesn't exist or it has zero volume and we're updating with real data
                if not existing or (existing.volume == 0 and record['volume'] > 0):
                    model_instance = model_class(**record)
                    self.db.merge(model_instance)
                    
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self._handle_error(e, f"saving records to {table_name}")

    def _get_latest_date_for_index(self, table_name: str, symbol: Optional[str] = None) -> Optional[date]:
        try:
            model_class = self.model_map.get(table_name)
            if not model_class:
                raise ValueError(f"Unknown table name: {table_name}")

            query = self.db.query(func.max(model_class.date))
            if symbol:
                query = query.filter(model_class.symbol == symbol)
            
            result = query.scalar()
            return result
        except Exception as e:
            self._handle_error(e, f"getting latest date for {table_name}")

    def _check_for_data_gaps(self, latest_date: date, table_name: str, symbol: str) -> List[tuple]:
        """
        Check for gaps in data from latest_date to today using market calendar.
        
        Args:
            latest_date: The latest date for which we have data
            table_name: The name of the table for this index
            symbol: Symbol to check
            
        Returns:
            List of tuples (start_date, end_date) for each gap that needs to be filled
        """
        today = datetime.now().date()
        
        # If latest_date is already today, we're up to date
        if latest_date >= today:
            self._log_progress(f"Index {symbol} data is already up to date with latest record on {latest_date}", "info")
            return []
            
        # Get existing dates from the database for this index
        model_class = self.model_map.get(table_name)
        if not model_class:
            self._log_progress(f"Unknown table name: {table_name}", "error")
            return []
            
        query = self.db.query(model_class.date).filter(
            model_class.symbol == symbol,
            model_class.date > latest_date,
            model_class.date <= today
        ).distinct()
        
        existing_dates = set(date for (date,) in query)
        
        # Use the market calendar to check for missing dates
        date_ranges = get_market_date_ranges(existing_dates, latest_date, today, "NYSE")
        
        # Log the findings
        for start, end in date_ranges:
            if start == end:
                self._log_progress(f"Will fetch index {symbol} data for market date {start.strftime('%Y-%m-%d')}", "info")
            else:
                self._log_progress(f"Will fetch index {symbol} data for market dates from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}", "info")
        
        return date_ranges 