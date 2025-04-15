from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_
import yfinance as yf
from .base_loader import BaseDataLoader
import pandas as pd
from sqlalchemy import text, func
from logging_config import logger
from database.models_and_schemas.models import SPXPrice, NDXPrice, DJIPrice, RUTPrice, VIXPrice, SOXPrice, OSXPrice
from .data_providers import YahooFinanceProvider
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
                    start_date = latest_date + timedelta(days=1)
                    end_date = today
                    if start_date > end_date:
                        self._log_progress(f"{config['db_symbol']} is up to date")
                        continue
                    validated_data, records = self._load_index_data(config, start_date, end_date)
                
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
                if not all(key in record for key in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']):
                    continue

                validated_record = {
                    'symbol': str(record['symbol']),
                    'date': record['date'],
                    'open': float(record['open']) if pd.notna(record['open']) else None,
                    'high': float(record['high']) if pd.notna(record['high']) else None,
                    'low': float(record['low']) if pd.notna(record['low']) else None,
                    'close': float(record['close']) if pd.notna(record['close']) else None,
                    'volume': int(record['volume']) if pd.notna(record['volume']) else None
                }

                if validated_record['close'] is not None and validated_record['close'] > 0:
                    validated_data.append(validated_record)
            except (ValueError, TypeError) as e:
                self._log_progress(f"Invalid record: {str(e)}", "warning")
                continue

        return validated_data

    def _load_index_data(self, config: Dict[str, str], start_date: Optional[date] = None, end_date: Optional[date] = None) -> tuple:
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()

        try:
            # Use the data provider strategy to fetch data
            yahoo_symbol = config['yahoo_symbol']
            raw_data = self.data_provider.fetch_data(start_date, end_date, yahoo_symbol)
            
            if not raw_data:
                return [], 0

            # Map Yahoo symbol to DB symbol
            records = []
            for record in raw_data:
                record['symbol'] = config['db_symbol']  # Replace Yahoo symbol with DB symbol
                records.append(record)
                
            validated_records = self.validate_data(records)
            self._save_records(validated_records, config['table_name'])
            return validated_records, len(validated_records)
        except Exception as e:
            self._handle_error(e, f"loading data for {config['db_symbol']}")

    def _save_records(self, records: List[Dict[str, Any]], table_name: str):
        try:
            model_class = self.model_map.get(table_name)
            if not model_class:
                raise ValueError(f"Unknown table name: {table_name}")

            for record in records:
                # Create model instance
                model_instance = model_class(**record)
                
                # Use merge to handle upsert
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