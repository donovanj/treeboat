from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from financial_prediction_system.infrastructure.database.models_and_schemas.models import Stock, StockPrice
from .base_loader import BaseDataLoader
from .data_providers import AlpacaProvider
from .cache_decorator import cacheable, invalidate_cache
from .cache import RedisCache
import pandas as pd

class StockDataLoader(BaseDataLoader):
    def __init__(self, db: Session, cache: Optional[RedisCache] = None):
        # Use AlpacaProvider as default strategy
        data_provider = AlpacaProvider()
        super().__init__(db, data_provider)
        self.cache = cache

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

        start_date = latest_date + timedelta(days=1)
        end_date = datetime.now().date()

        if start_date > end_date:
            self._log_progress("Data is up to date")
            return 0
            
        records = self.load_historical_data(start_date, end_date, symbol=symbol)
        
        # Invalidate historical cache if we updated data
        if self.cache and records > 0:
            invalidate_cache("stock_historical", symbol)(self.cache)
            
        return records

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
                if not all(key in record for key in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']):
                    continue
                
                # Type conversion and cleaning
                validated_record = {
                    'symbol': str(record['symbol']),
                    'date': record['date'],
                    'open': float(record['open']) if record['open'] else None,
                    'high': float(record['high']) if record['high'] else None,
                    'low': float(record['low']) if record['low'] else None,
                    'close': float(record['close']) if record['close'] else None,
                    'volume': int(record['volume']) if record['volume'] else None
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
            # Use the data provider to fetch data
            records = self.data_provider.fetch_data(start_date, end_date, symbol)
            
            if not records:
                return [], 0
                
            validated_records = self.validate_data(records)
            self._save_records(validated_records)
            return validated_records, len(validated_records)
        except Exception as e:
            self._handle_error(e, f"loading data for {symbol}")

    def _save_records(self, records: List[Dict[str, Any]]):
        try:
            for record in records:
                price = StockPrice(**record)
                self.db.merge(price)  # Use merge for upsert functionality
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self._handle_error(e, "saving records") 