from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
import requests
from bs4 import BeautifulSoup
from financial_prediction_system.infrastructure.database.models_and_schemas.models import TreasuryYield
from .base_loader import BaseDataLoader
import os
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
import time
from .data_providers import TreasuryProvider
from .cache_decorator import cacheable, invalidate_cache
from .cache import RedisCache

load_dotenv()

class TreasuryDataLoader(BaseDataLoader):
    def __init__(self, db: Session, cache: Optional[RedisCache] = None):
        # Use TreasuryProvider as default strategy
        data_provider = TreasuryProvider()
        super().__init__(db, data_provider)
        self.cache = cache
        self.base_url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.rate_limit = 1  # requests per second
        self.last_request_time = 0

    @sleep_and_retry
    @limits(calls=1, period=1)  # 1 request per second
    def _make_request(self, url: str) -> requests.Response:
        """Make a rate-limited request to the Treasury website"""
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response

    @cacheable("treasury_historical")
    def load_historical_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> int:
        if not start_date:
            start_date = date(2015, 1, 1)  # Treasury data typically starts from 2015
        if not end_date:
            end_date = datetime.now().date()

        try:
            # Use the data provider strategy to fetch data
            raw_data = self.data_provider.fetch_data(start_date, end_date)
            
            if not raw_data:
                return 0
                
            validated_records = self.validate_data(raw_data)
            self._save_records(validated_records)
            
            # Notify observers about data quality
            if validated_records and self.quality_observers:
                quality_results = self.notify_data_loaded(validated_records, start_date, end_date, symbol)
                self._log_debug(f"Data quality results: {quality_results}")
                
            return len(validated_records)
        except Exception as e:
            self._handle_error(e, f"loading historical treasury data")
            return 0

    @cacheable("treasury_daily")
    def update_daily_data(self, symbol: Optional[str] = None) -> int:
        latest_date = self.get_latest_date()
        if not latest_date:
            return self.load_historical_data()

        start_date = latest_date + timedelta(days=1)
        end_date = datetime.now().date()

        if start_date > end_date:
            self._log_progress("Data is up to date")
            return 0

        records = self.load_historical_data(start_date, end_date)
        
        # Invalidate historical cache if we updated data
        if self.cache and records > 0:
            invalidate_cache("treasury_historical")(self.cache)
            
        return records

    def get_latest_date(self, symbol: Optional[str] = None) -> Optional[date]:
        try:
            result = self.db.query(TreasuryYield.date).order_by(TreasuryYield.date.desc()).first()
            return result[0] if result else None
        except Exception as e:
            self._handle_error(e, "getting latest date")

    def _load_year_data(self, year: int) -> int:
        try:
            url = f"{self.base_url}?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
            response = self._make_request(url)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'usa-table'})
            
            if not table:
                raise ValueError(f"No data table found for year {year}")
            
            records = []
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if len(cells) < 11:  # Ensure we have all required columns
                    continue
                
                try:
                    date_str = cells[0].find('time').get('datetime')
                    record_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').date()
                    
                    # Extract yields, handling N/A values
                    yields = {
                        'mo1': self._parse_yield(cells[12].text.strip()),  # 1 Mo
                        'mo2': self._parse_yield(cells[14].text.strip()),  # 2 Mo
                        'mo3': self._parse_yield(cells[15].text.strip()),  # 3 Mo
                        'mo6': self._parse_yield(cells[16].text.strip()),  # 6 Mo
                        'yr1': self._parse_yield(cells[17].text.strip()),  # 1 Yr
                        'yr2': self._parse_yield(cells[18].text.strip()),  # 2 Yr
                        'yr5': self._parse_yield(cells[20].text.strip()),  # 5 Yr
                        'yr10': self._parse_yield(cells[22].text.strip()),  # 10 Yr
                        'yr30': self._parse_yield(cells[24].text.strip())   # 30 Yr
                    }
                    
                    record = {'date': record_date, **yields}
                    records.append(record)
                except (ValueError, AttributeError) as e:
                    self._log_progress(f"Error parsing row: {str(e)}", "warning")
                    continue
            
            # Validate and save records
            validated_records = self.validate_data(records)
            self._save_records(validated_records)
            
            return len(validated_records)
        except Exception as e:
            self._handle_error(e, f"loading data for year {year}")

    def _parse_yield(self, value: str) -> Optional[float]:
        """Parse yield value, handling N/A and other special cases"""
        try:
            if value == 'N/A' or not value.strip():
                return None
            return float(value)
        except ValueError:
            return None

    def validate_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validated_data = []
        for record in data:
            try:
                if not all(key in record for key in ['date', 'mo1', 'mo2', 'mo3', 'mo6', 'yr1', 'yr2', 'yr5', 'yr10', 'yr30']):
                    continue

                validated_record = {
                    'date': record['date'],
                    'mo1': record['mo1'],
                    'mo2': record['mo2'],
                    'mo3': record['mo3'],
                    'mo6': record['mo6'],
                    'yr1': record['yr1'],
                    'yr2': record['yr2'],
                    'yr5': record['yr5'],
                    'yr10': record['yr10'],
                    'yr30': record['yr30']
                }

                # At least one yield should be present
                if any(value is not None for key, value in validated_record.items() if key != 'date'):
                    validated_data.append(validated_record)
            except (ValueError, TypeError) as e:
                self._log_progress(f"Invalid record: {str(e)}", "warning")
                continue

        return validated_data

    def _save_records(self, records: List[Dict[str, Any]]):
        try:
            for record in records:
                yield_data = TreasuryYield(**record)
                self.db.merge(yield_data)  # Use merge for upsert functionality
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            self._handle_error(e, "saving records") 