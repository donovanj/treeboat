from typing import List, Dict, Any, Optional, Set
from datetime import date, datetime, timedelta
import yfinance as yf
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
import time
from financial_prediction_system.logging_config import logger
import QuantLib as ql
from .rate_limiter import rate_limited, with_retry

load_dotenv()

class YahooFinanceProvider:
    """Data provider for Yahoo Finance"""
    
    @with_retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    @rate_limited(name="yahoo_finance_api", tokens=1, tokens_per_second=2.0, max_tokens=3)
    def _fetch_ticker_history(self, ticker, start_date, end_date):
        """Rate-limited method to fetch ticker history from Yahoo Finance"""
        return ticker.history(start=start_date, end=end_date, interval="1d")
    
    def fetch_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch data from Yahoo Finance"""
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()
        if not symbol:
            return []
            
        try:
            ticker = yf.Ticker(symbol)
            df = self._fetch_ticker_history(ticker, start_date, end_date)
            
            if df.empty:
                return []
                
            records = []
            for index, row in df.iterrows():
                record = {
                    'symbol': symbol,
                    'date': index.date(),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume']
                }
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
            return []

class AlpacaProvider:
    """Data provider for Alpaca"""
    
    def __init__(self):
        self.client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_KEY"),
            secret_key=os.getenv("ALPACA_SECRET")
        )
        self.max_days_per_request = 1000  # Alpaca's limit
        
    @with_retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    @rate_limited(name="alpaca_api", tokens=1, tokens_per_second=5.0, max_tokens=5)
    def _get_stock_bars(self, request):
        """
        Rate-limited method to get stock bars from Alpaca.
        Alpaca has a rate limit of 200 requests per minute (about 3.33 req/sec).
        We use a more conservative 5 req/sec with max burst of 5 to stay comfortably under the limit.
        """
        return self.client.get_stock_bars(request)
        
    def fetch_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch data from Alpaca"""
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()
        if not symbol:
            return []
            
        total_days = (end_date - start_date).days
        all_records = []

        for chunk_start in range(0, total_days, self.max_days_per_request):
            chunk_start_date = start_date + timedelta(days=chunk_start)
            chunk_end_date = min(
                chunk_start_date + timedelta(days=self.max_days_per_request),
                end_date
            )

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=chunk_start_date,
                    end=chunk_end_date
                )
                
                # Use the rate-limited method
                bars = self._get_stock_bars(request)
                
                if bars and hasattr(bars, 'data') and symbol in bars.data:
                    for bar in bars.data[symbol]:
                        record = {
                            'symbol': bar.symbol,
                            'date': bar.timestamp.date(),
                            'open': bar.open,
                            'high': bar.high,
                            'low': bar.low,
                            'close': bar.close,
                            'volume': bar.volume
                        }
                        all_records.append(record)
            except Exception as e:
                logger.error(f"Error loading chunk for {symbol}: {str(e)}")
                continue

        return all_records

class TreasuryProvider:
    """Data provider for Treasury yields"""
    
    def __init__(self):
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
        
    def _parse_yield(self, value: str) -> Optional[float]:
        """Parse yield value, handling N/A and other special cases"""
        try:
            if value == 'N/A' or not value.strip():
                return None
            return float(value)
        except ValueError:
            return None
    
    def fetch_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch Treasury yield data"""
        if not start_date:
            start_date = date(2015, 1, 1)  # Treasury data typically starts from 2015
        if not end_date:
            end_date = datetime.now().date()
            
        all_records = []
        current_year = start_date.year
        
        while current_year <= end_date.year:
            try:
                url = f"{self.base_url}?type=daily_treasury_yield_curve&field_tdr_date_value={current_year}"
                response = self._make_request(url)
                
                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'class': 'usa-table'})
                
                if not table:
                    logger.warning(f"No data table found for year {current_year}")
                    current_year += 1
                    continue
                
                for row in table.find_all('tr')[1:]:  # Skip header row
                    cells = row.find_all('td')
                    if len(cells) < 11:  # Ensure we have all required columns
                        continue
                    
                    try:
                        date_str = cells[0].find('time').get('datetime')
                        record_date = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ').date()
                        
                        # Skip if date is outside requested range
                        if record_date < start_date or record_date > end_date:
                            continue
                        
                        # Extract yields, handling N/A values
                        record = {
                            'date': record_date,
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
                        
                        all_records.append(record)
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Error parsing row: {str(e)}")
                        continue
                    
            except Exception as e:
                logger.error(f"Error loading data for year {current_year}: {str(e)}")
            
            current_year += 1
            
        return all_records 

def get_market_calendar_dates(start_date: date, end_date: date, calendar_name: str = "NYSE") -> List[date]:
    """
    Get a list of dates when the market was open between start_date and end_date.
    
    Args:
        start_date: Start date
        end_date: End date
        calendar_name: Market calendar to use (default: NYSE)
        
    Returns:
        List of dates when market was open
    """
    # Map calendar names to QuantLib calendar classes
    calendar_map = {
        "NYSE": ql.UnitedStates(ql.UnitedStates.NYSE)
    }
    
    if calendar_name not in calendar_map:
        raise ValueError(f"Unknown calendar: {calendar_name}")
    
    calendar = calendar_map[calendar_name]
    
    # Convert Python dates to QuantLib dates
    ql_start_date = ql.Date(start_date.day, start_date.month, start_date.year)
    ql_end_date = ql.Date(end_date.day, end_date.month, end_date.year)
    
    # Generate list of business days
    market_dates = []
    current_date = ql_start_date
    
    while current_date <= ql_end_date:
        if calendar.isBusinessDay(current_date):
            # Convert back to Python date
            py_date = date(current_date.year(), current_date.month(), current_date.dayOfMonth())
            market_dates.append(py_date)
        current_date = current_date + 1
    
    return market_dates

def check_missing_market_dates(db_dates: Set[date], start_date: date, end_date: date, calendar_name: str = "NYSE") -> List[date]:
    """
    Check for missing market dates in the database.
    
    Args:
        db_dates: Set of dates present in the database
        start_date: Start date to check from
        end_date: End date to check to
        calendar_name: Market calendar to use
        
    Returns:
        List of dates that are missing from the database
    """
    market_dates = set(get_market_calendar_dates(start_date, end_date, calendar_name))
    return sorted(market_dates - db_dates)

def get_market_date_ranges(db_dates: Set[date], start_date: date, end_date: date, calendar_name: str = "NYSE") -> List[tuple]:
    """
    Get ranges of missing market dates.
    
    Args:
        db_dates: Set of dates present in the database
        start_date: Start date to check from
        end_date: End date to check to
        calendar_name: Market calendar to use
        
    Returns:
        List of (start_date, end_date) tuples for missing date ranges
    """
    missing_dates = check_missing_market_dates(db_dates, start_date, end_date, calendar_name)
    
    if not missing_dates:
        return []
        
    date_ranges = []
    start_range = missing_dates[0]
    prev_date = start_range
    
    for i in range(1, len(missing_dates)):
        curr_date = missing_dates[i]
        # If dates are not consecutive business days
        if (curr_date - prev_date).days > 3:  # Allow for weekends
            date_ranges.append((start_range, prev_date))
            start_range = curr_date
        prev_date = curr_date
        
    # Add the last range
    date_ranges.append((start_range, missing_dates[-1]))
    
    return date_ranges 