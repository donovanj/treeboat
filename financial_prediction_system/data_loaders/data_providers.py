from typing import List, Dict, Any, Optional, Set, Tuple
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
import threading
from financial_prediction_system.config import get_alpaca_settings

load_dotenv()

class YahooFinanceProvider:
    """Data provider for Yahoo Finance"""
    
    # These indices require the ^ prefix for Yahoo Finance
    SPECIAL_INDICES = ['SPX', 'NDX', 'VIX', 'RUT', 'DJI', 'SOX', 'OSX']
    
    @with_retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    @rate_limited(name="yahoo_finance_api", tokens=1, tokens_per_second=2.0, max_tokens=3)
    def _fetch_ticker_history(self, ticker, start_date=None, end_date=None, period=None, interval="1d"):
        """
        Rate-limited method to fetch ticker history from Yahoo Finance
        
        Args:
            ticker: Yahoo Finance ticker object
            start_date: Start date for data fetching
            end_date: End date for data fetching
            period: Period string (e.g., '1d', '5d', '1mo') - used if start_date/end_date not provided
            interval: Data interval (e.g., '1d', '1m')
        
        Returns:
            DataFrame with historical data
        """
        if start_date and end_date:
            return ticker.history(start=start_date, end=end_date, interval=interval, prepost=False, actions=False)
        else:
            return ticker.history(period=period, interval=interval, prepost=False, actions=False)
    
    def fetch_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch data from Yahoo Finance"""
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()
        if not symbol:
            return []
            
        try:
            # Handle index symbols that need the ^ prefix for Yahoo Finance
            if symbol in self.SPECIAL_INDICES:
                ticker_symbol = f"^{symbol}"
                logger.debug(f"Using Yahoo Finance symbol {ticker_symbol} for {symbol}")
            else:
                ticker_symbol = symbol
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Check if we're fetching today's data
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # If we're fetching recent data including today
            if end_date >= today and (today - start_date).days <= 5:
                logger.debug(f"Fetching recent data for {symbol} using explicit period")
                
                # Get historical data for past days
                historical_end = min(yesterday, end_date)
                historical_df = None
                
                if start_date <= historical_end:
                    logger.debug(f"Fetching historical data for {symbol} from {start_date} to {historical_end}")
                    historical_df = self._fetch_ticker_history(
                        ticker, 
                        start_date=start_date,
                        end_date=historical_end + timedelta(days=1),  # Add a day to include the end date
                        interval="1d"
                    )
                
                # Get today's data if needed
                if end_date >= today:
                    logger.debug(f"Fetching today's data for {symbol}")
                    # Get today's data with period='1d' which is more reliable for the current day
                    today_df = self._fetch_ticker_history(ticker, period='1d', interval='1d')
                    
                    # Make sure we got data and handle timezone conversion
                    if not today_df.empty:
                        logger.debug(f"Received today's data for {symbol}: {len(today_df)} rows")
                        # Combine historical and today's data
                        if historical_df is not None and not historical_df.empty:
                            df = pd.concat([historical_df, today_df], axis=0)
                            # Remove duplicates, keeping the latest version (today's data)
                            df = df.loc[~df.index.duplicated(keep='last')]
                        else:
                            df = today_df
                    else:
                        logger.warning(f"No today's data returned for {symbol}")
                        df = historical_df if historical_df is not None else pd.DataFrame()
                else:
                    df = historical_df if historical_df is not None else pd.DataFrame()
            else:
                # For purely historical data, use the regular approach
                logger.debug(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
                df = self._fetch_ticker_history(ticker, start_date=start_date, end_date=end_date + timedelta(days=1), interval="1d")
            
            if df.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return []
            
            logger.debug(f"Processing Yahoo Finance data for {symbol}, got {len(df)} rows with columns: {df.columns.tolist()}")   
            records = []
            for index, row in df.iterrows():
                try:
                    # Convert the DatetimeIndex to a date object
                    record_date = index.date() if hasattr(index, 'date') else index
                    
                    record = {
                        'symbol': symbol,  # Use the original symbol, not the Yahoo ticker symbol
                        'date': record_date,
                        'open': float(row['Open']) if not pd.isna(row['Open']) else None,
                        'high': float(row['High']) if not pd.isna(row['High']) else None,
                        'low': float(row['Low']) if not pd.isna(row['Low']) else None,
                        'close': float(row['Close']) if not pd.isna(row['Close']) else None,
                        'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                    }
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Error processing row for {symbol} on {index}: {type(e).__name__}")
                    continue
                    
            logger.debug(f"Processed {len(records)} records for {symbol}")
            return records
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {type(e).__name__}")
            return []

class AlpacaProvider:
    """Data provider for Alpaca"""
    
    def __init__(self):
        # Get settings from Pydantic model to avoid exposing in logs
        try:
            alpaca_settings = get_alpaca_settings()
            api_key = alpaca_settings.ALPACA_KEY
            secret_key = alpaca_settings.ALPACA_SECRET
            
            # Add debug logging to see what credentials we're getting
            logger.warning(f"Alpaca API key exists: {api_key is not None}, Secret exists: {secret_key is not None}")
            logger.warning(f"Alpaca API key length: {len(api_key) if api_key else 0}, Secret length: {len(secret_key) if secret_key else 0}")
            
            # Also check environment directly
            env_key = os.environ.get('ALPACA_KEY')
            env_secret = os.environ.get('ALPACA_SECRET')
            logger.warning(f"Environment ALPACA_KEY exists: {env_key is not None}, ALPACA_SECRET exists: {env_secret is not None}")
            
            if not api_key or not secret_key:
                logger.warning("Alpaca API credentials not found in settings")
                raise ValueError("Missing Alpaca API credentials")
                
            self.client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key
            )
            self.api_key = api_key
            self.secret_key = secret_key
            self.max_days_per_request = 1000  # Alpaca's limit
            self.base_url = "https://data.alpaca.markets/v2"
            
            # Dynamic rate limiting state
            self.rate_limit = 200  # Default limit
            self.rate_remaining = 200  # Default remaining
            self.rate_reset_time = datetime.now().timestamp() + 60  # Default reset time (1 minute from now)
            self.rate_lock = threading.Lock()  # For thread safety
            
            logger.info("Alpaca provider initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Alpaca provider: {type(e).__name__}")
            # Re-raise but with a cleaner message that doesn't expose credentials
            raise RuntimeError(f"Failed to initialize Alpaca provider: {type(e).__name__}")
        
    def _handle_rate_limiting(self, response: requests.Response) -> None:
        """
        Extract rate limit headers from response and adjust rate limiting accordingly.
        
        Args:
            response: HTTP response from Alpaca API
        """
        try:
            with self.rate_lock:
                # Extract rate limit headers
                if 'x-ratelimit-limit' in response.headers:
                    self.rate_limit = int(response.headers['x-ratelimit-limit'])
                    
                if 'x-ratelimit-remaining' in response.headers:
                    self.rate_remaining = int(response.headers['x-ratelimit-remaining'])
                    
                if 'x-ratelimit-reset' in response.headers:
                    self.rate_reset_time = int(response.headers['x-ratelimit-reset'])
                
                # If we're running low on remaining requests, add a delay
                if self.rate_remaining < 20:
                    # Calculate time until reset
                    now = datetime.now().timestamp()
                    time_until_reset = max(0, self.rate_reset_time - now)
                    
                    # If reset is soon, wait a bit
                    if time_until_reset < 10:
                        logger.warning(f"Rate limit almost reached ({self.rate_remaining}/{self.rate_limit}), waiting for reset in {time_until_reset:.1f}s")
                        time.sleep(time_until_reset + 1)  # Add 1 second buffer
                    else:
                        # Otherwise, slow down proportionally to how close we are to the limit
                        delay = 1.0 - (self.rate_remaining / self.rate_limit)
                        logger.info(f"Rate limit getting low ({self.rate_remaining}/{self.rate_limit}), adding delay of {delay:.2f}s")
                        time.sleep(delay)
                
                logger.debug(f"Alpaca API rate limit: {self.rate_remaining}/{self.rate_limit}, reset at {datetime.fromtimestamp(self.rate_reset_time).strftime('%H:%M:%S')}")
        except Exception as e:
            logger.warning(f"Error handling rate limit headers: {type(e).__name__}")
        
    @with_retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _get_stock_bars(self, request):
        """
        Get stock bars from Alpaca using SDK.
        We still use retry but rate limiting is now handled dynamically.
        """
        return self.client.get_stock_bars(request)
    
    @with_retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _get_latest_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get latest data with 15-minute delay for a symbol using direct REST API.
        This is useful for getting today's data when the market is open.
        
        Args:
            symbol: Stock symbol to get data for
            
        Returns:
            Dictionary containing the latest bar data
        """
        url = f"{self.base_url}/stocks/bars/latest?symbols={symbol}&feed=delayed_sip"
        
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        
        try:
            response = requests.get(url, headers=headers)
            # Process rate limit headers
            self._handle_rate_limiting(response)
            
            response.raise_for_status()
            data = response.json()
            
            if "bars" in data and symbol in data["bars"]:
                return data["bars"][symbol]
            else:
                logger.warning(f"No latest data found for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Error fetching latest data for {symbol}: {type(e).__name__}")
            return None
    
    @with_retry(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def _get_daily_bars_directly(self, symbol: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """
        Get daily bars directly using REST API instead of the Alpaca SDK.
        This is useful for getting same-day data with 15-minute delay.
        
        Args:
            symbol: Stock symbol to get data for
            start_date: Start date
            end_date: End date
            
        Returns:
            List of bar data
        """
        # Format dates as ISO strings
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()
        
        url = f"{self.base_url}/stocks/bars?symbols={symbol}&timeframe=1D&start={start_str}&end={end_str}&limit=1000&adjustment=raw&feed=iex&sort=asc"
        
        headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
        
        try:
            response = requests.get(url, headers=headers)
            # Process rate limit headers
            self._handle_rate_limiting(response)
            
            response.raise_for_status()
            data = response.json()
            
            if "bars" in data and symbol in data["bars"]:
                return data["bars"][symbol]
            else:
                logger.warning(f"No daily bars found for {symbol} between {start_date} and {end_date}")
                return []
        except Exception as e:
            logger.error(f"Error fetching daily bars for {symbol}: {type(e).__name__}")
            return []
        
    def fetch_data(self, start_date: Optional[date] = None, end_date: Optional[date] = None, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch data from Alpaca"""
        if not start_date:
            start_date = date(2000, 1, 1)
        if not end_date:
            end_date = datetime.now().date()
        if not symbol:
            return []
            
        # Check if we're looking for recent data (including today)
        today = datetime.now().date()
        looking_for_today = end_date >= today
        recent_data_window = 5  # Days
        
        all_records = []
        
        # If we're looking for today's data or very recent data
        if looking_for_today and (today - start_date).days <= recent_data_window:
            logger.debug(f"Fetching recent data for {symbol} including today using direct API")
            
            # First try to get historical data for the complete range
            bars = self._get_daily_bars_directly(symbol, start_date, end_date)
            
            # Process the bars data
            for bar in bars:
                # Converting ISO timestamp to date
                timestamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                record = {
                    'symbol': symbol,
                    'date': timestamp.date(),
                    'open': bar["o"],
                    'high': bar["h"],
                    'low': bar["l"],
                    'close': bar["c"],
                    'volume': bar["v"]
                }
                all_records.append(record)
            
            # If we need today's data and it's not in our results yet, get the latest bar
            today_data = [r for r in all_records if r['date'] == today]
            if looking_for_today and not today_data:
                latest_bar = self._get_latest_data(symbol)
                if latest_bar:
                    timestamp = datetime.fromisoformat(latest_bar["t"].replace("Z", "+00:00"))
                    # Only add if it's actually from today
                    if timestamp.date() == today:
                        record = {
                            'symbol': symbol,
                            'date': timestamp.date(),
                            'open': latest_bar["o"],
                            'high': latest_bar["h"],
                            'low': latest_bar["l"],
                            'close': latest_bar["c"],
                            'volume': latest_bar["v"]
                        }
                        all_records.append(record)
                        logger.debug(f"Added latest data for {symbol} for today")
            
            return all_records
            
        # For historical data, use the existing method with SDK
        total_days = (end_date - start_date).days + 1
        
        for chunk_start in range(0, total_days, self.max_days_per_request):
            chunk_start_date = start_date + timedelta(days=chunk_start)
            chunk_end_date = min(
                chunk_start_date + timedelta(days=self.max_days_per_request - 1),
                end_date
            )

            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Day,
                    start=chunk_start_date,
                    end=chunk_end_date
                )
                
                # Use the retry-enabled method
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
                logger.error(f"Error loading chunk for {symbol}: {type(e).__name__}")
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