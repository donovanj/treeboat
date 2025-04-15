from typing import List, Dict, Any, Optional
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

load_dotenv()

class YahooFinanceProvider:
    """Data provider for Yahoo Finance"""
    
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
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
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
                
                bars = self.client.get_stock_bars(request)
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