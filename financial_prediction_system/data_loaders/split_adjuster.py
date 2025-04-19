import yfinance as yf
import psycopg2
import time
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_splits_adjust.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Database connection parameters - adjust these to your configuration
DB_PARAMS = {
    "dbname": "market_database",
    "user": "dj",
    "password": "ZgNNQd1YimwFE4Kd",
    "host": "localhost",
    "port": "5432"
}

# Rate limiting parameters
REQUEST_DELAY_MIN = 1.0  # Minimum delay between requests in seconds
REQUEST_DELAY_MAX = 2.5  # Maximum delay between requests in seconds
BATCH_SIZE = 50  # Number of stocks to process before taking a longer pause

def get_connection():
    """Create and return a database connection"""
    return psycopg2.connect(**DB_PARAMS)

def get_active_symbols():
    """Fetch all active stock symbols from the database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT symbol FROM stocks WHERE is_active = TRUE")
        symbols = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"Fetched {len(symbols)} active symbols from database")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise
    finally:
        if conn:
            cursor.close()
            conn.close()

def get_historical_splits(symbol, start_date):
    """Get historical splits for a symbol using ticker.history"""
    try:
        # Format start date to string if it's a date object
        if isinstance(start_date, (datetime, date)):
            start_date_str = start_date.strftime('%Y-%m-%d')
        else:
            start_date_str = start_date
        
        # Apply rate limiting
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
        
        # Get splits using ticker.history() method
        ticker = yf.Ticker(symbol)
        history = ticker.history(start=start_date_str, actions=True)
        
        if 'Stock Splits' not in history.columns:
            return []
        
        # Filter for non-zero splits
        splits = history[history['Stock Splits'] > 0]['Stock Splits']
        
        if splits.empty:
            return []
        
        # Convert to list of tuples
        result = []
        for split_date, ratio in splits.items():
            date_obj = split_date.date() if hasattr(split_date, 'date') else split_date
            if ratio > 0:  # Additional check to ensure ratio is positive
                result.append((date_obj, float(ratio)))
        
        if result:
            logger.info(f"Found {len(result)} splits for {symbol}")
            for date_obj, ratio in result:
                logger.info(f"  {symbol}: {date_obj} - {ratio}:1 split")
        
        return result
    
    except Exception as e:
        logger.error(f"Error getting historical splits for {symbol}: {e}")
        return []

def adjust_prices_for_split(symbol, split_date, split_ratio):
    """Adjust historical prices for a stock split"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Find all price records before the split date
        cursor.execute("""
            SELECT COUNT(*) 
            FROM stock_prices 
            WHERE symbol = %s AND date < %s
        """, (symbol, split_date))
        
        count = cursor.fetchone()[0]
        if count == 0:
            logger.info(f"No historical price records to adjust for {symbol} before {split_date}")
            return 0
        
        # Update all historical prices before the split date
        # - Divide prices by split ratio (to match post-split scale)
        # - Multiply volume by split ratio
        cursor.execute("""
            UPDATE stock_prices 
            SET 
                open = open / %s,
                high = high / %s,
                low = low / %s,
                close = close / %s,
                volume = volume * %s
            WHERE symbol = %s AND date < %s
        """, (split_ratio, split_ratio, split_ratio, split_ratio, split_ratio, symbol, split_date))
        
        updated_rows = cursor.rowcount
        conn.commit()
        
        if updated_rows > 0:
            logger.info(f"Adjusted {updated_rows} historical price records for {symbol} {split_ratio}:1 split on {split_date}")
        
        return updated_rows
    
    except Exception as e:
        logger.error(f"Error adjusting prices for {symbol}: {e}")
        if conn:
            conn.rollback()
        return 0
    
    finally:
        if conn:
            cursor.close()
            conn.close()

def manually_adjust_symbol(symbol, split_date, split_ratio):
    """Manually adjust a specific symbol based on provided split info"""
    logger.info(f"Manually adjusting {symbol} for {split_ratio}:1 split on {split_date}")
    
    # Parse date string if provided
    if isinstance(split_date, str):
        split_date = datetime.strptime(split_date, '%Y-%m-%d').date()
    
    # Perform the adjustment
    adjusted = adjust_prices_for_split(symbol, split_date, split_ratio)
    
    logger.info(f"Manual adjustment complete. Adjusted {adjusted} records.")
    return adjusted

def main(start_date=None):
    """Main function to fetch splits and adjust historical prices"""
    logger.info("Starting historical price adjustment for stock splits")
    
    try:
        # Set default start date if not provided
        if start_date is None:
            start_date = datetime(2020, 1, 1).date()
            logger.info(f"Using default start date: {start_date}")
        
        # Get all active symbols
        symbols = get_active_symbols()
        total_symbols = len(symbols)
        logger.info(f"Processing {total_symbols} symbols")
        
        # Track statistics
        total_splits = 0
        total_adjusted = 0
        processed_symbols = 0
        
        # Process each symbol
        for i, symbol in enumerate(tqdm(symbols, desc="Processing symbols")):
            processed_symbols += 1
            
            # Rate limiting for batches
            if i > 0 and i % BATCH_SIZE == 0:
                time.sleep(random.uniform(3, 5))  # Longer pause between batches
                logger.info(f"Processed {i}/{total_symbols} symbols...")
            
            try:
                # Get historical splits for this symbol
                splits = get_historical_splits(symbol, start_date)
                total_splits += len(splits)
                
                # Process each split
                for split_date, split_ratio in splits:
                    # Adjust historical prices for this split
                    adjusted = adjust_prices_for_split(symbol, split_date, split_ratio)
                    total_adjusted += adjusted
                    
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                continue
        
        # Log summary
        logger.info(f"Historical price adjustment complete:")
        logger.info(f"  Processed {processed_symbols} symbols")
        logger.info(f"  Found {total_splits} splits")
        logger.info(f"  Adjusted {total_adjusted} historical price records")
        
        return total_adjusted
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

def check_prices(symbol, around_date=None, days_before=7, days_after=7):
    """Check prices for a symbol around a specific date"""
    try:
        if around_date is None:
            # Default to current date if not specified
            around_date = datetime.now().date()
        elif isinstance(around_date, str):
            # Parse date string if provided
            around_date = datetime.strptime(around_date, '%Y-%m-%d').date()
        
        # Calculate date range
        start_date = around_date - timedelta(days=days_before)
        end_date = around_date + timedelta(days=days_after)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol = %s AND date BETWEEN %s AND %s
            ORDER BY date
        """, (symbol, start_date, end_date))
        
        rows = cursor.fetchall()
        
        if not rows:
            print(f"No price data found for {symbol} between {start_date} and {end_date}")
            return
        
        print(f"\nPrice data for {symbol} from {start_date} to {end_date}:")
        print(f"{'Date':<12} {'Open':<10} {'High':<10} {'Low':<10} {'Close':<10} {'Volume':<15}")
        print("-" * 72)
        
        for row in rows:
            date_val, open_p, high, low, close, volume = row
            date_str = date_val.strftime('%Y-%m-%d')
            # Highlight the around_date row
            if date_val == around_date:
                print(f"{date_str}* {float(open_p):<10.2f} {float(high):<10.2f} {float(low):<10.2f} {float(close):<10.2f} {volume:<15}")
            else:
                print(f"{date_str}  {float(open_p):<10.2f} {float(high):<10.2f} {float(low):<10.2f} {float(close):<10.2f} {volume:<15}")
    
    except Exception as e:
        logger.error(f"Error checking prices: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Adjust historical stock prices for splits')
    parser.add_argument('--start-date', type=str, help='Start date for checking splits (YYYY-MM-DD)')
    parser.add_argument('--check', type=str, help='Check prices for a specific symbol')
    parser.add_argument('--around-date', type=str, help='Date to check prices around (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, help='Symbol for manual adjustment')
    parser.add_argument('--split-date', type=str, help='Split date for manual adjustment (YYYY-MM-DD)')
    parser.add_argument('--ratio', type=float, help='Split ratio for manual adjustment')
    
    args = parser.parse_args()
    
    # Handle checking prices for a symbol
    if args.check:
        check_prices(args.check, args.around_date)
    
    # Handle manual adjustment
    elif args.symbol and args.split_date and args.ratio:
        manually_adjust_symbol(args.symbol, args.split_date, args.ratio)
    
    # Run the main adjustment process
    else:
        start_date = None
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        
        main(start_date=start_date)
