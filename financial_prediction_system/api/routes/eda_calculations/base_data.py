# Placeholder for base data fetching and preprocessing 

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta

def fetch_stock_data(db: Session, symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetches stock price data for a given symbol and date range."""
    sql_query = text("""
        SELECT date, open, high, low, close, volume 
        FROM stock_prices 
        WHERE symbol = :symbol AND date >= :start AND date <= :end
        ORDER BY date ASC
    """)
    df = pd.read_sql(
        sql_query, 
        db.connection(),
        params={'symbol': symbol, 'start': start, 'end': end}, 
        parse_dates=['date']
    )
    if df.empty:
        return df # Return empty df, error handled in main endpoint

    # Ensure correct data types and drop NAs from failed conversion
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True) 
    
    # --- Create Year-Month column for chronological grouping --- 
    # Keep date as column initially for easier passing, set index later if needed by functions
    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    
    return df

def fetch_other_market_data(db: Session, start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetches data for indices (SPX, NDX, etc.) and Treasury yields."""
    
    def _fetch_single_table(table_name, date_col='date', close_col='close', index_symbol=None, rename_col=None):
        query = f"SELECT {date_col}, {close_col} FROM {table_name} WHERE {date_col} >= :start AND {date_col} <= :end"
        params = {'start': start, 'end': end}
        if index_symbol:
            query += f" AND symbol = :symbol"
            params['symbol'] = index_symbol
        query += f" ORDER BY {date_col} ASC"
        
        other_df = pd.read_sql(
            text(query), db.connection(), params=params, parse_dates=[date_col]
        )
        rename_to = rename_col or table_name # Use provided rename_col or default to table_name
        other_df.rename(columns={close_col: rename_to, date_col: 'date'}, inplace=True)
        other_df.set_index('date', inplace=True)
        # Ensure the column is numeric
        other_df[rename_to] = pd.to_numeric(other_df[rename_to], errors='coerce') 
        return other_df

    data = {}
    data['spx_prices'] = _fetch_single_table('spx_prices', index_symbol='SPX')
    data['ndx_prices'] = _fetch_single_table('ndx_prices', index_symbol='NDX')
    data['vix_prices'] = _fetch_single_table('vix_prices', index_symbol='VIX')
    data['dji_prices'] = _fetch_single_table('dji_prices', index_symbol='DJI')
    data['rut_prices'] = _fetch_single_table('rut_prices', index_symbol='RUT')
    data['osx_prices'] = _fetch_single_table('osx_prices', index_symbol='OSX')
    data['sox_prices'] = _fetch_single_table('sox_prices', index_symbol='SOX')
    data['US10Y'] = _fetch_single_table('treasury_yields', close_col='yr10', rename_col='US10Y')
    
    return data

def combine_data(stock_df: pd.DataFrame, other_data: dict[str, pd.DataFrame], symbol: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Combines stock data with other market data and calculates log returns."""
    if stock_df.empty:
        return None, None

    stock_df_indexed = stock_df.set_index('date') # Set index for joining
    
    # Extract DataFrames from the dictionary
    other_dfs = list(other_data.values())
    
    # Join stock data with other market data
    combined_df = stock_df_indexed[['close']].join(other_dfs, how='inner')
    combined_df.rename(columns={'close': symbol}, inplace=True)

    # Drop rows with NaNs that might result from joins or missing data
    combined_df.dropna(inplace=True)

    if combined_df.empty:
        print(f"Warning: No overlapping data found for {symbol} and other markets between specified dates.")
        return combined_df, None # Return empty combined_df, None for returns

    # Calculate Log Returns
    returns_df = np.log(combined_df / combined_df.shift(1)).dropna()

    # Return the original df (with date as column) and the returns_df (indexed by date)
    return stock_df, returns_df

def get_default_dates() -> tuple[str, str]:
    """Returns default start and end dates (1 year ago to today)."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    return start_date.isoformat(), end_date.isoformat()

def prepare_data_for_analysis(db: Session, symbol: str | None, start: str | None, end: str | None) -> tuple[str, str, str, pd.DataFrame | None, pd.DataFrame | None, dict[str, pd.DataFrame] | None]:
    """
    Main function to orchestrate data fetching and preparation.
    Returns symbol, start, end, stock_df, returns_df, other_data.
    Handles default values and potential data fetching errors.
    """
    if not symbol:
        symbol = "AAPL" # Default symbol
        
    if not start or not end:
        start, end = get_default_dates()
    
    try:
        stock_df = fetch_stock_data(db, symbol, start, end)
        if stock_df.empty:
            print(f"Warning: No stock data found for symbol {symbol} between {start} and {end}")
            return symbol, start, end, None, None, None # Indicate failure

        other_data = fetch_other_market_data(db, start, end)
        
        stock_df, returns_df = combine_data(stock_df, other_data, symbol)
        
        # Check if combination resulted in empty dataframes
        if stock_df is None or (returns_df is not None and returns_df.empty):
             print(f"Warning: Data combination resulted in empty dataframes for {symbol}")
             # Return original stock_df if it existed but combination failed
             return symbol, start, end, stock_df if stock_df is not None else None, None, other_data

        return symbol, start, end, stock_df, returns_df, other_data

    except Exception as e:
        print(f"Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return symbol, start, end, None, None, None # Indicate failure 