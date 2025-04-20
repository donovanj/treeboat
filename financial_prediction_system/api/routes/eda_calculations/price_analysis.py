# Placeholder for price analysis calculations 

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import json
from plotly.utils import PlotlyJSONEncoder
from datetime import datetime

# --- Helper function for box/violin plots (moved here as it's used by price/volume) ---
def create_box_violin_pair(values, by_category, title, y_title):
    """Generates paired box and violin plots, grouping by the provided category series."""
    if values is None or by_category is None or len(values) != len(by_category):
        print(f"Warning: Mismatched lengths or None input for box/violin plot: {title}")
        return None, None

    temp_df = pd.DataFrame({'value': values, 'category': by_category})
    temp_df.dropna(inplace=True) # Drop rows with NaNs in value or category

    if temp_df.empty:
        print(f"Warning: Empty DataFrame after dropping NaNs for box/violin plot: {title}")
        return None, None
        
    # Attempt to sort categories chronologically if they look like 'YYYY-MM'
    try:
        unique_cats = sorted(temp_df['category'].unique(), key=lambda x: datetime.strptime(x, '%Y-%m'))
        cat_labels = [datetime.strptime(cat, '%Y-%m').strftime('%b %Y') for cat in unique_cats]
    except (ValueError, TypeError): # Handle non-date strings or other types
        unique_cats = sorted(temp_df['category'].unique())
        cat_labels = [str(cat) for cat in unique_cats] # Fallback to string representation
            
    grouped_data = [temp_df['value'][temp_df['category'] == cat].tolist() for cat in unique_cats]
    
    # Filter out empty groups before plotting
    valid_indices = [i for i, data in enumerate(grouped_data) if len(data) > 0]
    if not valid_indices:
        print(f"Warning: No valid data groups found for box/violin plot: {title}")
        return None, None
        
    filtered_labels = [cat_labels[i] for i in valid_indices]
    filtered_data = [grouped_data[i] for i in valid_indices]

    box_data = {
        "data": [
            {
                "type": "box",
                "y": y_data,
                "name": cat,
                "boxmean": True # Show mean
            } for cat, y_data in zip(filtered_labels, filtered_data)
        ],
        "layout": {
            "title": f"{title} (Box Plot)",
            "yaxis": {"title": y_title},
            "xaxis": {"title": "Category"} # Add category axis title
        }
    }
    
    violin_data = {
        "data": [
            {
                "type": "violin",
                "y": y_data,
                "name": cat,
                "box": {"visible": True},
                "meanline": {"visible": True}
            } for cat, y_data in zip(filtered_labels, filtered_data)
        ],
        "layout": {
            "title": f"{title} (Violin Plot)",
            "yaxis": {"title": y_title},
             "xaxis": {"title": "Category"} # Add category axis title
        }
    }
    
    return json.loads(json.dumps(box_data, cls=PlotlyJSONEncoder)), json.loads(json.dumps(violin_data, cls=PlotlyJSONEncoder))

# --- Price Analysis Functions ---

def calculate_candlestick(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates the candlestick chart data."""
    if df is None or df.empty:
        return None
    return {
        "data": [
            {
                "type": "candlestick",
                "x": df['date'].tolist(),
                "open": df['open'].tolist(),
                "high": df['high'].tolist(),
                "low": df['low'].tolist(),
                "close": df['close'].tolist(),
                "name": symbol
            },
            {
                "type": "bar",
                "x": df['date'].tolist(),
                "y": df['volume'].tolist(),
                "yaxis": "y2",
                "name": "Volume",
                "marker": {"color": "rgba(0,0,255,0.3)"}
            }
        ],
        "layout": {
            "title": f"{symbol} Price Chart",
            "yaxis": {"title": "Price", "type": "log"},
            "yaxis2": {
                "title": "Volume",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
                "type": "log" # Log scale for volume
            }
        }
    }

def calculate_trendlines(df: pd.DataFrame, symbol: str) -> dict | None:
    """Calculates linear and exponential trendlines."""
    if df is None or df.empty or len(df) < 2:
        return None
        
    x = np.arange(len(df))
    y = df['close'].values
    y_log = np.log(y + 1e-10) # Add small epsilon for log calculation stability
    
    try:
        # Linear Trend
        linear_fit = np.polyfit(x, y, 1)
        linear_trend = linear_fit[0] * x + linear_fit[1]
        
        # Exponential Trend (fit on log(y))
        exp_fit = np.polyfit(x, y_log, 1)
        exp_trend = np.exp(exp_fit[1]) * np.exp(exp_fit[0] * x)
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Error calculating trendlines for {symbol}: {e}")
        return None

    return {
        "data": [
            {
                "type": "candlestick",
                "x": df['date'].tolist(),
                "open": df['open'].tolist(),
                "high": df['high'].tolist(),
                "low": df['low'].tolist(),
                "close": df['close'].tolist(),
                "name": symbol,
                "showlegend": False # Hide legend for candlestick in trend plot
            },
            {
                "type": "scatter",
                "x": df['date'].tolist(),
                "y": linear_trend.tolist(),
                "name": "Linear Trend",
                "mode": "lines",
                "line": {"color": "red", "width": 2}
            },
            {
                "type": "scatter",
                "x": df['date'].tolist(),
                "y": exp_trend.tolist(), 
                "name": "Exp Trend",
                "mode": "lines",
                "line": {"color": "green", "width": 2, "dash": "dash"}
            }
        ],
        "layout": {
            "title": f"{symbol} Price with Trendlines",
            "yaxis": {"title": "Price", "type": "log"}, # Log scale for price
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1} # Adjust legend position
        }
    }

def calculate_returns_ecdf(df: pd.DataFrame, symbol: str) -> dict | None:
    """Calculates the ECDF for daily returns."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < 2:
        return None
        
    returns = df['close'].pct_change().dropna()
    if returns.empty:
        return None
        
    sorted_returns = np.sort(returns)
    ecdf_y = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    return {
        "data": [
            {
                "type": "scatter",
                "x": sorted_returns.tolist(),
                "y": ecdf_y.tolist(),
                "mode": "lines",
                "name": "ECDF",
                "line": {"color": "blue", "width": 2}
            }
        ],
        "layout": {
            "title": f"{symbol} Daily Returns ECDF",
            "xaxis": {"title": "Daily Return"},
            "yaxis": {"title": "Cumulative Probability"}
        }
    }

def calculate_returns_histogram(df: pd.DataFrame, symbol: str) -> dict | None:
    """Calculates the histogram for daily returns."""
    if df is None or df.empty or 'close' not in df.columns or len(df) < 2:
        return None
        
    returns = df['close'].pct_change().dropna()
    if returns.empty:
        return None

    return {
        "data": [
            {
                "type": "histogram",
                "x": returns.tolist(),
                "nbinsx": 50, # Increased bins for potentially better detail
                "name": "Returns Distribution",
                "marker": {"color": "rgba(0,100,80,0.7)"} # Changed color
            }
        ],
        "layout": {
            "title": f"{symbol} Daily Returns Distribution",
            "xaxis": {"title": "Daily Return"},
            "yaxis": {"title": "Frequency"},
            "bargap": 0.1 # Add a small gap between bars
        }
    }

def calculate_price_range_analysis(df: pd.DataFrame, symbol: str) -> tuple[dict | None, dict | None]:
    """Calculates price range (High - Low) analysis plots."""
    if df is None or df.empty or not all(c in df.columns for c in ['high', 'low', 'year_month']):
        return None, None
    price_range = df['high'] - df['low']
    return create_box_violin_pair(
        price_range.values, df['year_month'], f"{symbol} Price Range by Month", "High - Low"
    )

def calculate_gap_analysis(df: pd.DataFrame, symbol: str) -> tuple[dict | None, dict | None]:
    """Calculates overnight gap analysis plots."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'year_month']) or len(df) < 2:
        return None, None
    gaps = df['open'].iloc[1:].values - df['close'].iloc[:-1].values
    gaps = np.insert(gaps, 0, np.nan) # Add NaN for the first day, keep length same as df
    return create_box_violin_pair(
        gaps, df['year_month'], f"{symbol} Overnight Gaps by Month", "Open - Previous Close"
    )

def calculate_close_positioning_analysis(df: pd.DataFrame, symbol: str) -> tuple[dict | None, dict | None]:
    """Calculates close relative to range positioning plots."""
    if df is None or df.empty or not all(c in df.columns for c in ['close', 'low', 'high', 'year_month']):
        return None, None
    # Add epsilon to avoid division by zero if high == low
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    close_position = close_position.clip(0, 1) # Ensure values are between 0 and 1
    return create_box_violin_pair(
        close_position.values, df['year_month'], f"{symbol} Close Position Within Range by Month", "(Close - Low) / (High - Low)"
    )


def run_all_price_analyses(df: pd.DataFrame, symbol: str) -> dict:
    """Runs all price analysis calculations and returns results."""
    results = {}
    results["candlestick"] = calculate_candlestick(df, symbol)
    results["trendlines"] = calculate_trendlines(df, symbol)
    results["ecdf"] = calculate_returns_ecdf(df, symbol)
    results["hist"] = calculate_returns_histogram(df, symbol)
    results["price_range_box"], results["price_range_violin"] = calculate_price_range_analysis(df, symbol)
    results["gap_analysis_box"], results["gap_analysis_violin"] = calculate_gap_analysis(df, symbol)
    results["close_positioning_box"], results["close_positioning_violin"] = calculate_close_positioning_analysis(df, symbol)
    
    # Placeholder for other price analyses (multitimeframe, seasonal - would require more data/logic)
    results["returns_dist"] = None # Placeholder - ECDF/Hist cover this
    results["multitimeframe_box"] = None
    results["multitimeframe_violin"] = None
    results["seasonal_box"] = None
    results["seasonal_violin"] = None

    # Filter out None results before returning
    return {k: v for k, v in results.items() if v is not None} 