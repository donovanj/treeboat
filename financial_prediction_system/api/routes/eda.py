from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
from sqlalchemy.orm import Session
from sqlalchemy import text # Import text for raw SQL

from financial_prediction_system.api.dependencies import get_db # Import db dependency

router = APIRouter(prefix="/api")

@router.get("/eda-test")
def eda_test_endpoint():
    """Simple test endpoint to verify the API is working correctly."""
    return {"status": "success", "message": "Test endpoint is working"}

@router.get("/eda")
def eda_endpoint(
    symbol: str = Query(None, description="Stock symbol"),
    start: str = Query(None, description="Start date in YYYY-MM-DD"),
    end: str = Query(None, description="End date in YYYY-MM-DD"),
    db: Session = Depends(get_db) # Inject DB session
):
    """
    Return EDA visualizations for the specified stock and date range.
    """
    # Set default values if not provided
    if not symbol:
        symbol = "AAPL" # Default symbol
        
    if not start or not end:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=365)
        start = start or start_date.isoformat()
        end = end or end_date.isoformat()
    
    try:
        # Construct the SQL query safely using parameters
        sql_query = text("""
            SELECT date, open, high, low, close, volume 
            FROM stock_prices 
            WHERE symbol = :symbol AND date >= :start AND date <= :end
            ORDER BY date ASC
        """)
        
        # Execute query and fetch into Pandas DataFrame
        df = pd.read_sql(
            sql_query, 
            db.connection(), # Use the connection from the session
            params={'symbol': symbol, 'start': start, 'end': end}, 
            parse_dates=['date'] # Automatically parse the date column
        )

        if df.empty:
             return JSONResponse(
                 status_code=404, 
                 content={"error": f"No data found for symbol {symbol} between {start} and {end}"}
             )

        # Ensure correct data types (especially numeric)
        for col in ['open', 'high', 'low', 'close', 'volume']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True) # Drop rows where conversion failed
        
        # --- Create Year-Month column for chronological grouping --- 
        df['year_month'] = df['date'].dt.strftime('%Y-%m')
        
        # Create candlestick chart
        candlestick = {
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
                "yaxis": {"title": "Price"},
                "yaxis2": {
                    "title": "Volume",
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": False
                }
            }
        }
        
        # Trendlines (linear and exponential)
        x = np.arange(len(df))
        y = df['close'].values
        
        linear_fit = np.polyfit(x, y, 1)
        linear_trend = linear_fit[0] * x + linear_fit[1]
        
        exp_fit = np.polyfit(x, np.log(y), 1)
        exp_trend = np.exp(exp_fit[1]) * np.exp(exp_fit[0] * x)
        
        trendlines_fig = {
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
                "yaxis": {"title": "Price"}
            }
        }
        
        # ECDF for returns
        returns = df['close'].pct_change().dropna()
        sorted_returns = np.sort(returns)
        ecdf = np.arange(1, len(sorted_returns)+1) / len(sorted_returns)
        
        ecdf_fig = {
            "data": [
                {
                    "type": "scatter",
                    "x": sorted_returns.tolist(),
                    "y": ecdf.tolist(),
                    "mode": "lines",
                    "name": "ECDF",
                    "line": {"color": "blue", "width": 2}
                }
            ],
            "layout": {
                "title": f"{symbol} Returns ECDF",
                "xaxis": {"title": "Return"},
                "yaxis": {"title": "Probability"}
            }
        }
        
        # Histogram/Distplot for returns
        hist_fig = {
            "data": [
                {
                    "type": "histogram",
                    "x": returns.tolist(),
                    "nbinsx": 30,
                    "name": "Returns Distribution",
                    "marker": {"color": "rgba(0,0,255,0.7)"}
                }
            ],
            "layout": {
                "title": f"{symbol} Returns Distribution",
                "xaxis": {"title": "Return"},
                "yaxis": {"title": "Frequency"}
            }
        }
        
        # --- Updated Helper function for box/violin plots --- 
        def create_box_violin_pair(values, by_category, title, y_title):
            """Generates paired box and violin plots, grouping by the provided category series."""
            
            temp_df = pd.DataFrame({'value': values, 'category': by_category})
            # Sort by category (which should be chronologically sortable, e.g., 'YYYY-MM')
            unique_cats = sorted(temp_df['category'].unique())
            
            grouped_data = [temp_df['value'][temp_df['category'] == cat].tolist() for cat in unique_cats]
            
            # --- Format labels from 'YYYY-MM' to 'Mon YYYY' --- 
            cat_labels = []
            for cat in unique_cats:
                try:
                    # Parse 'YYYY-MM' and format to 'Mon YYYY' (e.g., "Oct 2024")
                    dt_obj = datetime.strptime(cat, '%Y-%m')
                    cat_labels.append(dt_obj.strftime('%b %Y'))
                except ValueError:
                    cat_labels.append(str(cat)) # Fallback to original string if parsing fails
            # --------------------------------------------------------
            
            box_data = {
                "data": [
                    {
                        "type": "box",
                        "y": y_data,
                        "name": cat,
                        "boxmean": True
                    } for cat, y_data in zip(cat_labels, grouped_data) if len(y_data) > 0
                ],
                "layout": {
                    "title": f"{title} (Box Plot)",
                    "yaxis": {"title": y_title}
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
                    } for cat, y_data in zip(cat_labels, grouped_data) if len(y_data) > 0
                ],
                "layout": {
                    "title": f"{title} (Violin Plot)",
                    "yaxis": {"title": y_title}
                }
            }
            
            return box_data, violin_data
        
        # Price range analysis - Use year_month for grouping
        price_range = df['high'] - df['low']
        price_range_box, price_range_violin = create_box_violin_pair(
            price_range.values, df['year_month'], f"{symbol} Price Range by Month", "High - Low"
        )
        
        # Gap analysis (overnight gaps) - Use year_month for grouping
        gaps = df['open'].values[1:] - df['close'].values[:-1]
        gaps = np.insert(gaps, 0, 0) # Add a 0 for the first day
        # Ensure gaps array matches df length if using year_month from original df
        if len(gaps) == len(df):
            gap_analysis_box, gap_analysis_violin = create_box_violin_pair(
                gaps, df['year_month'], f"{symbol} Overnight Gaps by Month", "Open - Previous Close"
            )
        else: # Handle potential length mismatch (though unlikely with insert)
             gap_analysis_box, gap_analysis_violin = None, None
        
        # Close relative to range positioning - Use year_month for grouping
        close_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)  
        close_positioning_box, close_positioning_violin = create_box_violin_pair(
            close_position.values, df['year_month'], f"{symbol} Close Position Within Range", "(Close - Low) / (High - Low)"
        )

        # Volume distribution by month - Use year_month for grouping
        volume_dist_box, volume_dist_violin = create_box_violin_pair(
            df['volume'].values, df['year_month'], f"{symbol} Volume by Month", "Volume"
        )

        # Up/down day categories for price-volume analysis (remains mostly the same, uses boolean indexing)
        returns = df['close'].pct_change() # Recalculate returns matching df length
        up_days = returns > 0
        down_days = returns <= 0
        volume = df['volume'].values

        # Check masks have same length as volume
        if len(up_days) == len(volume):
            volume_up = volume[up_days].tolist()
            volume_down = volume[down_days].tolist()
            
            if len(volume_up) > 0 and len(volume_down) > 0:
                price_volume_box = {
                    "data": [
                        {"type": "box", "y": volume_up, "name": "Up Days", "boxmean": True},
                        {"type": "box", "y": volume_down, "name": "Down Days", "boxmean": True}
                    ],
                    "layout": {
                        "title": f"{symbol} Volume on Up vs Down Days",
                        "yaxis": {"title": "Volume"}
                    }
                }
                price_volume_violin = {
                    "data": [
                        {"type": "violin", "y": volume_up, "name": "Up Days", "box": {"visible": True}, "meanline": {"visible": True}},
                        {"type": "violin", "y": volume_down, "name": "Down Days", "box": {"visible": True}, "meanline": {"visible": True}}
                    ],
                    "layout": {
                        "title": f"{symbol} Volume on Up vs Down Days",
                        "yaxis": {"title": "Volume"}
                    }
                }
            else:
                price_volume_box = None
                price_volume_violin = None
        else:
             price_volume_box = None
             price_volume_violin = None

        # Prepare the complete response
        response_data = {
            "candlestick": candlestick,
            "trendlines": trendlines_fig,
            "ecdf": ecdf_fig,
            "hist": hist_fig,
            "price_range_box": price_range_box,
            "price_range_violin": price_range_violin, 
            "gap_analysis_box": gap_analysis_box,
            "gap_analysis_violin": gap_analysis_violin,
            "close_positioning_box": close_positioning_box,
            "close_positioning_violin": close_positioning_violin,
            "volume_dist_box": volume_dist_box,
            "volume_dist_violin": volume_dist_violin
        }
        
        if price_volume_box and price_volume_violin:
            response_data["price_volume_box"] = price_volume_box
            response_data["price_volume_violin"] = price_volume_violin
            
        # Make sure we convert any numpy types to Python native types for JSON serialization
        return response_data
        
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "trace": traceback.format_exc()
            }
        )
