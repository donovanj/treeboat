from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from plotly.utils import PlotlyJSONEncoder
import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objs as go

router = APIRouter()

# Dummy data generation for demonstration. Replace with your real data source.
def get_price_data(start, end):
    dates = pd.date_range(start, end)
    np.random.seed(0)
    prices = np.cumprod(1 + np.random.normal(0, 0.01, len(dates))) * 100
    volumes = np.random.randint(1000, 5000, len(dates))
    df = pd.DataFrame({'date': dates, 'close': prices, 'volume': volumes})
    return df

@router.get("/api/eda")
def eda_endpoint(
    start: str = Query(..., description="Start date in YYYY-MM-DD"),
    end: str = Query(..., description="End date in YYYY-MM-DD")
):
    df = get_price_data(start, end)
    # Trendlines (linear and exponential)
    x = np.arange(len(df))
    y = df['close'].values
    linear_fit = np.polyfit(x, y, 1)
    linear_trend = linear_fit[0] * x + linear_fit[1]
    exp_fit = np.polyfit(x, np.log(y), 1)
    exp_trend = np.exp(exp_fit[1]) * np.exp(exp_fit[0] * x)
    trendlines_fig = {
        "data": [
            go.Scatter(x=df['date'], y=y, mode='lines', name='Price').to_plotly_json(),
            go.Scatter(x=df['date'], y=linear_trend, mode='lines', name='Linear Trend').to_plotly_json(),
            go.Scatter(x=df['date'], y=exp_trend, mode='lines', name='Exponential Trend').to_plotly_json(),
        ],
        "layout": {"title": "Price with Linear and Exponential Trendlines"}
    }
    # ECDF for returns
    returns = df['close'].pct_change().dropna()
    sorted_returns = np.sort(returns)
    ecdf = np.arange(1, len(sorted_returns)+1) / len(sorted_returns)
    ecdf_fig = {
        "data": [go.Scatter(x=sorted_returns, y=ecdf, mode='lines', name='ECDF').to_plotly_json()],
        "layout": {"title": "Empirical Cumulative Distribution of Returns", "xaxis": {"title": "Return"}, "yaxis": {"title": "ECDF"}}
    }
    # Histogram/Distplot for returns and volume
    hist_fig = {
        "data": [
            go.Histogram(x=returns, name='Returns', opacity=0.7).to_plotly_json(),
            go.Histogram(x=df['volume'], name='Volume', opacity=0.7, xaxis='x2', marker_color='orange').to_plotly_json()
        ],
        "layout": {
            "title": "Histogram of Returns and Volume",
            "barmode": "overlay",
            "xaxis": {"title": "Returns"},
            "xaxis2": {"title": "Volume", "overlaying": "x", "side": "top"},
            "yaxis": {"title": "Count"}
        }
    }
    return JSONResponse(
        content=json.loads(json.dumps({
            "trendlines": trendlines_fig,
            "ecdf": ecdf_fig,
            "hist": hist_fig
        }, cls=PlotlyJSONEncoder))
    )
