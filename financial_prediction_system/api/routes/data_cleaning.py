from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from financial_prediction_system.infrastructure.database.connection import SessionLocal

router = APIRouter(prefix="/api")

from fastapi import Query

@router.get("/stock-list")
def get_stock_list():
    """Return all active stocks with symbol and company_name for autocomplete."""
    import pandas as pd
    session = SessionLocal()
    try:
        stocks_df = pd.read_sql("SELECT symbol, company_name FROM stocks WHERE is_active = TRUE", session.bind)
        return [{"symbol": row["symbol"], "company_name": row["company_name"]} for _, row in stocks_df.iterrows()]
    finally:
        session.close()

@router.get("/data-cleaning")
def get_cleaning_eda(symbol: str = Query(None)):
    """
    Financial data cleaning and EDA for stocks and stock_prices tables.
    Returns summary, issues, and a sample Plotly candlestick chart JSON.
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.utils
    import io, base64, re, traceback, datetime
    from financial_prediction_system.data_loaders.data_providers import get_market_calendar_dates
    session = SessionLocal()
    try:
        # --- STOCKS TABLE ANALYSIS ---
        stocks_df = pd.read_sql("SELECT * FROM stocks", session.bind)
        now = pd.Timestamp.now()
        summary = {}
        issues = {}
        # Missing values
        for col in ["company_name", "sector", "industry"]:
            summary[f"missing_{col}"] = int(stocks_df[col].isnull().sum())
        # Standardize text fields
        for col in ["sector", "industry"]:
            stocks_df[col] = stocks_df[col].astype(str).str.strip().str.title()
        # Symbol validation
        symbol_regex = re.compile(r"^[A-Z0-9.-]+$")
        invalid_symbols = stocks_df[~stocks_df['symbol'].str.match(symbol_regex)]["symbol"].tolist()
        issues["invalid_symbols"] = invalid_symbols
        # Duplicates
        duplicate_symbols = stocks_df[stocks_df.duplicated("symbol")]["symbol"].tolist()
        issues["duplicate_symbols"] = duplicate_symbols
        # Outdated records
        stocks_df["last_updated"] = pd.to_datetime(stocks_df["last_updated"], errors="coerce")
        outdated = stocks_df[stocks_df["last_updated"] < (now - pd.Timedelta(days=365))]["symbol"].tolist()
        issues["outdated_symbols"] = outdated
        # Inactive stocks
        inactive = stocks_df[~stocks_df["is_active"]]["symbol"].tolist()
        summary["inactive_count"] = len(inactive)
        # --- STOCK_PRICES TABLE ANALYSIS ---
        # Find symbol to analyze
        active_symbols = stocks_df[stocks_df["is_active"]]["symbol"].tolist()
        if symbol:
            sample_symbol = symbol
        else:
            sample_symbol = active_symbols[0] if active_symbols else None
        plotly_fig = None
        price_issues = {}
        if sample_symbol:
            prices_df = pd.read_sql("SELECT * FROM stock_prices WHERE symbol = %s ORDER BY date", session.bind, params=(sample_symbol,))
            if not prices_df.empty:
                # Outliers (z-score)
                for col in ["open", "high", "low", "close", "volume"]:
                    arr = prices_df[col].astype(float)
                    z = np.abs((arr - arr.mean()) / (arr.std() if arr.std() > 0 else 1))
                    price_issues[f"{col}_outliers"] = int((z > 3).sum())
                # Missing dates (using official market calendar)
                prices_df["date"] = pd.to_datetime(prices_df["date"])
                min_date = prices_df["date"].min().date()
                max_date = prices_df["date"].max().date()
                market_dates = get_market_calendar_dates(min_date, max_date)
                present_dates = set(prices_df["date"].dt.date)
                missing_dates = [str(d) for d in market_dates if d not in present_dates]
                price_issues["missing_dates"] = missing_dates[:10]
                # Price consistency
                inconsistent = prices_df[(prices_df["low"] > prices_df["open"]) |
                                         (prices_df["close"] > prices_df["high"]) |
                                         (prices_df["low"] > prices_df["close"]) |
                                         (prices_df["open"] > prices_df["high"])]
                price_issues["inconsistent_rows"] = inconsistent.head(5).to_dict(orient="records")
                # Log(volume) for visualization
                prices_df["log_volume"] = np.log1p(prices_df["volume"])
                # --- EDA VISUALIZATIONS ---
                # 1. OHLCV: Close line plot + volume bars
                ohlcv_fig = go.Figure()
                ohlcv_fig.add_trace(go.Scatter(x=prices_df['date'], y=prices_df['close'], name='Close', line=dict(color='blue')))
                ohlcv_fig.add_trace(go.Bar(x=prices_df['date'], y=prices_df['volume'], name='Volume', yaxis='y2', marker_color='orange', opacity=0.3))
                ohlcv_fig.update_layout(
                    title=f"{sample_symbol} Close Price & Volume",
                    xaxis_title="Date",
                    yaxis=dict(title='Close'),
                    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                    barmode='overlay',
                    legend=dict(x=0, y=1.1, orientation='h')
                )
                # 2. Bands: High-Low, Bollinger Bands, ATR
                high = prices_df['high']
                low = prices_df['low']
                close = prices_df['close']
                rolling_mean = close.rolling(window=20).mean()
                rolling_std = close.rolling(window=20).std()
                upper_band = rolling_mean + 2 * rolling_std
                lower_band = rolling_mean - 2 * rolling_std
                tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
                atr = tr.rolling(window=14).mean()
                bands_fig = go.Figure()
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=close, name='Close', line=dict(color='blue')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=high, name='High', line=dict(color='green', dash='dot')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=low, name='Low', line=dict(color='red', dash='dot')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=upper_band, name='Boll Upper', line=dict(color='gray', dash='dash')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=lower_band, name='Boll Lower', line=dict(color='gray', dash='dash')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=rolling_mean, name='MA20', line=dict(color='purple', dash='dot')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=close+atr, name='ATR+', line=dict(color='orange', dash='dot')))
                bands_fig.add_trace(go.Scatter(x=prices_df['date'], y=close-atr, name='ATR-', line=dict(color='orange', dash='dot')))
                bands_fig.update_layout(title=f"{sample_symbol} Bands & Ranges", xaxis_title="Date", yaxis_title="Price")
                # 3. Trendlines: Linear & Exponential
                x = np.arange(len(prices_df))
                y = close.values
                linear_fit = np.polyfit(x, y, 1)
                linear_trend = linear_fit[0] * x + linear_fit[1]
                exp_fit = np.polyfit(x, np.log(y + 1e-8), 1)
                exp_trend = np.exp(exp_fit[1]) * np.exp(exp_fit[0] * x)
                trendlines_fig = go.Figure()
                trendlines_fig.add_trace(go.Scatter(x=prices_df['date'], y=y, name='Close', line=dict(color='blue')))
                trendlines_fig.add_trace(go.Scatter(x=prices_df['date'], y=linear_trend, name='Linear Trend', line=dict(color='orange')))
                trendlines_fig.add_trace(go.Scatter(x=prices_df['date'], y=exp_trend, name='Exp Trend', line=dict(color='green', dash='dot')))
                trendlines_fig.update_layout(title=f"{sample_symbol} Trendlines", xaxis_title="Date", yaxis_title="Price")
                # 4. Descriptive stats
                returns = close.pct_change().dropna()
                describe_stats = {
                    'close': prices_df['close'].describe().to_dict(),
                    'volume': prices_df['volume'].describe().to_dict(),
                    'returns': returns.describe().to_dict()
                }
                # 5. Missingness heatmap
                missingness = prices_df.isnull().astype(int)
                if missingness.values.sum() > 0:
                    missing_heatmap_fig = go.Figure(data=[go.Heatmap(z=missingness.values.T, x=prices_df['date'], y=missingness.columns, colorscale='Reds')])
                    missing_heatmap_fig.update_layout(title="Missing Data Heatmap", xaxis_title="Date", yaxis_title="Field")
                    missing_heatmap_fig = missing_heatmap_fig.to_plotly_json()
                else:
                    missing_heatmap_fig = None
                # 6. Box/Violin plots
                box_violin_fig = go.Figure()
                box_violin_fig.add_trace(go.Box(y=prices_df['close'], name='Close'))
                box_violin_fig.add_trace(go.Box(y=prices_df['volume'], name='Volume'))
                box_violin_fig.add_trace(go.Box(y=returns, name='Returns'))
                box_violin_fig.add_trace(go.Violin(y=prices_df['close'], name='Close', box_visible=True, meanline_visible=True))
                box_violin_fig.add_trace(go.Violin(y=prices_df['volume'], name='Volume', box_visible=True, meanline_visible=True))
                box_violin_fig.add_trace(go.Violin(y=returns, name='Returns', box_visible=True, meanline_visible=True))
                box_violin_fig.update_layout(title=f"{sample_symbol} Box/Violin Plots")
                # Plotly candlestick chart (legacy)
                fig = go.Figure(data=[go.Candlestick(
                    x=list(prices_df['date']),
                    open=list(prices_df['open']),
                    high=list(prices_df['high']),
                    low=list(prices_df['low']),
                    close=list(prices_df['close']),
                    name=sample_symbol
                )])
                fig.update_layout(
                    title=f"{sample_symbol} Price & Volume",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                plotly_fig = fig.to_plotly_json()
                # Convert all new figs to JSON
                ohlcv_fig = ohlcv_fig.to_plotly_json()
                bands_fig = bands_fig.to_plotly_json()
                trendlines_fig = trendlines_fig.to_plotly_json()
                box_violin_fig = box_violin_fig.to_plotly_json()
        # --- RESPONSE ---
        import json
        from plotly.utils import PlotlyJSONEncoder
        def to_serializable(obj):
            if obj is None:
                return None
            return json.loads(json.dumps(obj, cls=PlotlyJSONEncoder))
        return {
            "success": True,
            "summary": summary,
            "issues": issues,
            "price_issues": price_issues,
            "sample_symbol": sample_symbol,
            "plotly_fig": to_serializable(plotly_fig) if plotly_fig else None,
            "ohlcv_fig": to_serializable(ohlcv_fig) if ohlcv_fig else None,
            "bands_fig": to_serializable(bands_fig) if bands_fig else None,
            "trendlines_fig": to_serializable(trendlines_fig) if trendlines_fig else None,
            "describe_stats": to_serializable(describe_stats) if describe_stats else None,
            "missing_heatmap_fig": to_serializable(missing_heatmap_fig) if missing_heatmap_fig else None,
            "box_violin_fig": to_serializable(box_violin_fig) if box_violin_fig else None
        }
    except Exception as e:
        return {"success": False, "error": str(e), "trace": traceback.format_exc()}
    finally:
        session.close()
