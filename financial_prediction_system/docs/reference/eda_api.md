# EDA API Reference

## Overview

The Exploratory Data Analysis (EDA) API provides endpoints for calculating and retrieving visualization data for financial assets.

## Base URL

```
/api/eda
```

## Endpoints

### Get Overview Analysis

```
GET /api/eda/overview
```

Returns summary statistics and basic visualizations for the specified stock and time period.

#### Parameters

| Parameter | Type   | Required | Description                             |
|-----------|--------|----------|-----------------------------------------|
| symbol    | string | Yes      | Stock ticker symbol (e.g., "AAPL")      |
| start_date| string | Yes      | Start date in ISO format (YYYY-MM-DD)   |
| end_date  | string | Yes      | End date in ISO format (YYYY-MM-DD)     |

#### Response Format

```json
{
  "candlestick_data": [...],  // OHLCV data for candlestick chart
  "trendline_data": {         // Linear and non-linear trendlines
    "linear": [...],
    "nonlinear": [...]
  },
  "returns_histogram": [...],  // Data for returns histogram
  "ecdf_data": [...],        // Empirical cumulative distribution
  "summary_stats": {         // Summary statistics
    "mean_return": 0.0012,
    "volatility": 0.023,
    ...
  }
}
```

### Get Price Analysis

```
GET /api/eda/price
```

Provides detailed price movement analysis including seasonal patterns and price range statistics.

#### Parameters

| Parameter | Type   | Required | Description                             |
|-----------|--------|----------|-----------------------------------------|
| symbol    | string | Yes      | Stock ticker symbol (e.g., "AAPL")      |
| start_date| string | Yes      | Start date in ISO format (YYYY-MM-DD)   |
| end_date  | string | Yes      | End date in ISO format (YYYY-MM-DD)     |

#### Response Format

```json
{
  "daily_range_by_month": {...},  // Box & violin plot data by month
  "overnight_gap_by_month": {...},  // Gap analysis data
  "close_to_range_by_month": {...}  // Closing price relative to daily range
}
```

### Get Volatility Analysis

```
GET /api/eda/volatility
```

Returns volatility metrics and visualizations for the given security.

#### Parameters

| Parameter | Type   | Required | Description                                |
|-----------|--------|----------|--------------------------------------------|  
| symbol    | string | Yes      | Stock ticker symbol (e.g., "AAPL")         |
| start_date| string | Yes      | Start date in ISO format (YYYY-MM-DD)      |
| end_date  | string | Yes      | End date in ISO format (YYYY-MM-DD)        |
| window    | integer| No       | Rolling window size (default: 20)          |

#### Response Format

```json
{
  "rolling_volatility": {...},  // Rolling standard deviation data
  "parkinson_volatility": {...},  // Parkinson high-low volatility
  "normalized_atr": {...},  // Average True Range data
  "volatility_price_comparison": {...},  // Volatility vs. price
  "volatility_term_structure": {...},  // Term structure heatmap
  "volatility_cone": {...},  // Min/max/avg volatility by time horizon
  "volatility_regimes": {...}  // Volatility regime classification
}
```

## Error Codes

| Code | Description                                    |
|------|------------------------------------------------|
| 400  | Bad Request - Invalid parameters               |
| 404  | Not Found - Symbol or data not available       |
| 500  | Server Error - Failed to calculate metrics     |

## Implementation Details

This API uses the calculation modules in `eda_calculations/` to generate visualization data. Each endpoint calls specific calculation functions in the corresponding module.

For example, the volatility endpoint uses functions from `volatility_analysis.py` to calculate various volatility metrics. The data preparation is handled by `base_data.py`.