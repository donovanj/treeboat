# Placeholder for volume analysis calculations 

import pandas as pd
import numpy as np
import json
from plotly.utils import PlotlyJSONEncoder

# Import the shared helper function from price_analysis
from .price_analysis import create_box_violin_pair

# --- Volume Analysis Functions ---

def calculate_volume_distribution(df: pd.DataFrame, symbol: str) -> tuple[dict | None, dict | None]:
    """Calculates volume distribution plots by month."""
    if df is None or df.empty or not all(c in df.columns for c in ['volume', 'year_month']):
        return None, None
    return create_box_violin_pair(
        df['volume'].values, df['year_month'], f"{symbol} Volume Distribution by Month", "Volume"
    )

def calculate_price_volume_analysis(df: pd.DataFrame, symbol: str) -> tuple[dict | None, dict | None]:
    """Compares volume on up days vs down days."""
    if df is None or df.empty or not all(c in df.columns for c in ['close', 'volume']) or len(df) < 2:
        return None, None

    returns = df['close'].pct_change()
    volume = df['volume'].values
    
    # Align returns and volume (iloc[1:] matches returns length)
    if len(returns) != len(volume):
         # Handle edge case where pct_change might yield different length unexpectedly
         # Usually happens if df has only 1 row after processing
         # Align volume to returns length if volume is longer
         if len(volume) == len(returns) + 1:
             volume = volume[1:] 
         else:
            print("Warning: Volume and returns length mismatch in price_volume analysis.")
            return None, None
            
    up_mask = (returns > 0).values
    down_mask = (returns <= 0).values

    # Ensure masks are boolean and correct length
    if up_mask.shape != volume.shape or down_mask.shape != volume.shape:
        print("Warning: Mask shape mismatch in price_volume analysis.")
        return None, None

    volume_up = volume[up_mask]
    volume_down = volume[down_mask]

    if len(volume_up) == 0 or len(volume_down) == 0:
        print("Warning: No up or down days found for price_volume analysis.")
        return None, None

    # Use a different category mechanism than create_box_violin_pair for this
    categories = ['Up Days'] * len(volume_up) + ['Down Days'] * len(volume_down)
    all_volumes = np.concatenate([volume_up, volume_down])
    
    temp_df = pd.DataFrame({'value': all_volumes, 'category': categories})
    unique_cats = ['Up Days', 'Down Days'] # Fixed categories
    cat_labels = unique_cats
    grouped_data = [temp_df['value'][temp_df['category'] == cat].tolist() for cat in unique_cats]

    box_data = {
        "data": [
            {
                "type": "box", "y": y_data, "name": cat, "boxmean": True
            } for cat, y_data in zip(cat_labels, grouped_data)
        ],
        "layout": {
            "title": f"{symbol} Volume on Up vs Down Days",
            "yaxis": {"title": "Volume"}
        }
    }
    violin_data = {
        "data": [
            {
                "type": "violin", "y": y_data, "name": cat, 
                "box": {"visible": True}, "meanline": {"visible": True}
            } for cat, y_data in zip(cat_labels, grouped_data)
        ],
        "layout": {
            "title": f"{symbol} Volume on Up vs Down Days",
            "yaxis": {"title": "Volume"}
        }
    }

    return json.loads(json.dumps(box_data, cls=PlotlyJSONEncoder)), json.loads(json.dumps(violin_data, cls=PlotlyJSONEncoder))

def calculate_gap_volume_scatter(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates scatter plot of overnight gaps vs volume with color coding for up/down days."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    returns = df['close'].pct_change() * 100
    
    # Create color categories
    colors = np.where(returns > 0, 'green', 'red')
    
    fig = {
        "data": [
            {
                "type": "scatter",
                "x": gaps.values[1:],  # Skip first day (no gap)
                "y": df['volume'].values[1:],
                "mode": "markers",
                "marker": {
                    "color": colors[1:],
                    "size": 8,
                    "opacity": 0.6
                },
                "name": "Gap-Volume Points"
            }
        ],
        "layout": {
            "title": f"{symbol} Overnight Gaps vs Volume",
            "xaxis": {"title": "Overnight Gap (%)", "zeroline": True},
            "yaxis": {"title": "Volume", "type": "log"},
            "showlegend": False
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_volume_bubble(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates bubble chart of gaps over time with volume sizing."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume', 'date']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    returns = df['close'].pct_change() * 100
    
    # Normalize volume for bubble sizes (sqrt for visual scaling)
    volume_norm = np.sqrt(df['volume'] / df['volume'].max()) * 50  # Max bubble size 50
    
    # Create color categories
    colors = np.where(returns > 0, 'green', 'red')
    
    fig = {
        "data": [
            {
                "type": "scatter",
                "x": df['date'].values[1:],
                "y": gaps.values[1:],
                "mode": "markers",
                "marker": {
                    "size": volume_norm[1:],
                    "color": colors[1:],
                    "opacity": 0.6,
                    "sizemode": "area"
                },
                "name": "Gaps"
            }
        ],
        "layout": {
            "title": f"{symbol} Gap Events Over Time",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Gap Size (%)", "zeroline": True},
            "showlegend": False
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_volume_heatmap(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates heatmap of gap size vs volume buckets."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Create bins for gaps and volume
    gap_bins = np.linspace(gaps.quantile(0.01), gaps.quantile(0.99), 10)
    volume_bins = np.linspace(df['volume'].quantile(0.01), df['volume'].quantile(0.99), 10)
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        gaps.values[1:],
        df['volume'].values[1:],
        bins=[gap_bins, volume_bins]
    )
    
    # Get bin centers for labels
    gap_centers = (gap_bins[:-1] + gap_bins[1:]) / 2
    volume_centers = (volume_bins[:-1] + volume_bins[1:]) / 2
    
    fig = {
        "data": [
            {
                "type": "heatmap",
                "z": H.T,  # Transpose for correct orientation
                "x": [f"{x:.1f}%" for x in gap_centers],
                "y": [f"{y:.0f}" for y in volume_centers],
                "colorscale": "Viridis"
            }
        ],
        "layout": {
            "title": f"{symbol} Gap Size vs Volume Distribution",
            "xaxis": {"title": "Gap Size (%)"},
            "yaxis": {"title": "Volume", "type": "log"},
            "colorbar": {"title": "Frequency"}
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_follow_through(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates scatter plot of gap size vs subsequent intraday movement."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'high', 'low', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps and intraday ranges
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    intraday_move = (df['close'] - df['open']) / df['open'] * 100
    
    # Normalize volume for marker sizes (sqrt for visual scaling)
    volume_norm = np.sqrt(df['volume'] / df['volume'].max()) * 50  # Max marker size 50
    
    # Color based on whether gap was filled
    gap_filled = np.where(
        gaps > 0,  # Gap up
        df['low'] <= df['close'].shift(1),  # Filled if low touches previous close
        df['high'] >= df['close'].shift(1)  # Filled if high touches previous close
    )
    colors = np.where(gap_filled[1:], 'blue', 'red')
    
    fig = {
        "data": [
            {
                "type": "scatter",
                "x": gaps.values[1:],
                "y": intraday_move.values[1:],
                "mode": "markers",
                "marker": {
                    "size": volume_norm[1:],
                    "color": colors,
                    "opacity": 0.6,
                    "sizemode": "area"
                },
                "name": "Gap Follow-through"
            }
        ],
        "layout": {
            "title": f"{symbol} Gap Follow-through Analysis",
            "xaxis": {"title": "Overnight Gap (%)"},
            "yaxis": {"title": "Intraday Move (%)"},
            "showlegend": False,
            "shapes": [
                {
                    "type": "line",
                    "x0": gaps.min(),
                    "x1": gaps.max(),
                    "y0": 0,
                    "y1": 0,
                    "line": {"color": "black", "dash": "dash"}
                }
            ]
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_category_boxplot(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates boxplots of volume distribution by gap categories."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Create gap categories
    def categorize_gap(gap):
        if pd.isna(gap):
            return 'No Gap'
        elif gap <= -1:
            return 'Large Gap Down'
        elif gap < 0:
            return 'Small Gap Down'
        elif gap == 0:
            return 'No Gap'
        elif gap <= 1:
            return 'Small Gap Up'
        else:
            return 'Large Gap Up'
    
    categories = pd.Series([categorize_gap(g) for g in gaps])
    category_order = ['Large Gap Down', 'Small Gap Down', 'No Gap', 'Small Gap Up', 'Large Gap Up']
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Volume': df['volume'].values,
        'Category': categories
    })
    
    # Group data by category
    grouped_data = [plot_data[plot_data['Category'] == cat]['Volume'].tolist() 
                   for cat in category_order if len(plot_data[plot_data['Category'] == cat]) > 0]
    valid_categories = [cat for cat in category_order 
                       if len(plot_data[plot_data['Category'] == cat]) > 0]
    
    fig = {
        "data": [
            {
                "type": "box",
                "y": data,
                "name": cat,
                "boxmean": True
            } for data, cat in zip(grouped_data, valid_categories)
        ],
        "layout": {
            "title": f"{symbol} Volume Distribution by Gap Category",
            "xaxis": {"title": "Gap Category"},
            "yaxis": {"title": "Volume", "type": "log"},
            "showlegend": False
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_volume_timeseries(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates dual-axis time series of volume and gaps."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume', 'date']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    returns = df['close'].pct_change() * 100
    
    fig = {
        "data": [
            {
                "type": "scatter",
                "x": df['date'].values[1:],
                "y": df['volume'].values[1:],
                "name": "Volume",
                "yaxis": "y",
                "line": {"color": "blue"}
            },
            {
                "type": "scatter",
                "x": df['date'].values[1:],
                "y": gaps.values[1:],
                "name": "Overnight Gap",
                "yaxis": "y2",
                "line": {"color": "red"},
                "mode": "lines+markers",
                "marker": {
                    "color": np.where(returns[1:] > 0, 'green', 'red'),
                    "size": 6
                }
            }
        ],
        "layout": {
            "title": f"{symbol} Volume and Gaps Over Time",
            "xaxis": {"title": "Date"},
            "yaxis": {
                "title": "Volume",
                "type": "log",
                "side": "left"
            },
            "yaxis2": {
                "title": "Gap Size (%)",
                "overlaying": "y",
                "side": "right"
            },
            "showlegend": True,
            "legend": {"x": 0, "y": 1.1, "orientation": "h"}
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_volume_weighted_gaps(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates volume-weighted gap distribution."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Create histogram bins
    counts, bins = np.histogram(gaps.dropna(), bins=30)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Calculate average volume for each bin
    volumes = []
    for i in range(len(bins)-1):
        mask = (gaps >= bins[i]) & (gaps < bins[i+1])
        avg_vol = df['volume'][mask].mean() if any(mask) else 0
        volumes.append(avg_vol)
    
    # Normalize volumes for bar width
    max_width = 0.8  # Maximum width of bars
    norm_volumes = np.array(volumes) / max(volumes) * max_width
    
    fig = {
        "data": [
            {
                "type": "bar",
                "x": bin_centers,
                "y": counts,
                "width": norm_volumes,
                "marker": {
                    "color": "blue",
                    "opacity": 0.6
                },
                "name": "Gap Distribution"
            }
        ],
        "layout": {
            "title": f"{symbol} Volume-Weighted Gap Distribution",
            "xaxis": {"title": "Gap Size (%)"},
            "yaxis": {"title": "Frequency"},
            "showlegend": False,
            "bargap": 0
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_volume_surface(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates 3D surface plot of gaps, volume, and frequency."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Create 2D histogram data with volume as z-axis
    gap_bins = np.linspace(gaps.quantile(0.01), gaps.quantile(0.99), 20)
    volume_bins = np.linspace(df['volume'].quantile(0.01), df['volume'].quantile(0.99), 20)
    
    H, xedges, yedges = np.histogram2d(
        gaps.values[1:],
        df['volume'].values[1:],
        bins=[gap_bins, volume_bins]
    )
    
    # Get bin centers
    gap_centers = (gap_bins[:-1] + gap_bins[1:]) / 2
    volume_centers = (volume_bins[:-1] + volume_bins[1:]) / 2
    
    # Create meshgrid for surface plot
    X, Y = np.meshgrid(gap_centers, volume_centers)
    
    fig = {
        "data": [
            {
                "type": "surface",
                "x": X,
                "y": Y,
                "z": H.T,
                "colorscale": "Viridis",
                "name": "Gap-Volume Surface"
            }
        ],
        "layout": {
            "title": f"{symbol} Gap-Volume-Frequency Surface",
            "scene": {
                "xaxis": {"title": "Gap Size (%)"},
                "yaxis": {"title": "Volume"},
                "zaxis": {"title": "Frequency"}
            }
        }
    }
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def calculate_gap_violin_strip(df: pd.DataFrame, symbol: str) -> dict | None:
    """Creates combined violin and strip plot by gap categories."""
    if df is None or df.empty or not all(c in df.columns for c in ['open', 'close', 'volume']) or len(df) < 2:
        return None
        
    # Calculate gaps as percentage
    gaps = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    
    # Create gap categories
    def categorize_gap(gap):
        if pd.isna(gap):
            return 'No Gap'
        elif gap <= -1:
            return 'Large Gap Down'
        elif gap < 0:
            return 'Small Gap Down'
        elif gap == 0:
            return 'No Gap'
        elif gap <= 1:
            return 'Small Gap Up'
        else:
            return 'Large Gap Up'
    
    categories = pd.Series([categorize_gap(g) for g in gaps])
    category_order = ['Large Gap Down', 'Small Gap Down', 'No Gap', 'Small Gap Up', 'Large Gap Up']
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'Volume': df['volume'].values,
        'Category': categories
    })
    
    # Group data by category
    grouped_data = [plot_data[plot_data['Category'] == cat]['Volume'].tolist() 
                   for cat in category_order if len(plot_data[plot_data['Category'] == cat]) > 0]
    valid_categories = [cat for cat in category_order 
                       if len(plot_data[plot_data['Category'] == cat]) > 0]
    
    # Create violin plots with individual points
    fig = {
        "data": [],
        "layout": {
            "title": f"{symbol} Volume Distribution by Gap Category (Violin + Strip)",
            "xaxis": {"title": "Gap Category"},
            "yaxis": {"title": "Volume", "type": "log"},
            "showlegend": False,
            "violingap": 0.2,
            "violinmode": "overlay"
        }
    }
    
    for data, cat in zip(grouped_data, valid_categories):
        # Add violin plot
        fig["data"].append({
            "type": "violin",
            "y": data,
            "name": cat,
            "box": {"visible": True},
            "meanline": {"visible": True},
            "side": "positive",
            "width": 0.8,
            "points": False
        })
        
        # Add strip plot (individual points)
        fig["data"].append({
            "type": "box",
            "y": data,
            "name": cat,
            "boxpoints": "all",
            "jitter": 0.3,
            "pointpos": 0,
            "marker": {"size": 3, "opacity": 0.4},
            "showbox": False,
            "width": 0.2
        })
    
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))

def run_all_volume_analyses(df: pd.DataFrame, symbol: str) -> dict:
    """Runs all volume analysis calculations and returns results."""
    results = {}
    # Basic volume distribution
    results["volume_dist_box"], results["volume_dist_violin"] = calculate_volume_distribution(df, symbol)
    results["price_volume_box"], results["price_volume_violin"] = calculate_price_volume_analysis(df, symbol)
    
    # Gap-Volume analysis visualizations
    results["gap_volume_scatter"] = calculate_gap_volume_scatter(df, symbol)
    results["gap_volume_bubble"] = calculate_gap_volume_bubble(df, symbol)
    results["gap_volume_heatmap"] = calculate_gap_volume_heatmap(df, symbol)
    results["gap_follow_through"] = calculate_gap_follow_through(df, symbol)
    results["gap_category_box"] = calculate_gap_category_boxplot(df, symbol)
    results["gap_volume_timeseries"] = calculate_gap_volume_timeseries(df, symbol)
    results["volume_weighted_gaps"] = calculate_volume_weighted_gaps(df, symbol)
    results["gap_volume_surface"] = calculate_gap_volume_surface(df, symbol)
    results["gap_violin_strip"] = calculate_gap_violin_strip(df, symbol)
    
    # Filter out None results
    return {k: v for k, v in results.items() if v is not None}