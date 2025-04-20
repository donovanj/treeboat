from fastapi import APIRouter, Query, Depends
from fastapi.responses import JSONResponse
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.utils import PlotlyJSONEncoder
from sqlalchemy.orm import Session
from sqlalchemy import text
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import plotly.express as px
import traceback # Keep for error handling

from financial_prediction_system.api.dependencies import get_db
from financial_prediction_system.utils.nan_safe_json import NaNHandlingResponse

# Import from the new calculation modules
from .eda_calculations import base_data
from .eda_calculations import price_analysis
from .eda_calculations import volume_analysis
from .eda_calculations import volatility_analysis
from .eda_calculations import correlation_analysis
# Import anomaly detection (placeholder for now, will be populated later)
from .eda_calculations import anomaly_detection 
# --- New Imports ---
from .eda_calculations import spectral_analysis
from .eda_calculations import derived_metrics
# -------------------

# --- Tail Risk ---
from .eda_calculations import tail_risk_analysis
# -----------------

router = APIRouter(prefix="/api")

@router.get("/eda-test")
def eda_test_endpoint():
    """Simple test endpoint to verify the API is working correctly."""
    return {"status": "success", "message": "Test endpoint is working"}

@router.get("/eda", response_class=NaNHandlingResponse)
def eda_endpoint(
    symbol: str = Query(None, description="Stock symbol"),
    start: str = Query(None, description="Start date in YYYY-MM-DD"),
    end: str = Query(None, description="End date in YYYY-MM-DD"),
    db: Session = Depends(get_db)
):
    """
    Return EDA visualizations for the specified stock and date range, 
    now generated using refactored calculation modules.
    """
    
    response_data = {}

    try:
        # 1. Prepare Base Data
        symbol, start, end, stock_df, returns_df, other_data = base_data.prepare_data_for_analysis(
            db, symbol, start, end
        )

        # Check if data preparation failed critically
        if stock_df is None:
             return JSONResponse(
                 status_code=404, 
                 content={"error": f"Could not fetch or prepare basic data for symbol {symbol} between {start} and {end}"}
             )

        # --- Run Analyses ---
        # Store results in the response_data dictionary
        
        # 2. Price Analysis
        price_results = price_analysis.run_all_price_analyses(stock_df, symbol)
        response_data.update(price_results)
        
        # 3. Volume Analysis
        volume_results = volume_analysis.run_all_volume_analyses(stock_df, symbol)
        response_data.update(volume_results)

        # 4. Volatility Analysis (returns updated df and results)
        # Pass the stock_df which may not have index set, functions inside handle indexing
        stock_df_vol, volatility_results = volatility_analysis.run_all_volatility_analyses(stock_df.copy(), symbol) # Pass a copy to avoid modifying original df unintentionally across modules
        response_data.update(volatility_results)
        # Use the potentially updated stock_df (with vol columns) for subsequent analyses if needed
        # For now, correlation and anomaly detection use returns_df or the original stock_df

        # 5. Correlation Analysis
        correlation_results = correlation_analysis.run_all_correlation_analyses(returns_df, symbol, start, end)
        response_data.update(correlation_results)
        
        # 6. Anomaly Detection (To be implemented)
        anomaly_results = anomaly_detection.run_all_anomaly_analyses(stock_df_vol, symbol) # Pass df with vol columns
        response_data.update(anomaly_results)

        # --- New Calculations ---
        # 7. Spectral Analysis 
        # Use the original stock_df as spectral methods often work on price series
        if not stock_df.empty:
            try:
                fft_fig = spectral_analysis.plot_fft(stock_df)
                response_data['fft_plot'] = json.loads(json.dumps(fft_fig, cls=PlotlyJSONEncoder))
            except Exception as e:
                 print(f"Error generating FFT plot: {e}")
                 response_data['fft_plot'] = None # Indicate error or missing plot
            try:
                dwt_fig = spectral_analysis.plot_dwt(stock_df)
                response_data['dwt_plot'] = json.loads(json.dumps(dwt_fig, cls=PlotlyJSONEncoder))
            except Exception as e:
                print(f"Error generating DWT plot: {e}")
                response_data['dwt_plot'] = None
            try:
                cwt_fig = spectral_analysis.plot_cwt(stock_df)
                response_data['cwt_plot'] = json.loads(json.dumps(cwt_fig, cls=PlotlyJSONEncoder))
            except Exception as e:
                print(f"Error generating CWT plot: {e}")
                response_data['cwt_plot'] = None
        else:
             response_data['fft_plot'] = None
             response_data['dwt_plot'] = None
             response_data['cwt_plot'] = None
             
        # 8. Derived Metrics Calculation
        # Calculate metrics, but don't necessarily plot them here.
        # Return the data itself, potentially for display in tables or custom FE plots.
        try:
            derived_metrics_df = derived_metrics.calculate_all_derived_metrics(stock_df)
            # Convert DataFrame to JSON serializable format (e.g., records orientation)
            # Handle potential NaNs/Infs explicitly before serialization if NaNHandlingResponse doesn't cover DataFrames
            derived_metrics_df_serializable = derived_metrics_df.replace([np.inf, -np.inf], None) # Replace Inf with None
            response_data['derived_metrics_data'] = json.loads(derived_metrics_df_serializable.to_json(orient='records', date_format='iso'))
        except Exception as e:
            print(f"Error calculating derived metrics: {e}")
            response_data['derived_metrics_data'] = None # Indicate error
        # ------------------------

        # 9. Tail Risk Analysis
        if not stock_df.empty:
            try:
                tail_risk_results = tail_risk_analysis.calculate_tail_risk_plots(stock_df)
                response_data.update(tail_risk_results) # Add plots/info from tail risk module
            except Exception as e:
                print(f"Error generating tail risk plots: {e}")
                # Add keys with None to indicate failure for this module
                response_data['var_comparison_plot'] = None
                response_data['rolling_var_cvar_plot'] = None
                response_data['evt_return_level_plot'] = None
                response_data['hill_plot'] = None
        else:
             response_data['var_comparison_plot'] = None
             response_data['rolling_var_cvar_plot'] = None
             response_data['evt_return_level_plot'] = None
             response_data['hill_plot'] = None
        # ------------------------

        # Final check if any data was generated
        # Check if *any* value in response_data is not None
        if not any(response_data.values()):
             print(f"Warning: No analysis results generated for {symbol} between {start} and {end}")
             # Return a 200 OK but with an info message if base data was found but no plots generated
             return JSONResponse(
                 status_code=200,
                 content={"info": f"Data found for {symbol}, but no analysis plots could be generated."}
             )

        # Return all collected plot data
        # Note: PlotlyJSONEncoder is handled within the calculation functions now
        # NaNHandlingResponse will take care of any remaining NaNs
        return response_data

    except Exception as e:
        print(f"Error during EDA endpoint execution: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"An unexpected error occurred: {str(e)}",
                "trace": traceback.format_exc() # Provide trace for debugging
            }
        )
