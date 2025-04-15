#!/usr/bin/env python3
"""
Script to run the daily inference pipeline manually

This script can be used to:
1. Update market data
2. Generate predictions for a specified model and symbols
3. Generate trading signals

Usage:
    python run_daily_inference.py --model-id 1 --symbols AAPL,MSFT,GOOG --time 16:30

Options:
    --model-id: ID of the model to use for predictions
    --symbols: Comma-separated list of symbols to predict
    --update-data: Whether to update market data first (default: True)
    --time: Time to run at in HH:MM format (default: now)
    --schedule: Whether to schedule this to run daily
"""

import sys
import os
import argparse
import asyncio
from datetime import datetime
from typing import List

# Add parent directory to path to make imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy.orm import Session
from financial_prediction_system.infrastructure.database.connection import engine, SessionLocal
from financial_prediction_system.api.routes.daily_inference import (
    update_all_data, generate_daily_predictions, schedule_daily_pipeline
)
from financial_prediction_system.logging_config import logger

async def run_pipeline(
    model_id: int, 
    symbols: List[str], 
    update_data: bool = True,
    schedule: bool = False,
    time: str = None
):
    """Run the daily inference pipeline"""
    # Create DB session
    db = SessionLocal()
    
    try:
        logger.info(f"Starting daily inference pipeline for model {model_id}")
        logger.info(f"Symbols: {', '.join(symbols)}")
        
        # Update data if requested
        if update_data:
            logger.info("Updating market data...")
            update_results = await update_all_data(db)
            logger.info(f"Data update completed: {update_results}")
        
        # Generate predictions
        logger.info(f"Generating predictions for {len(symbols)} symbols...")
        prediction_results = await generate_daily_predictions(model_id, symbols, db)
        
        if "error" in prediction_results:
            logger.error(f"Error in prediction generation: {prediction_results['error']}")
            return False
            
        logger.info(f"Predictions saved to: {prediction_results.get('saved_to')}")
        
        # Schedule for daily execution if requested
        if schedule and time:
            logger.info(f"Scheduling daily execution at {time}...")
            await schedule_daily_pipeline(model_id, symbols, time)
            logger.info(f"Pipeline scheduled to run daily at {time}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error running pipeline: {str(e)}", exc_info=True)
        return False
    finally:
        db.close()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run daily inference pipeline")
    
    parser.add_argument(
        "--model-id", 
        type=int, 
        required=True,
        help="ID of the model to use for predictions"
    )
    
    parser.add_argument(
        "--symbols", 
        type=str, 
        required=True,
        help="Comma-separated list of symbols to predict"
    )
    
    parser.add_argument(
        "--update-data", 
        type=bool, 
        default=True,
        help="Whether to update market data first"
    )
    
    parser.add_argument(
        "--schedule", 
        action="store_true",
        help="Schedule this pipeline to run daily"
    )
    
    parser.add_argument(
        "--time", 
        type=str, 
        default="16:30",
        help="Time to run at in HH:MM format (for scheduling)"
    )
    
    return parser.parse_args()

async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Convert comma-separated symbols to list
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    # Run the pipeline
    success = await run_pipeline(
        model_id=args.model_id,
        symbols=symbols,
        update_data=args.update_data,
        schedule=args.schedule,
        time=args.time
    )
    
    if success:
        logger.info("Pipeline completed successfully")
    else:
        logger.error("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 