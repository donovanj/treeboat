from typing import List, Optional, Dict, Any
from datetime import date, timedelta
from sqlalchemy.orm import Session
from financial_prediction_system.logging_config import logger
from .loader_factory import DataLoaderFactory
from .cache_factory import CacheFactory
from .data_quality import DataQualityManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial

class DataUpdateManager:
    def __init__(self, db: Session, use_cache: bool = True, enable_quality_checks: bool = True):
        self.db = db
        self.use_cache = use_cache
        self.loader_types = DataLoaderFactory.get_available_loaders()
        self.max_workers = min(len(self.loader_types), 4)  # Limit concurrent workers
        self.quality_manager = DataQualityManager(db) if enable_quality_checks else None
        logger.info(f"Initialized DataUpdateManager with loaders: {self.loader_types}")

    async def update_all_today(self, loader_type: Optional[str] = None, symbol: Optional[str] = None) -> dict:
        """Update all loaders in parallel (default implementation)"""
        logger.info(f"Starting parallel daily update for {'all loaders' if not loader_type else loader_type}")
        results = {}
        
        # Create a thread pool for CPU-bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create tasks for each loader
            tasks = []
            loaders_to_update = [loader_type] if loader_type else self.loader_types
            for loader in loaders_to_update:
                task = asyncio.get_event_loop().run_in_executor(
                    executor,
                    partial(self._update_loader, loader, symbol)
                )
                tasks.append((task, loader))
            
            # Wait for all tasks to complete
            for task, loader in tasks:
                try:
                    results[loader] = await task
                except Exception as e:
                    logger.error(f"Error updating {loader}: {str(e)}", exc_info=True)
                    results[loader] = {"error": str(e)}
        
        return results

    def _update_loader(self, loader_type: str, symbol: Optional[str] = None) -> dict:
        """Internal method to update a single loader"""
        try:
            loader = DataLoaderFactory.get_loader(loader_type, self.db, self.use_cache)
            
            # Register quality manager as observer if enabled
            if self.quality_manager:
                loader.register_quality_observer(self.quality_manager)
                
            records = loader.update_daily_data(symbol=symbol)
            return {
                "records_processed": records,
                "date": date.today().isoformat(),
                "symbol": symbol if symbol else "all"
            }
        except Exception as e:
            logger.error(f"Error in loader {loader_type}: {str(e)}", exc_info=True)
            raise

    async def fill_all_gaps(self, loader_type: Optional[str] = None, symbol: Optional[str] = None) -> dict:
        """Fill data gaps for all loaders in parallel"""
        logger.info(f"Starting parallel gap fill for {'all loaders' if not loader_type else loader_type}")
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            loaders_to_update = [loader_type] if loader_type else self.loader_types
            for loader in loaders_to_update:
                task = asyncio.get_event_loop().run_in_executor(
                    executor,
                    partial(self._fill_gaps_loader, loader, symbol)
                )
                tasks.append((task, loader))
            
            for task, loader in tasks:
                try:
                    results[loader] = await task
                except Exception as e:
                    logger.error(f"Error filling gaps for {loader}: {str(e)}", exc_info=True)
                    results[loader] = {"error": str(e)}
        
        return results

    def _fill_gaps_loader(self, loader_type: str, symbol: Optional[str] = None) -> dict:
        """Internal method to fill gaps for a single loader"""
        try:
            loader = DataLoaderFactory.get_loader(loader_type, self.db, self.use_cache)
            
            # Register quality manager as observer if enabled
            if self.quality_manager:
                loader.register_quality_observer(self.quality_manager)
                
            latest_date = loader.get_latest_date(symbol=symbol)
            
            if not latest_date:
                logger.warning(f"No data found for {loader_type}, loading full history")
                return self._load_bulk_loader(loader_type, symbol=symbol)
            
            start_date = latest_date + timedelta(days=1)
            end_date = date.today() - timedelta(days=1)
            
            if start_date > end_date:
                logger.info(f"No gaps to fill for {loader_type}")
                return {"message": "No gaps to fill", "latest_date": latest_date.isoformat()}
            
            logger.info(f"Filling gaps from {start_date} to {end_date} for {loader_type}")
            records = loader.load_historical_data(start_date, end_date, symbol=symbol)
            return {
                "records_processed": records,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbol": symbol if symbol else "all"
            }
        except Exception as e:
            logger.error(f"Error in gap fill for {loader_type}: {str(e)}", exc_info=True)
            raise

    async def load_all_bulk(self, start_date: Optional[date] = None, end_date: Optional[date] = None, 
                          loader_type: Optional[str] = None, symbol: Optional[str] = None) -> dict:
        """Load bulk data for all loaders in parallel"""
        logger.info(f"Starting parallel bulk load for {'all loaders' if not loader_type else loader_type} from {start_date} to {end_date}")
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = []
            loaders_to_update = [loader_type] if loader_type else self.loader_types
            for loader in loaders_to_update:
                task = asyncio.get_event_loop().run_in_executor(
                    executor,
                    partial(self._load_bulk_loader, loader, start_date, end_date, symbol)
                )
                tasks.append((task, loader))
            
            for task, loader in tasks:
                try:
                    results[loader] = await task
                except Exception as e:
                    logger.error(f"Error in bulk load for {loader}: {str(e)}", exc_info=True)
                    results[loader] = {"error": str(e)}
        
        return results

    def _load_bulk_loader(self, loader_type: str, start_date: Optional[date] = None, 
                         end_date: Optional[date] = None, symbol: Optional[str] = None) -> dict:
        """Internal method to load bulk data for a single loader"""
        try:
            loader = DataLoaderFactory.get_loader(loader_type, self.db, self.use_cache)
            
            # Register quality manager as observer if enabled
            if self.quality_manager:
                loader.register_quality_observer(self.quality_manager)
                
            records = loader.load_historical_data(start_date, end_date, symbol=symbol)
            return {
                "records_processed": records,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "symbol": symbol if symbol else "all"
            }
        except Exception as e:
            logger.error(f"Error in bulk load for {loader_type}: {str(e)}", exc_info=True)
            raise

    async def update_n_days(self, n: int, loader_type: Optional[str] = None, symbol: Optional[str] = None) -> dict:
        """Update data for the last N days in parallel"""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=n-1)
        
        logger.info(f"Updating last {n} days for {'all loaders' if not loader_type else loader_type}")
        return await self.load_all_bulk(start_date, end_date, loader_type=loader_type, symbol=symbol)
        
    async def clear_cache(self) -> dict:
        """Clear all cache data"""
        try:
            if CacheFactory.clear_cache():
                return {"success": True, "message": "Cache cleared successfully"}
            else:
                return {"success": False, "message": "Failed to clear cache"}
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
            
    async def run_quality_checks(self, days: int = 30, loader_type: Optional[str] = None) -> dict:
        """Run quality checks on recent data"""
        if not self.quality_manager:
            return {"error": "Quality manager is not enabled"}
            
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Running quality checks for the last {days} days")
        
        results = {}
        loaders_to_check = [loader_type] if loader_type else self.loader_types
        
        for loader in loaders_to_check:
            try:
                quality_results = self.quality_manager.validate_data_quality(
                    loader, start_date, end_date
                )
                results[loader] = quality_results
            except Exception as e:
                logger.error(f"Error running quality checks for {loader}: {str(e)}", exc_info=True)
                results[loader] = {"error": str(e)}
                
        return results 