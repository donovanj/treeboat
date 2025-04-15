from typing import Dict, List, Optional
from datetime import date, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
from logging_config import logger

class DataQualityManager:
    def __init__(self, db: Session):
        self.db = db
        self.logger = logger

    def on_data_loaded(self, loader_type: str, data: List[Dict], start_date: Optional[date], end_date: Optional[date], symbol: Optional[str]) -> Dict:
        """
        Observer method called when data is loaded.
        Implements DataQualityObserver protocol.
        """
        if not start_date or not end_date:
            self.logger.warning("Cannot run quality checks: missing date range")
            return {"error": "Missing date range for quality checks"}
            
        try:
            # Run validation only on the loaded data's date range
            quality_results = self.validate_data_quality(loader_type, start_date, end_date)
            
            # Add summary statistics about the loaded data
            quality_results["data_summary"] = self._summarize_data(data, loader_type)
            
            return quality_results
        except Exception as e:
            self.logger.error(f"Error in quality observer: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def validate_data_quality(self, loader_type: str, start_date: date, end_date: date) -> dict:
        """Check for data anomalies, missing values, and consistency"""
        checks = {
            "missing_dates": self._check_missing_dates,
            "outliers": self._check_outliers,
            "consistency": self._check_consistency
        }
        results = {}
        for check_name, check_func in checks.items():
            try:
                results[check_name] = check_func(loader_type, start_date, end_date)
            except Exception as e:
                self.logger.error(f"Error in {check_name} check: {str(e)}", exc_info=True)
                results[check_name] = {"error": str(e)}
        return results
    
    def _summarize_data(self, data: List[Dict], loader_type: str) -> Dict:
        """Generate summary statistics for loaded data"""
        if not data:
            return {"count": 0, "message": "No data to summarize"}
            
        try:
            # Extract dates
            dates = [record.get('date') for record in data if 'date' in record]
            date_range = f"{min(dates)} to {max(dates)}" if dates else "N/A"
            
            # Get key value fields based on loader type
            if loader_type in ["stock", "index"]:
                price_field = "close"
                volume_field = "volume"
                
                prices = [float(record.get(price_field)) for record in data 
                         if price_field in record and record.get(price_field) is not None]
                
                volumes = [int(record.get(volume_field)) for record in data 
                          if volume_field in record and record.get(volume_field) is not None]
                
                price_stats = {
                    "min": min(prices) if prices else None,
                    "max": max(prices) if prices else None,
                    "avg": sum(prices) / len(prices) if prices else None,
                    "count": len(prices)
                }
                
                volume_stats = {
                    "min": min(volumes) if volumes else None,
                    "max": max(volumes) if volumes else None,
                    "avg": sum(volumes) / len(volumes) if volumes else None,
                    "count": len(volumes)
                }
                
                return {
                    "record_count": len(data),
                    "date_range": date_range,
                    "price_statistics": price_stats,
                    "volume_statistics": volume_stats
                }
            elif loader_type == "treasury":
                # Treasury data has different fields
                yield_fields = ["mo1", "mo2", "mo3", "mo6", "yr1", "yr2", "yr5", "yr10", "yr30"]
                stats = {}
                
                for field in yield_fields:
                    values = [float(record.get(field)) for record in data 
                             if field in record and record.get(field) is not None]
                    
                    if values:
                        stats[field] = {
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values),
                            "count": len(values)
                        }
                
                return {
                    "record_count": len(data),
                    "date_range": date_range,
                    "yield_statistics": stats
                }
            else:
                return {
                    "record_count": len(data),
                    "date_range": date_range,
                    "message": f"No specific summary available for {loader_type}"
                }
        except Exception as e:
            self.logger.error(f"Error summarizing data: {str(e)}")
            return {"error": f"Error summarizing data: {str(e)}"}

    def _check_missing_dates(self, loader_type: str, start_date: date, end_date: date) -> dict:
        """Check for missing dates in the data range"""
        from database.models_and_schemas.models import StockPrice, TreasuryYield, IndexPrice
        
        model_map = {
            "stock": StockPrice,
            "treasury": TreasuryYield,
            "index": IndexPrice
        }
        
        if loader_type not in model_map:
            raise ValueError(f"Invalid loader type: {loader_type}")
            
        model = model_map[loader_type]
        existing_dates = set(
            date for date, in self.db.query(model.date)
            .filter(model.date.between(start_date, end_date))
            .all()
        )
        
        all_dates = set(
            start_date + timedelta(days=x)
            for x in range((end_date - start_date).days + 1)
        )
        
        missing_dates = sorted(all_dates - existing_dates)
        return {
            "total_dates": len(all_dates),
            "missing_dates": len(missing_dates),
            "missing_date_list": [d.isoformat() for d in missing_dates]
        }

    def _check_outliers(self, loader_type: str, start_date: date, end_date: date) -> dict:
        """Check for statistical outliers in the data"""
        from database.models_and_schemas.models import StockPrice, TreasuryYield, IndexPrice
        
        model_map = {
            "stock": StockPrice,
            "treasury": TreasuryYield,
            "index": IndexPrice
        }
        
        if loader_type not in model_map:
            raise ValueError(f"Invalid loader type: {loader_type}")
            
        model = model_map[loader_type]
        price_column = "close" if loader_type in ["stock", "index"] else "yr10"  # Use 10-yr yield for Treasury
        
        # Get all values in the date range
        values = [
            getattr(record, price_column)
            for record in self.db.query(model)
            .filter(model.date.between(start_date, end_date))
            .all()
        ]
        
        if not values:
            return {"error": "No data found in the specified range"}
            
        # Calculate statistical measures
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        # Define outliers as values beyond 3 standard deviations
        outliers = values_array[np.abs(values_array - mean) > 3 * std]
        
        return {
            "mean": float(mean),
            "std_dev": float(std),
            "outlier_count": len(outliers),
            "outlier_percentage": float(len(outliers) / len(values) * 100)
        }

    def _check_consistency(self, loader_type: str, start_date: date, end_date: date) -> dict:
        """Check for data consistency and integrity"""
        from database.models_and_schemas.models import StockPrice, TreasuryYield, IndexPrice
        
        model_map = {
            "stock": StockPrice,
            "treasury": TreasuryYield,
            "index": IndexPrice
        }
        
        if loader_type not in model_map:
            raise ValueError(f"Invalid loader type: {loader_type}")
            
        model = model_map[loader_type]
        
        # Check for duplicate dates
        duplicate_dates = self.db.query(model.date)\
            .filter(model.date.between(start_date, end_date))\
            .group_by(model.date)\
            .having(func.count() > 1)\
            .all()
            
        # Check for null values
        null_counts = {}
        for column in model.__table__.columns:
            null_count = self.db.query(func.count())\
                .filter(model.date.between(start_date, end_date))\
                .filter(getattr(model, column.name) == None)\
                .scalar()
            null_counts[column.name] = null_count
            
        return {
            "duplicate_dates": [d[0].isoformat() for d in duplicate_dates],
            "null_values": null_counts
        } 