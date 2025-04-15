"""Base schemas for financial prediction API"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class BaseResponse(BaseModel):
    """Base response model for all API endpoints"""
    success: bool = Field(True, description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Optional message")
    
class ErrorResponse(BaseResponse):
    """Error response model"""
    success: bool = Field(False, description="Request was not successful")
    error: str = Field(..., description="Error message")
    error_code: Optional[int] = Field(None, description="Error code")
    
class PaginatedResponse(BaseResponse):
    """Base paginated response model"""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
