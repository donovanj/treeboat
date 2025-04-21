from typing import Dict, Any, List
from pydantic import BaseModel, Field, validator
import numpy as np
from fastapi import HTTPException

class FeatureParameter(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    value: Any
    type: str = Field(..., regex='^(number|string|boolean|array)$')

class FeatureConfig(BaseModel):
    type: str = Field(..., min_length=1, max_length=50)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    name: str = Field(None, min_length=1, max_length=100)
    description: str = Field(None, max_length=500)

    @validator('type')
    def validate_feature_type(cls, v):
        allowed_types = {
            'technical', 'volatility', 'volume', 'market_regime',
            'sector', 'yield_curve', 'seasonality', 'custom'
        }
        if v not in allowed_types:
            raise ValueError(f'Feature type must be one of: {allowed_types}')
        return v

    @validator('parameters')
    def validate_parameters(cls, v):
        # Ensure parameters don't contain any harmful code
        for key, value in v.items():
            if isinstance(value, str):
                dangerous_terms = {'exec', 'eval', 'import', 'open', '__'}
                if any(term in value.lower() for term in dangerous_terms):
                    raise ValueError('Parameters contain potentially harmful code')
        return v

class FeaturePreviewRequest(BaseModel):
    features: List[FeatureConfig]
    data: Dict[str, List[Any]]

    @validator('features')
    def validate_features_length(cls, v):
        if not v:
            raise ValueError('At least one feature must be provided')
        if len(v) > 20:  # Limit number of features per request
            raise ValueError('Maximum of 20 features per request')
        return v

    @validator('data')
    def validate_data(cls, v):
        required_fields = {'date', 'close'}
        if not all(field in v for field in required_fields):
            raise ValueError(f'Data must contain all required fields: {required_fields}')
        
        # Ensure all arrays have the same length
        lengths = {len(arr) for arr in v.values()}
        if len(lengths) > 1:
            raise ValueError('All data arrays must have the same length')
        
        # Ensure reasonable data size
        if lengths and list(lengths)[0] > 10000:
            raise ValueError('Data exceeds maximum allowed length (10000 points)')
        
        return v

class TargetPreviewRequest(BaseModel):
    data: Dict[str, List[Any]]
    target_type: str = Field(..., regex='^(classification|regression|probabilistic|ranking)$')
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @validator('parameters')
    def validate_target_parameters(cls, v):
        # Ensure required parameters are present based on target type
        required_params = {
            'horizon': lambda x: isinstance(x, (int, float)) and 1 <= x <= 252,  # Max 1 year
            'threshold': lambda x: isinstance(x, (int, float)) and -100 <= x <= 100,
            'vol_adjust': lambda x: isinstance(x, bool),
            'n_bins': lambda x: isinstance(x, int) and 2 <= x <= 10
        }
        
        for param, validator_fn in required_params.items():
            if param in v and not validator_fn(v[param]):
                raise ValueError(f'Invalid value for parameter: {param}')
        
        return v

def validate_feature_calculation(data: np.ndarray) -> np.ndarray:
    """Validate calculated feature data"""
    if data is None or data.size == 0:
        raise HTTPException(
            status_code=400,
            detail="Feature calculation produced no data"
        )
    
    if np.all(np.isnan(data)):
        raise HTTPException(
            status_code=400,
            detail="Feature calculation produced all NaN values"
        )
    
    if not np.isfinite(data).all():
        # Replace inf values with NaN
        data = np.nan_to_num(data, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    return data

def validate_formula(code: str) -> bool:
    """Validate custom feature formula"""
    dangerous_terms = {
        'import', 'exec', 'eval', 'open', 'os', 'sys', 'subprocess',
        'read', 'write', 'delete', 'remove', '__'
    }
    
    code_lower = code.lower()
    for term in dangerous_terms:
        if term in code_lower:
            raise HTTPException(
                status_code=400,
                detail=f"Formula contains forbidden term: {term}"
            )
    
    if not 'result =' in code:
        raise HTTPException(
            status_code=400,
            detail="Formula must assign to 'result' variable"
        )
    
    return True