""":mod:`nan_safe_json` - Custom JSON response handling for NaN values."""

import math
import json
from typing import Any
from starlette.responses import JSONResponse

def convert_nan_to_none(obj: Any) -> Any:
    """Recursively converts NaN values in nested structures to None."""
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

class NaNHandlingResponse(JSONResponse):
    """
    Custom JSONResponse that handles NaN values by converting them to None (JSON null).
    """
    def render(self, content: Any) -> bytes:
        """Renders the content, converting NaNs to None before standard JSON encoding."""
        safe_content = convert_nan_to_none(content)
        return json.dumps(
            safe_content,
            ensure_ascii=False,
            allow_nan=False, # Ensure standard JSON compliance
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8") 