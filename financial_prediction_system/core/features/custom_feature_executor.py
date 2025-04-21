import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional
import ast
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class RestrictedVisitor(ast.NodeVisitor):
    """AST visitor to check for forbidden operations in custom formulas"""
    
    def __init__(self):
        self.errors = []
        self.allowed_functions = {
            'min', 'max', 'sum', 'abs', 'round', 'len',
            'any', 'all', 'enumerate', 'zip', 'map', 'filter'
        }
        self.allowed_modules = {'np', 'pd', 'ta'}
    
    def visit_Import(self, node):
        self.errors.append(f"Import statements are not allowed: {ast.dump(node)}")
    
    def visit_ImportFrom(self, node):
        self.errors.append(f"Import statements are not allowed: {ast.dump(node)}")
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.allowed_functions:
                self.errors.append(f"Function '{func_name}' is not allowed")
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                if module_name not in self.allowed_modules:
                    self.errors.append(f"Module '{module_name}' is not allowed")
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id not in self.allowed_modules and hasattr(node, 'attr'):
                if node.attr.startswith('__'):
                    self.errors.append(f"Access to dunder methods is not allowed: {node.attr}")
        self.generic_visit(node)

@contextmanager
def time_limit(seconds):
    """Context manager for timing out long-running calculations"""
    timer = threading.Timer(seconds, lambda: (_ for _ in ()).throw(TimeoutError()))
    timer.start()
    try:
        yield
    finally:
        timer.cancel()

class MemoryLimitedDataFrame(pd.DataFrame):
    """DataFrame subclass with memory usage tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._memory_limit = 1e8  # 100MB limit
    
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            if result.memory_usage().sum() > self._memory_limit:
                raise MemoryError("Memory usage exceeded limit")
        return result

class CustomFeatureExecutor:
    """Executes custom feature formulas in a restricted environment"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def validate_formula(self, code: str) -> Optional[str]:
        """Validate formula for security and correctness"""
        try:
            tree = ast.parse(code)
            visitor = RestrictedVisitor()
            visitor.visit(tree)
            
            if visitor.errors:
                return "\n".join(visitor.errors)
            
            if 'result =' not in code:
                return "Formula must assign output to 'result' variable"
            
            return None
            
        except SyntaxError as e:
            return f"Syntax error in formula: {str(e)}"
        except Exception as e:
            return f"Error validating formula: {str(e)}"
    
    def create_restricted_globals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create restricted globals dictionary for formula execution"""
        df = MemoryLimitedDataFrame(data)
        
        allowed_numpy = {
            'array', 'zeros', 'ones', 'arange', 'linspace',
            'sum', 'mean', 'std', 'min', 'max', 'abs',
            'log', 'exp', 'sqrt', 'square', 'power',
            'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
            'isnan', 'nan_to_num', 'roll'
        }
        
        allowed_talib = {
            'SMA', 'EMA', 'WMA', 'DEMA', 'TEMA', 'TRIMA', 'KAMA',
            'RSI', 'MACD', 'STOCH', 'BBANDS', 'ATR', 'ADX',
            'CCI', 'ROC', 'WILLR', 'MOM', 'TSF', 'HT_TRENDLINE'
        }
        
        restricted_np = {
            name: getattr(np, name)
            for name in allowed_numpy
            if hasattr(np, name)
        }
        
        restricted_ta = {
            name: getattr(talib, name)
            for name in allowed_talib
            if hasattr(talib, name)
        }
        
        return {
            'df': df,
            'np': type('RestrictedNumpy', (), restricted_np),
            'pd': pd,
            'ta': type('RestrictedTALib', (), restricted_ta)
        }
    
    async def execute_formula(
        self,
        code: str,
        data: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Execute a custom feature formula"""
        validation_error = self.validate_formula(code)
        if validation_error:
            raise ValueError(validation_error)
        
        globals_dict = self.create_restricted_globals(data)
        locals_dict = {}
        
        try:
            # Execute in thread pool with timeout
            future = self.executor.submit(
                self._execute_with_timeout,
                code, globals_dict, locals_dict
            )
            result = future.result(timeout=self.timeout)
            
            if result is None:
                raise ValueError("Formula did not produce a result")
            
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            
            # Basic validation of result
            if result.size == 0:
                raise ValueError("Formula produced empty result")
            
            if not np.isfinite(result).all():
                # Replace inf values with NaN
                result = np.nan_to_num(
                    result,
                    nan=np.nan,
                    posinf=np.nan,
                    neginf=np.nan
                )
            
            return result
            
        except TimeoutError:
            raise ValueError(
                f"Formula execution timed out after {self.timeout} seconds"
            )
        except Exception as e:
            logger.error(
                f"Error executing formula: {str(e)}\n{traceback.format_exc()}"
            )
            raise ValueError(f"Error executing formula: {str(e)}")
    
    def _execute_with_timeout(
        self,
        code: str,
        globals_dict: Dict[str, Any],
        locals_dict: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Execute code with timeout"""
        with time_limit(self.timeout):
            exec(code, globals_dict, locals_dict)
            return locals_dict.get('result')