"""Backtesting module for evaluating trading strategies"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

class Backtester:
    """Backtester for evaluating model performance"""
    
    def __init__(self, model, data, features, parameters=None):
        """Initialize backtester
        
        Args:
            model: The prediction model to backtest
            data: The market data to use for backtesting
            features: List of feature sets to use
            parameters: Dictionary of backtest parameters
        """
        self.model = model
        self.data = data
        self.features = features
        self.parameters = parameters or {}
        
        # Set default parameters if not provided
        self.threshold = self.parameters.get('threshold', 0.5)
        self.trade_size = self.parameters.get('trade_size', 10000)
        self.commission = self.parameters.get('commission', 0.001)  # 0.1%
        
    def run(self) -> Dict[str, Any]:
        """Run the backtest
        
        Returns:
            Dictionary with backtest results
        """
        # This is a simplified placeholder implementation
        # In a real system, this would include proper feature engineering,
        # trading logic, and performance calculations
        
        # Prepare the data for backtesting
        backtest_data = self._prepare_data()
        
        # Generate trading signals
        signals = self._generate_signals(backtest_data)
        
        # Simulate trades
        trades, equity_curve = self._simulate_trades(backtest_data, signals)
        
        # Return the results
        return {
            "trades": trades,
            "equity_curve": equity_curve.to_dict(orient='records'),
            "data": backtest_data.tail(100).to_dict(orient='records')  # Sample of the data for analysis
        }
    
    def _prepare_data(self) -> pd.DataFrame:
        """Prepare data for backtesting"""
        # Clone the data to avoid modifying the original
        data = self.data.copy()
        
        # Add basic technical indicators if not already present
        if 'close' in data.columns and 'sma_20' not in data.columns:
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Drop any rows with missing values
        data = data.dropna()
        
        return data
    
    def _generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from model predictions"""
        # In a real implementation, we would properly prepare features for the model
        # This is a simplified version that just uses the data as is
        try:
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                # For classification models
                probas = self.model.predict_proba(data)
                signals = (probas[:, 1] > self.threshold).astype(int)  # Long signal if probability > threshold
            else:
                # For regression models
                predictions = self.model.predict(data)
                signals = np.where(predictions > 0, 1, -1)  # Long if positive, short if negative
                
            return pd.Series(signals, index=data.index)
        except Exception as e:
            # Return random signals for demo purposes if the model can't make predictions
            print(f"Error generating signals: {str(e)}. Using random signals for demonstration.")
            return pd.Series(np.random.choice([-1, 0, 1], size=len(data)), index=data.index)
    
    def _simulate_trades(self, data: pd.DataFrame, signals: pd.Series) -> tuple:
        """Simulate trades based on signals"""
        # Create a copy of the data with the signals
        trading_data = data.copy()
        trading_data['signal'] = signals
        
        # Initialize portfolio tracking
        trading_data['position'] = trading_data['signal'].shift(1).fillna(0)
        trading_data['returns'] = trading_data['close'].pct_change().fillna(0)
        trading_data['strategy_returns'] = trading_data['position'] * trading_data['returns']
        
        # Apply commission for position changes
        position_changes = trading_data['position'].diff().abs().fillna(0)
        trading_data['commission'] = position_changes * self.commission * self.trade_size
        trading_data['strategy_returns_net'] = trading_data['strategy_returns'] * self.trade_size - trading_data['commission']
        
        # Calculate cumulative returns
        trading_data['equity_curve'] = (1 + trading_data['strategy_returns']).cumprod()
        trading_data['equity_curve_net'] = 1.0
        for i in range(1, len(trading_data)):
            prev_equity = trading_data.iloc[i-1]['equity_curve_net']
            curr_return = trading_data.iloc[i]['strategy_returns']
            curr_commission = trading_data.iloc[i]['commission'] / self.trade_size  # Convert to percentage
            trading_data.iloc[i, trading_data.columns.get_loc('equity_curve_net')] = prev_equity * (1 + curr_return - curr_commission)
        
        # Extract trades
        trades = []
        current_position = 0
        entry_price = 0
        entry_date = None
        
        for i, row in trading_data.iterrows():
            new_position = row['position']
            
            # Position changed - we have a trade
            if new_position != current_position:
                # Close previous position if we had one
                if current_position != 0:
                    exit_price = row['close']
                    exit_date = i
                    pnl = (exit_price - entry_price) * current_position * self.trade_size
                    pnl_pct = (exit_price / entry_price - 1) * 100 * current_position
                    commission = self.commission * self.trade_size * 2  # Entry and exit
                    net_pnl = pnl - commission
                    
                    trades.append({
                        'entry_date': entry_date.strftime('%Y-%m-%d'),
                        'exit_date': exit_date.strftime('%Y-%m-%d'),
                        'type': 'LONG' if current_position > 0 else 'SHORT',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl,
                        'pnl_pct': pnl_pct,
                        'commission': commission
                    })
                
                # Open new position if not flat
                if new_position != 0:
                    entry_price = row['close']
                    entry_date = i
                
                current_position = new_position
        
        # Close any open position at the end
        if current_position != 0:
            exit_price = trading_data.iloc[-1]['close']
            exit_date = trading_data.index[-1]
            pnl = (exit_price - entry_price) * current_position * self.trade_size
            pnl_pct = (exit_price / entry_price - 1) * 100 * current_position
            commission = self.commission * self.trade_size * 2  # Entry and exit
            net_pnl = pnl - commission
            
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'type': 'LONG' if current_position > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'commission': commission
            })
        
        # Prepare equity curve for return
        equity_curve = trading_data[['equity_curve_net']].copy()
        equity_curve.columns = ['equity']
        equity_curve['date'] = equity_curve.index
        
        return trades, equity_curve