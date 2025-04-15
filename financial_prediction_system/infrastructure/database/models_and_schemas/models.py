"""
Database models for the financial prediction system.
"""
from sqlalchemy import Column, String, Date, Numeric, BigInteger, Boolean, DateTime, func, Float
from sqlalchemy.ext.declarative import declarative_base
from financial_prediction_system.infrastructure.database.connection import Base

class Stock(Base):
    """SQLAlchemy model for stocks"""
    __tablename__ = "stocks"
    
    symbol = Column(String(20), primary_key=True, index=True)
    company_name = Column(String(255), nullable=True)
    quotetype = Column(String(50), nullable=True)
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    last_updated = Column(DateTime, nullable=True, default=func.now())
    is_active = Column(Boolean, nullable=True, default=True)

class StockPrice(Base):
    """SQLAlchemy model for stock prices"""
    __tablename__ = "stock_prices"
    
    symbol = Column(String(20), primary_key=True, index=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

# Add additional models for other tables like index prices, etc.
class SPXPrice(Base):
    """SQLAlchemy model for S&P 500 prices"""
    __tablename__ = "spx_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class NDXPrice(Base):
    """SQLAlchemy model for NASDAQ-100 prices"""
    __tablename__ = "ndx_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class DJIPrice(Base):
    """SQLAlchemy model for Dow Jones prices"""
    __tablename__ = "dji_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class RUTPrice(Base):
    """SQLAlchemy model for Russell 2000 prices"""
    __tablename__ = "rut_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class VIXPrice(Base):
    """SQLAlchemy model for VIX prices"""
    __tablename__ = "vix_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class SOXPrice(Base):
    """SQLAlchemy model for PHLX Semiconductor Sector prices"""
    __tablename__ = "sox_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class OSXPrice(Base):
    """SQLAlchemy model for PHLX Oil Service Sector prices"""
    __tablename__ = "osx_prices"
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)

class TreasuryYield(Base):
    """SQLAlchemy model for Treasury yields"""
    __tablename__ = "treasury_yields"
    
    date = Column(Date, primary_key=True, index=True)
    mo1 = Column(Float, nullable=True)
    mo2 = Column(Float, nullable=True)
    mo3 = Column(Float, nullable=True)
    mo6 = Column(Float, nullable=True)
    yr1 = Column(Float, nullable=True)
    yr2 = Column(Float, nullable=True)
    yr5 = Column(Float, nullable=True)
    yr10 = Column(Float, nullable=True)
    yr30 = Column(Float, nullable=True)

# For data_quality.py compatibility
class IndexPrice(Base):
    """Composite model for all index prices (not an actual table)"""
    __abstract__ = True
    
    symbol = Column(String(20), primary_key=True)
    date = Column(Date, primary_key=True, index=True)
    open = Column(Numeric(10, 2), nullable=False)
    high = Column(Numeric(10, 2), nullable=False)
    low = Column(Numeric(10, 2), nullable=False)
    close = Column(Numeric(10, 2), nullable=False)
    volume = Column(BigInteger, nullable=False)