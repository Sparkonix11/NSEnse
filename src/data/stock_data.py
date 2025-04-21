import pandas as pd
import yfinance as yf
from nsetools import Nse
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

def get_stock_data(ticker, start_date, end_date, source='yfinance'):
    """
    Fetch stock data from either yfinance or NSE
    
    Args:
        ticker (str): Stock ticker symbol (with .NS suffix for yfinance)
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        source (str): 'yfinance' or 'nse'
        
    Returns:
        pd.DataFrame: OHLCV data for the requested stock
    """
    if source == 'yfinance':
        # For Indian stocks on yfinance, we need to add .NS suffix if not already present
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            ticker = f"{ticker}.NS"
        
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Basic data cleaning
        if not data.empty:
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            # Create target variable (1 for positive return, 0 for negative)
            data['Target'] = (data['Returns'] > 0).astype(int)
            # Add price change percentage
            data['Change%'] = data['Returns'] * 100
            
            # Add date columns for easier analysis
            data['Year'] = data.index.year
            data['Month'] = data.index.month
            data['Day'] = data.index.day
            data['DayOfWeek'] = data.index.dayofweek
            
        return data
    
    elif source == 'nse':
        # This is a placeholder for NSE data collection
        # The nsetools library doesn't provide historical data
        # We would need to implement web scraping or use another API
        nse = Nse()
        try:
            quote = nse.get_quote(ticker)
            logger.info(f"NSE tools provides only current data, using yfinance instead for {ticker}")
            return get_stock_data(ticker, start_date, end_date, source='yfinance')
        except Exception as e:
            logger.error(f"Error fetching NSE data: {e}")
            logger.info("Falling back to yfinance")
            return get_stock_data(ticker, start_date, end_date, source='yfinance')
    
    else:
        raise ValueError("Source must be either 'yfinance' or 'nse'")


def add_technical_indicators(data):
    """
    Add technical indicators to stock data
    
    Args:
        data (pd.DataFrame): Stock price data
        
    Returns:
        pd.DataFrame: Data with technical indicators
    """
    # Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # Bollinger Bands
    data['BB_middle'] = data['SMA_20']
    data['BB_upper'] = data['SMA_20'] + (data['Close'].rolling(window=20).std() * 2)
    data['BB_lower'] = data['SMA_20'] - (data['Close'].rolling(window=20).std() * 2)
    data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
    
    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Average True Range (ATR)
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    data['ATR'] = true_range.rolling(14).mean()
    
    return data


def get_nifty50_tickers():
    """
    Get a list of Nifty50 stock tickers
    
    Returns:
        list: List of Nifty50 stock tickers with .NS suffix
    """
    # This is a simplified list for demonstration
    # In a real-world scenario, you'd scrape this from NSE website or use an API
    nifty50 = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "HINDUNILVR.NS", "HDFC.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
        "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS"
    ]
    return nifty50