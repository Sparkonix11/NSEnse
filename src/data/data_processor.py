import pandas as pd
import numpy as np
import logging
from src.models.sentiment_analyzer import SentimentAnalyzer
import talib
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.stats import zscore

logger = logging.getLogger(__name__)

def create_dataset(stock_data, news_data, sentiment_analyzer=None):
    """
    Create a dataset combining stock data and news sentiment
    
    Args:
        stock_data (pd.DataFrame): Historical stock price data
        news_data (pd.DataFrame): News headlines with dates
        sentiment_analyzer (SentimentAnalyzer, optional): Initialized sentiment analyzer object
        
    Returns:
        pd.DataFrame: Combined dataset with stock features and sentiment scores
    """
    # Convert news dates to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(news_data['date']):
        news_data['date'] = pd.to_datetime(news_data['date'])
    
    # If sentiment scores not already in news_data and sentiment_analyzer is provided, compute them
    if 'sentiment_score' not in news_data.columns and sentiment_analyzer is not None:
        logger.info("Computing sentiment scores for news headlines")
        sentiment_results = sentiment_analyzer.analyze(news_data['headline'].tolist())
        sentiment_df = pd.DataFrame(sentiment_results)
        news_data['sentiment_score'] = sentiment_df['sentiment_score']
        news_data['sentiment_label'] = sentiment_df['label']
    elif 'sentiment_score' not in news_data.columns:
        logger.error("No sentiment scores in news data and no sentiment_analyzer provided")
        raise ValueError("News data must contain sentiment scores or a sentiment_analyzer must be provided")
    
    # Group news by date and aggregate sentiment
    daily_sentiment = news_data.groupby('date').agg(
        sentiment_mean=('sentiment_score', 'mean'),
        sentiment_std=('sentiment_score', 'std'),
        sentiment_min=('sentiment_score', 'min'),
        sentiment_max=('sentiment_score', 'max'),
        sentiment_range=lambda x: x.max() - x.min(),  # Range of sentiment in a day
        news_count=('headline', 'count'),
        positive_news_ratio=lambda x: sum(x == 'positive') / len(x) if len(x) > 0 else 0,  # Ratio of positive news
        negative_news_ratio=lambda x: sum(x == 'negative') / len(x) if len(x) > 0 else 0   # Ratio of negative news
    ).reset_index()
    
    # Fill missing values for days with no news
    daily_sentiment['sentiment_std'].fillna(0, inplace=True)
    daily_sentiment['sentiment_range'].fillna(0, inplace=True)
    
    # Convert stock_data index to datetime if not already
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)
    
    # Merge stock data with daily sentiment
    merged_data = stock_data.reset_index().merge(
        daily_sentiment,
        left_on='Date',
        right_on='date',
        how='left'
    ).set_index('Date')
    
    # Fill missing sentiment values using forward fill (assume sentiment persists)
    sentiment_cols = [
        'sentiment_mean', 'sentiment_std', 'sentiment_min', 
        'sentiment_max', 'sentiment_range', 'news_count',
        'positive_news_ratio', 'negative_news_ratio'
    ]
    merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(method='ffill')
    
    # If there are still missing values (at the beginning), fill with zeros
    merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(0)
    
    # Create additional features for better prediction
    
    # 1. Price-based features
    merged_data['price_momentum'] = merged_data['Close'].pct_change(periods=5)
    merged_data['price_acceleration'] = merged_data['price_momentum'].diff()
    
    # 2. Volatility features
    merged_data['volatility_5d'] = merged_data['Close'].pct_change().rolling(window=5).std()
    merged_data['volatility_10d'] = merged_data['Close'].pct_change().rolling(window=10).std()
    merged_data['volatility_ratio'] = merged_data['volatility_5d'] / merged_data['volatility_10d']
    
    # 3. Price relative to moving averages
    merged_data['price_sma5_ratio'] = merged_data['Close'] / merged_data['SMA_5']
    merged_data['price_sma20_ratio'] = merged_data['Close'] / merged_data['SMA_20']
    merged_data['sma_ratio'] = merged_data['SMA_5'] / merged_data['SMA_20']
    
    # 4. Advanced technical indicators
    try:
        merged_data['OBV'] = talib.OBV(merged_data['Close'].values, merged_data['Volume'].values)
    except:
        logger.warning("TALib not available for OBV, using manual calculation")
        # Simple OBV calculation
        obv = [0]
        for i in range(1, len(merged_data)):
            if merged_data['Close'].iloc[i] > merged_data['Close'].iloc[i-1]:
                obv.append(obv[-1] + merged_data['Volume'].iloc[i])
            elif merged_data['Close'].iloc[i] < merged_data['Close'].iloc[i-1]:
                obv.append(obv[-1] - merged_data['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        merged_data['OBV'] = obv
    
    # OBV momentum
    merged_data['OBV_momentum'] = merged_data['OBV'].pct_change(5)
    
    # 5. Interaction features between sentiment and price
    merged_data['sentiment_price_momentum'] = merged_data['sentiment_mean'] * merged_data['price_momentum']
    merged_data['sentiment_volume_ratio'] = merged_data['sentiment_mean'] * (merged_data['Volume'] / merged_data['Volume'].rolling(window=20).mean())
    
    # 6. Sentiment momentum and trend
    merged_data['sentiment_momentum'] = merged_data['sentiment_mean'].diff(periods=3)
    merged_data['sentiment_trend'] = merged_data['sentiment_mean'].rolling(window=5).mean()
    
    # 7. Volatility-adjusted returns
    merged_data['volatility_adjusted_return'] = merged_data['Returns'] / merged_data['volatility_5d']
    
    # 8. Custom momentum signals
    merged_data['momentum_signal'] = np.where(
        (merged_data['SMA_5'] > merged_data['SMA_20']) & 
        (merged_data['Close'] > merged_data['SMA_5']), 
        1, -1
    )
    
    # Clean up the dataframe
    if 'date' in merged_data.columns:
        merged_data.drop('date', axis=1, inplace=True)
    
    # Replace inf and -inf with NaNs, then fill NaNs with 0
    merged_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_data.fillna(0, inplace=True)
    
    return merged_data

def add_advanced_features(data):
    """
    Add advanced features to improve model accuracy
    
    Args:
        data (pd.DataFrame): Dataset with basic features
        
    Returns:
        pd.DataFrame: Dataset with additional features
    """
    # Copy to avoid modifying the original dataframe
    df = data.copy()
    
    # 1. Price gap features
    df['gap_up'] = (df['Open'] > df['Close'].shift(1)).astype(int)
    df['gap_down'] = (df['Open'] < df['Close'].shift(1)).astype(int)
    df['gap_magnitude'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    
    # 2. Rolling statistics for recent returns
    for window in [3, 5, 10]:
        # Mean return
        df[f'return_mean_{window}d'] = df['Returns'].rolling(window=window).mean()
        # Return volatility
        df[f'return_volatility_{window}d'] = df['Returns'].rolling(window=window).std()
        # Return skewness (measure of asymmetry)
        df[f'return_skew_{window}d'] = df['Returns'].rolling(window=window).skew()
    
    # 3. Bollinger Band features
    df['bb_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['bb_squeeze'] = (df['BB_upper'] - df['BB_lower']) / df['SMA_20']
    
    # 4. MACD features (if available)
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        df['macd_diff'] = df['MACD'] - df['MACD_signal']
        df['macd_cross'] = ((df['macd_diff'] > 0) & (df['macd_diff'].shift() < 0)).astype(int) - \
                        ((df['macd_diff'] < 0) & (df['macd_diff'].shift() > 0)).astype(int)
    
    # 5. RSI features
    if 'RSI' in df.columns:
        df['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        df['rsi_oversold'] = (df['RSI'] < 30).astype(int)
        df['rsi_cross_50_up'] = ((df['RSI'] > 50) & (df['RSI'].shift() < 50)).astype(int)
        df['rsi_cross_50_down'] = ((df['RSI'] < 50) & (df['RSI'].shift() > 50)).astype(int)
    
    # 6. Volume features
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['volume_change'] = df['Volume'].pct_change()
    
    # 7. Day of week features (market often behaves differently on different days)
    df['day_of_week'] = df.index.dayofweek
    
    # 8. Month features (seasonal effects)
    df['month'] = df.index.month
    
    # 9. Trend strength indicators
    df['adx'] = calculate_adx(df)
    
    # 10. Candle pattern features
    df['doji'] = ((abs(df['Close'] - df['Open']) / (df['High'] - df['Low'])) < 0.1).astype(int)
    df['hammer'] = (((df['High'] - df['Low']) > 3 * (df['Open'] - df['Close'])) & 
                   (df['Close'] > df['Open']) & 
                   ((df['Close'] - df['Low']) / (.001 + df['High'] - df['Low']) > 0.6)).astype(int)
    
    # 11. Sentiment to price change correlation (rolling)
    if 'sentiment_mean' in df.columns:
        df['sentiment_price_corr'] = df['sentiment_mean'].rolling(window=10).corr(df['Returns'])
    
    # Replace inf and -inf with NaNs, then fill NaNs with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def calculate_adx(data, period=14):
    """Calculate Average Directional Index (ADX)"""
    # Simple implementation of ADX calculation
    # In a real implementation, you might want to use TALib
    
    df = data.copy()
    df['tr1'] = abs(df['High'] - df['Low'])
    df['tr2'] = abs(df['High'] - df['Close'].shift(1))
    df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    df['up_move'] = df['High'] - df['High'].shift(1)
    df['down_move'] = df['Low'].shift(1) - df['Low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['plus_di'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['atr'])
    df['minus_di'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['atr'])
    
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(window=period).mean()
    
    return df['adx']

def normalize_features(data, method='standard'):
    """
    Normalize features to improve model performance
    
    Args:
        data (pd.DataFrame): Dataset with features
        method (str): Normalization method ('standard', 'robust', 'minmax')
        
    Returns:
        pd.DataFrame: Dataset with normalized features
    """
    df = data.copy()
    
    # Columns to exclude from normalization
    exclude_cols = ['Target']  # Add other columns that shouldn't be normalized
    
    # Columns to normalize
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if method == 'standard':
        # Standardize to zero mean and unit variance
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    elif method == 'robust':
        # Use robust scaler which is less sensitive to outliers
        scaler = RobustScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    elif method == 'minmax':
        # Min-max scaling to [0, 1]
        df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())
    elif method == 'zscore':
        # Z-score normalization
        for col in feature_cols:
            df[col] = zscore(df[col], nan_policy='omit')
            df[col].fillna(0, inplace=True)  # Replace NaNs after zscore
    
    return df

def remove_outliers(data, cols=None, threshold=3.0):
    """
    Remove or cap outliers in the dataset
    
    Args:
        data (pd.DataFrame): Dataset
        cols (list): Columns to process
        threshold (float): Z-score threshold for outliers
        
    Returns:
        pd.DataFrame: Dataset with outliers handled
    """
    df = data.copy()
    
    if cols is None:
        # Exclude target and categorical variables
        exclude = ['Target', 'day_of_week', 'month']
        cols = [col for col in df.columns if col not in exclude]
    
    # Cap outliers (winsorization)
    for col in cols:
        if df[col].dtype in ['float64', 'int64']:
            mean = df[col].mean()
            std = df[col].std()
            
            # Cap values outside the threshold
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
            # Replace outliers with bounds
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    return df

def prepare_training_data(data, features, target='Target', test_size=0.2, sequence_length=5,
                         normalize=True, outlier_removal=False, add_features=True):
    """
    Prepare data for time series prediction with additional preprocessing options
    
    Args:
        data (pd.DataFrame): Combined dataset with features
        features (list): List of feature column names
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Number of previous days to use as input
        normalize (bool): Whether to normalize features
        outlier_removal (bool): Whether to handle outliers
        add_features (bool): Whether to add advanced features
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Check if all features exist in the dataframe
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise ValueError(f"Features not found in dataframe: {missing_features}")
    
    # Check if target exists
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe")
    
    # Process the data
    processed_data = data.copy()
    
    # Add advanced features if requested
    if add_features:
        processed_data = add_advanced_features(processed_data)
        features = list(set(features + [col for col in processed_data.columns if col not in data.columns]))
    
    # Handle outliers if requested
    if outlier_removal:
        processed_data = remove_outliers(processed_data, cols=features)
    
    # Normalize features if requested
    if normalize:
        processed_data = normalize_features(processed_data, method='robust')
    
    # Create sequences for time series prediction
    X = []
    y = []
    
    logger.info(f"Creating sequences with length {sequence_length}")
    
    for i in range(len(processed_data) - sequence_length):
        X.append(processed_data[features].iloc[i:i+sequence_length].values)
        y.append(processed_data[target].iloc[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets (time-series aware split)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def prepare_lstm_data(data, features, target='Target', test_size=0.2, sequence_length=5,
                     normalize=True, outlier_removal=False):
    """
    Prepare data specifically for LSTM model (3D arrays)
    
    Args:
        data (pd.DataFrame): Combined dataset with features
        features (list): List of feature column names
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Number of previous days to use as input
        normalize (bool): Whether to normalize features
        outlier_removal (bool): Whether to handle outliers
        
    Returns:
        tuple: X_train, X_test, y_train, y_test formatted for LSTM
    """
    # Get training data with preprocessing options
    X_train, X_test, y_train, y_test = prepare_training_data(
        data, features, target, test_size, sequence_length,
        normalize=normalize, outlier_removal=outlier_removal
    )
    
    # For LSTM, keep the 3D shape (samples, time steps, features)
    # No need to reshape as prepare_training_data already returns 3D arrays
    
    return X_train, X_test, y_train, y_test


def prepare_xgboost_data(data, features, target='Target', test_size=0.2, sequence_length=5,
                        normalize=True, outlier_removal=False):
    """
    Prepare data specifically for XGBoost model (flattened 2D arrays)
    
    Args:
        data (pd.DataFrame): Combined dataset with features
        features (list): List of feature column names
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Number of previous days to use as input
        normalize (bool): Whether to normalize features
        outlier_removal (bool): Whether to handle outliers
        
    Returns:
        tuple: X_train, X_test, y_train, y_test formatted for XGBoost
    """
    # Get training data in basic format with preprocessing options
    X_train, X_test, y_train, y_test = prepare_training_data(
        data, features, target, test_size, sequence_length,
        normalize=normalize, outlier_removal=outlier_removal,
        add_features=True  # Add advanced features for XGBoost
    )
    
    # For XGBoost, flatten the sequence data to 2D arrays
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    return X_train_flat, X_test_flat, y_train, y_test