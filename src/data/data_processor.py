import pandas as pd
import numpy as np
import logging
from src.models.sentiment_analyzer import SentimentAnalyzer

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
        news_count=('headline', 'count')
    ).reset_index()
    
    # Fill missing values for days with no news
    daily_sentiment['sentiment_std'].fillna(0, inplace=True)
    
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
    sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max', 'news_count']
    merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(method='ffill')
    
    # If there are still missing values (at the beginning), fill with zeros
    merged_data[sentiment_cols] = merged_data[sentiment_cols].fillna(0)
    
    # Clean up the dataframe
    if 'date' in merged_data.columns:
        merged_data.drop('date', axis=1, inplace=True)
    
    return merged_data


def prepare_training_data(data, features, target='Target', test_size=0.2, sequence_length=5):
    """
    Prepare data for time series prediction
    
    Args:
        data (pd.DataFrame): Combined dataset with features
        features (list): List of feature column names
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Number of previous days to use as input
        
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
    
    # Create sequences for time series prediction
    X = []
    y = []
    
    logger.info(f"Creating sequences with length {sequence_length}")
    
    for i in range(len(data) - sequence_length):
        X.append(data[features].iloc[i:i+sequence_length].values)
        y.append(data[target].iloc[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def prepare_lstm_data(data, features, target='Target', test_size=0.2, sequence_length=5):
    """
    Prepare data specifically for LSTM model (3D arrays)
    
    Args:
        data (pd.DataFrame): Combined dataset with features
        features (list): List of feature column names
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Number of previous days to use as input
        
    Returns:
        tuple: X_train, X_test, y_train, y_test formatted for LSTM
    """
    # Get training data in basic format
    X_train, X_test, y_train, y_test = prepare_training_data(
        data, features, target, test_size, sequence_length
    )
    
    # For LSTM, keep the 3D shape (samples, time steps, features)
    # No need to reshape as prepare_training_data already returns 3D arrays
    
    return X_train, X_test, y_train, y_test


def prepare_xgboost_data(data, features, target='Target', test_size=0.2, sequence_length=5):
    """
    Prepare data specifically for XGBoost model (flattened 2D arrays)
    
    Args:
        data (pd.DataFrame): Combined dataset with features
        features (list): List of feature column names
        target (str): Target column name
        test_size (float): Proportion of data to use for testing
        sequence_length (int): Number of previous days to use as input
        
    Returns:
        tuple: X_train, X_test, y_train, y_test formatted for XGBoost
    """
    # Get training data in basic format
    X_train, X_test, y_train, y_test = prepare_training_data(
        data, features, target, test_size, sequence_length
    )
    
    # For XGBoost, flatten the sequence data to 2D arrays
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    return X_train_flat, X_test_flat, y_train, y_test