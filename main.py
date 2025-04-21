import os
import argparse
import logging
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import shap

# Import project modules
from src.data.stock_data import get_stock_data, add_technical_indicators, get_nifty50_tickers
from src.data.news_data import get_company_news
from src.data.data_processor import create_dataset, prepare_xgboost_data, prepare_lstm_data
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.xgb_model import StockXGBoostModel
from src.models.lstm_model import StockLSTMModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nsense.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_dirs():
    """Create necessary directories for the project"""
    dirs = ['data', 'models', 'results', 'logs']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory '{dir_path}' created or already exists")

def fetch_data(ticker, start_date, end_date, days_back=30):
    """
    Fetch stock and news data for the given ticker
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for stock data
        end_date (str): End date for stock data
        days_back (int): Number of days to look back for news
        
    Returns:
        tuple: Stock data and news data
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    # Get company name from ticker (simple approach)
    company_name = ticker.split('.')[0] if '.' in ticker else ticker
    
    # Fetch stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Add technical indicators
    stock_data = add_technical_indicators(stock_data)
    
    # Fetch news data
    news_data = get_company_news(company_name, sources=['scrape'], days_back=days_back)
    
    return stock_data, news_data

def train_xgboost_model(data, features, target='Target', test_size=0.2, sequence_length=5):
    """
    Train XGBoost model
    
    Args:
        data (pd.DataFrame): Combined dataset
        features (list): List of features to use
        target (str): Target column name
        test_size (float): Test set size
        sequence_length (int): Sequence length for time series
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    logger.info("Preparing data for XGBoost model")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_xgboost_data(
        data, features, target, test_size, sequence_length
    )
    
    # Create feature names for time series data
    feature_names = []
    for i in range(sequence_length, 0, -1):
        for feature in features:
            feature_names.append(f"{feature}_t-{i}")
    
    # Train model
    logger.info("Training XGBoost model")
    model = StockXGBoostModel()
    model.fit(X_train, y_train, feature_names=feature_names)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics, (X_test, y_test)

def train_lstm_model(data, features, target='Target', test_size=0.2, sequence_length=5):
    """
    Train LSTM model
    
    Args:
        data (pd.DataFrame): Combined dataset
        features (list): List of features to use
        target (str): Target column name
        test_size (float): Test set size
        sequence_length (int): Sequence length for time series
        
    Returns:
        tuple: Trained model and evaluation metrics
    """
    logger.info("Preparing data for LSTM model")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_lstm_data(
        data, features, target, test_size, sequence_length
    )
    
    # Train model
    logger.info("Training LSTM model")
    input_size = X_train.shape[2]  # Number of features
    model = StockLSTMModel(input_size=input_size)
    model.fit(
        X_train, y_train, 
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32,
        feature_names=features
    )
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    return model, metrics, (X_test, y_test)

def explain_predictions(model, X_test, feature_names):
    """
    Explain model predictions using SHAP
    
    Args:
        model: Trained model
        X_test: Test data
        feature_names: Feature names
        
    Returns:
        shap.Explainer: SHAP explainer
    """
    logger.info("Generating SHAP explanations")
    
    # Create explainer
    explainer = shap.Explainer(model.model)
    
    # Calculate SHAP values
    shap_values = explainer(X_test)
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    plt.savefig('results/shap_summary.png')
    
    return explainer

def main(args):
    """Main function"""
    # Create directories
    create_dirs()
    
    # Fetch data
    stock_data, news_data = fetch_data(
        args.ticker, 
        args.start_date, 
        args.end_date, 
        days_back=args.days_back
    )
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Create combined dataset
    combined_data = create_dataset(stock_data, news_data, sentiment_analyzer)
    
    # Save dataset
    combined_data.to_csv('data/combined_data.csv')
    logger.info("Combined dataset saved to data/combined_data.csv")
    
    # Define features for prediction
    features = [
        # Sentiment features
        'sentiment_mean', 'sentiment_std', 'news_count',
        # Price features
        'SMA_5', 'SMA_20', 'RSI',
        # Return features
        'Returns'
    ]
    
    # Train models based on arguments
    if args.model == 'xgboost' or args.model == 'all':
        # Train XGBoost model
        xgb_model, xgb_metrics, (xgb_X_test, xgb_y_test) = train_xgboost_model(
            combined_data, 
            features, 
            sequence_length=args.sequence_length
        )
        
        # Save model
        xgb_model.save('models/xgboost_model.pkl')
        
        # Explain predictions
        feature_names = []
        for i in range(args.sequence_length, 0, -1):
            for feature in features:
                feature_names.append(f"{feature}_t-{i}")
                
        xgb_explainer = explain_predictions(xgb_model, xgb_X_test, feature_names)
    
    if args.model == 'lstm' or args.model == 'all':
        # Train LSTM model
        lstm_model, lstm_metrics, (lstm_X_test, lstm_y_test) = train_lstm_model(
            combined_data, 
            features, 
            sequence_length=args.sequence_length
        )
        
        # Save model
        lstm_model.save('models/lstm_model.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NSEnse: Stock Movement Prediction')
    
    # Add arguments
    parser.add_argument('--ticker', type=str, default='RELIANCE.NS', help='Stock ticker')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=30, help='Number of days to look back for news')
    parser.add_argument('--model', type=str, choices=['xgboost', 'lstm', 'all'], default='xgboost', help='Model to train')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for time series')
    
    args = parser.parse_args()
    main(args)