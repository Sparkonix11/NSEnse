import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import shap
import pickle
import torch

# Import project modules
from src.data.stock_data import get_stock_data, add_technical_indicators, get_nifty50_tickers
from src.data.news_data import get_company_news
from src.data.data_processor import create_dataset, prepare_xgboost_data
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.xgb_model import StockXGBoostModel
from src.models.lstm_model import StockLSTMModel

# Page config
st.set_page_config(
    page_title="NSEnse - Stock Movement Predictor",
    page_icon="📈",
    layout="wide"
)

# Helper functions
@st.cache_resource
def load_sentiment_analyzer():
    """Load sentiment analyzer"""
    return SentimentAnalyzer()

@st.cache_resource
def load_xgboost_model():
    """Load XGBoost model"""
    model_path = "models/xgboost_model.pkl"
    if os.path.exists(model_path):
        return StockXGBoostModel.load(model_path)
    else:
        st.warning("XGBoost model not found. Train the model first using main.py")
        return None

@st.cache_resource
def load_lstm_model():
    """Load LSTM model"""
    model_path = "models/lstm_model.pt"
    if os.path.exists(model_path):
        return StockLSTMModel.load(model_path)
    else:
        st.warning("LSTM model not found. Train the model first using main.py")
        return None

@st.cache_data
def get_stock_data_cached(ticker, start_date, end_date):
    """Cached function to get stock data"""
    data = get_stock_data(ticker, start_date, end_date)
    return add_technical_indicators(data)

@st.cache_data
def get_news_data_cached(company, days_back=30):
    """Cached function to get news data"""
    return get_company_news(company, sources=['scrape'], days_back=days_back)

def plot_stock_candlestick(data, ticker):
    """Plot stock candlestick chart"""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add volume bars
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        yaxis='y2',
        marker_color='rgba(0,0,255,0.5)'
    ))
    
    # Set up layout
    fig.update_layout(
        title=f'{ticker} Stock Price',
        yaxis_title='Price (INR)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False,
        height=500
    )
    
    return fig

def plot_technical_indicators(data):
    """Plot technical indicators"""
    fig = go.Figure()
    
    # Add close price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Close',
        line=dict(color='black', width=1)
    ))
    
    # Add SMA lines
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_5'],
        name='SMA 5',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['SMA_20'],
        name='SMA 20',
        line=dict(color='orange', width=1)
    ))
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_upper'],
        name='BB Upper',
        line=dict(color='rgba(0,100,0,0.3)', width=1),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['BB_lower'],
        name='BB Lower',
        line=dict(color='rgba(0,100,0,0.3)', width=1),
        fill='tonexty',
        fillcolor='rgba(0,100,0,0.1)',
        showlegend=False
    ))
    
    # Set up layout
    fig.update_layout(
        title='Technical Indicators',
        yaxis_title='Price',
        height=400
    )
    
    return fig

def plot_rsi(data):
    """Plot RSI indicator"""
    fig = go.Figure()
    
    # Add RSI line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        name='RSI',
        line=dict(color='purple', width=1)
    ))
    
    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    # Set up layout
    fig.update_layout(
        title='Relative Strength Index (RSI)',
        yaxis_title='RSI',
        yaxis=dict(range=[0, 100]),
        height=300
    )
    
    return fig

def plot_sentiment(news_data):
    """Plot sentiment over time"""
    if news_data.empty:
        return None
    
    # Ensure data is sorted by date
    news_data = news_data.sort_values('date')
    
    fig = px.bar(
        news_data,
        x='date',
        y='sentiment_score',
        color='sentiment_label',
        hover_data=['headline'],
        title='News Sentiment Over Time',
        color_discrete_map={
            'positive': 'green',
            'negative': 'red',
            'neutral': 'gray'
        }
    )
    
    fig.update_layout(
        yaxis_title='Sentiment Score (-1 to 1)',
        height=400
    )
    
    return fig

def predict_stock_movement(model_type, data, features, sequence_length=5):
    """
    Predict stock movement using trained models
    
    Args:
        model_type (str): Type of model ('xgboost' or 'lstm')
        data (pd.DataFrame): Data for prediction
        features (list): List of features
        sequence_length (int): Sequence length for time series
        
    Returns:
        tuple: (prediction, probability)
    """
    # Check if data has enough samples
    if len(data) < sequence_length + 1:
        return None, None
    
    # Extract the most recent sequence from the data
    recent_data = data[features].iloc[-sequence_length:].values
    recent_data = np.expand_dims(recent_data, axis=0)  # Add batch dimension
    
    if model_type == 'xgboost':
        model = load_xgboost_model()
        if model is None:
            return None, None
            
        # Reshape for XGBoost
        recent_data_flat = recent_data.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(recent_data_flat)[0]
        probability = model.predict_proba(recent_data_flat)[0, 1]
        
        return "Up" if prediction == 1 else "Down", probability
        
    elif model_type == 'lstm':
        model = load_lstm_model()
        if model is None:
            return None, None
            
        # Make prediction
        prediction = model.predict(recent_data)[0]
        probability = model.predict_proba(recent_data)[0, 1]
        
        return "Up" if prediction == 1 else "Down", probability
    
    else:
        return None, None

def main():
    """Main Streamlit app"""
    # App title
    st.title("NSEnse: Indian Stock Movement Predictor")
    st.markdown("Predict stock movements using news sentiment and historical prices")
    
    # Sidebar
    st.sidebar.header("Input Parameters")
    
    # Stock selection
    tickers = get_nifty50_tickers()
    ticker = st.sidebar.selectbox("Select Stock", tickers)
    
    # Date range selection
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", start_date)
    end_date = st.sidebar.date_input("End Date", end_date)
    
    # Convert dates to string
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Prediction Model", 
        ['xgboost', 'lstm'],
        format_func=lambda x: "XGBoost" if x == 'xgboost' else "LSTM"
    )
    
    # Days back for news
    days_back = st.sidebar.slider(
        "Days to Look Back for News", 
        min_value=7, 
        max_value=90, 
        value=30
    )
    
    # Sequence length
    sequence_length = st.sidebar.slider(
        "Sequence Length for Prediction", 
        min_value=3, 
        max_value=20, 
        value=5
    )
    
    # Get data button
    if st.sidebar.button("Get Data and Predict"):
        # Create loading spinner
        with st.spinner("Fetching data and making predictions..."):
            # Get company name from ticker
            company_name = ticker.split('.')[0] if '.' in ticker else ticker
            
            # Get stock and news data
            stock_data = get_stock_data_cached(ticker, start_date_str, end_date_str)
            news_data = get_news_data_cached(company_name, days_back)
            
            # Check if we have data
            if stock_data.empty:
                st.error(f"No stock data available for {ticker} in the selected date range.")
                return
                
            # Initialize sentiment analyzer
            sentiment_analyzer = load_sentiment_analyzer()
            
            # Create combined dataset for prediction
            with st.spinner("Processing data and calculating sentiment..."):
                if news_data.empty:
                    st.warning(f"No news data found for {company_name}. Using only stock data for prediction.")
                    # Create a dummy news DataFrame with one row to avoid errors
                    news_data = pd.DataFrame({
                        'date': [stock_data.index[-1]],
                        'headline': [f"No news for {company_name}"],
                        'sentiment_score': [0],
                        'sentiment_label': ['neutral']
                    })
                else:
                    # Analyze sentiment if not already in the data
                    if 'sentiment_score' not in news_data.columns:
                        sentiment_results = sentiment_analyzer.analyze(news_data['headline'].tolist())
                        sentiment_df = pd.DataFrame(sentiment_results)
                        news_data['sentiment_score'] = sentiment_df['sentiment_score']
                        news_data['sentiment_label'] = sentiment_df['label']
                
                # Create dataset
                combined_data = create_dataset(stock_data, news_data)
                
                # Define features for prediction
                features = [
                    'sentiment_mean', 'sentiment_std', 'news_count',
                    'SMA_5', 'SMA_20', 'RSI', 'Returns'
                ]
                
                # Make prediction
                prediction, probability = predict_stock_movement(
                    model_type, 
                    combined_data, 
                    features, 
                    sequence_length
                )
            
            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Sentiment Analysis", "Prediction"])
            
            with tab1:
                st.header("Stock Price Analysis")
                
                # Plot stock data
                candlestick_fig = plot_stock_candlestick(stock_data, ticker)
                st.plotly_chart(candlestick_fig, use_container_width=True)
                
                # Plot technical indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    tech_fig = plot_technical_indicators(stock_data)
                    st.plotly_chart(tech_fig, use_container_width=True)
                
                with col2:
                    rsi_fig = plot_rsi(stock_data)
                    st.plotly_chart(rsi_fig, use_container_width=True)
                
                # Display recent stock data
                st.subheader("Recent Stock Data")
                st.dataframe(stock_data.tail().style.format("{:.2f}"))
            
            with tab2:
                st.header("News Sentiment Analysis")
                
                if news_data.empty:
                    st.warning(f"No news data found for {company_name} in the selected date range.")
                else:
                    # Plot sentiment over time
                    sentiment_fig = plot_sentiment(news_data)
                    if sentiment_fig:
                        st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                    # Display news with sentiment
                    st.subheader("Recent News")
                    news_display = news_data.sort_values('date', ascending=False)
                    news_display = news_display[['date', 'headline', 'sentiment_label', 'sentiment_score']]
                    st.dataframe(news_display.head(10).style.format({
                        'sentiment_score': "{:.2f}"
                    }))
            
            with tab3:
                st.header("Price Movement Prediction")
                
                if prediction is None:
                    st.warning("Unable to make prediction. Make sure the model is trained and data is available.")
                else:
                    # Display prediction
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Predicted Direction", 
                            value=prediction,
                            delta="↑" if prediction == "Up" else "↓"
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence", 
                            value=f"{probability:.2f}"
                        )
                    
                    # Display prediction details
                    st.subheader("Prediction Details")
                    st.write(f"Model used: {model_type.upper()}")
                    st.write(f"Prediction based on the last {sequence_length} days of data")
                    st.write(f"Confidence: {probability:.2f}")
                    
                    # Try to display feature importance or SHAP values
                    st.subheader("Model Explanation")
                    
                    if os.path.exists("results/shap_summary.png"):
                        st.image("results/shap_summary.png", caption="Feature Importance (SHAP values)")
                    else:
                        st.info("No explanation available. Train the model with explanations enabled.")
                        
                        # Show simple feature importance instead
                        if model_type == 'xgboost':
                            model = load_xgboost_model()
                            if model:
                                importance_df = model.get_feature_importance()
                                
                                # Create a bar chart of feature importance
                                importance_fig = px.bar(
                                    importance_df.head(10),
                                    x='importance',
                                    y='feature',
                                    orientation='h',
                                    title='Top 10 Feature Importance'
                                )
                                st.plotly_chart(importance_fig, use_container_width=True)

if __name__ == "__main__":
    main()