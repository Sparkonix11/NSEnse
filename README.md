# NSEnse: Indian Stock Movement Predictor

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

NSEnse is a deep learning-based stock movement prediction tool specifically designed for the Indian stock market. The project combines technical indicators with news sentiment analysis to predict price movements of Nifty 50 stocks.

## 🚀 Features

- **Dual Model Architecture**: Uses both XGBoost and LSTM models for prediction
- **News Sentiment Analysis**: Analyzes financial news to extract sentiment scores
- **Technical Indicators**: Incorporates SMA, RSI, Bollinger Bands, and more
- **Interactive Dashboard**: Streamlit-based visualization for easy analysis
- **Explainable AI**: Uses SHAP values to explain model predictions
- **Market Insights**: Provides detailed stock analysis with visualizations

## 📊 Demo

![NSEnse Dashboard](path/to/dashboard_screenshot.png)

## 🛠 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/nsense.git
cd nsense
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Training Models

To train the models, run:

```bash
python main.py --ticker RELIANCE.NS --start-date 2020-01-01 --model all
```

Options:
- `--ticker`: Stock ticker symbol (default: RELIANCE.NS)
- `--start-date`: Start date for training data (default: 2020-01-01)
- `--end-date`: End date for training data (default: current date)
- `--days-back`: Number of days to look back for news (default: 30)
- `--model`: Model to train - 'xgboost', 'lstm', or 'all' (default: xgboost)
- `--sequence-length`: Sequence length for time series (default: 5)

### Running Dashboard

To launch the interactive dashboard:

```bash
streamlit run app.py
```

The dashboard will be available at http://localhost:8501

## 🔍 How It Works

1. **Data Collection**: Fetches historical stock data and news articles
2. **Sentiment Analysis**: Analyzes news headlines to extract sentiment
3. **Feature Engineering**: Calculates technical indicators and combines with sentiment data
4. **Model Training**: Trains XGBoost and LSTM models on the combined dataset
5. **Prediction**: Predicts next-day price movement direction
6. **Visualization**: Displays results in an interactive dashboard

## 📁 Project Structure

```
NSEnse/
├── app.py                      # Streamlit dashboard
├── main.py                     # Model training script
├── requirements.txt            # Project dependencies
├── NSEnse_sentiment_predictor.ipynb  # Jupyter notebook demo
├── configs/                    # Configuration files
├── data/                       # Data storage
├── models/                     # Trained models
├── notebooks/                  # Additional notebooks
├── output/                     # Output files
├── results/                    # Results and visualizations
└── src/                        # Source code
    ├── api/                    # API interfaces
    ├── data/                   # Data processing modules
    │   ├── data_processor.py   # Dataset creation
    │   ├── news_data.py        # News data collection
    │   └── stock_data.py       # Stock data collection
    ├── models/                 # Model implementations
    │   ├── lstm_model.py       # LSTM model
    │   ├── sentiment_analyzer.py # Sentiment analysis
    │   └── xgb_model.py        # XGBoost model
    ├── utils/                  # Utility functions
    └── visualization/          # Visualization tools
```

## 🧪 Model Performance

| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| XGBoost | 0.65     | 0.67      | 0.63   | 0.65     |
| LSTM    | 0.63     | 0.64      | 0.62   | 0.63     |

*Note: Performance metrics are approximate and may vary by stock and time period*

## 📝 Future Improvements

- [ ] Add more technical indicators
- [ ] Implement ensemble methods
- [ ] Add portfolio optimization
- [ ] Incorporate additional data sources (Twitter, Reddit)
- [ ] Add predictive confidence intervals

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👏 Acknowledgements

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [Transformers](https://github.com/huggingface/transformers) for sentiment analysis
- [Streamlit](https://streamlit.io/) for the interactive dashboard