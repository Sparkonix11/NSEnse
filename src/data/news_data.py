import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import logging
import time
import random
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def scrape_economic_times_news(company, pages=2, max_results=50):
    """
    Scrape news about a company from Economic Times
    
    Args:
        company (str): Company name
        pages (int): Number of pages to scrape
        max_results (int): Maximum number of results to return
        
    Returns:
        pd.DataFrame: News data with headlines, dates, and URLs
    """
    base_url = "https://economictimes.indiatimes.com/archivelist/keywords"
    company_formatted = company.lower().replace(" ", "-")
    
    all_articles = []
    
    logger.info(f"Scraping news for {company}...")
    
    # In a real implementation, you would iterate through pages and extract data
    # This is a placeholder implementation
    
    # For demo purposes, let's create some mock data
    mock_data = [
        {"date": "2025-04-15", "headline": f"{company} reports strong Q4 results", "url": f"https://economictimes.com/{company_formatted}/article1", "source": "Economic Times"},
        {"date": "2025-04-10", "headline": f"{company} announces new product launch", "url": f"https://economictimes.com/{company_formatted}/article2", "source": "Economic Times"},
        {"date": "2025-04-05", "headline": f"{company} expands operations in South India", "url": f"https://economictimes.com/{company_formatted}/article3", "source": "Economic Times"},
        {"date": "2025-04-01", "headline": f"{company} faces regulatory scrutiny", "url": f"https://economictimes.com/{company_formatted}/article4", "source": "Economic Times"},
        {"date": "2025-03-28", "headline": f"{company} announces partnership with global tech firm", "url": f"https://economictimes.com/{company_formatted}/article5", "source": "Economic Times"},
    ]
    
    return pd.DataFrame(mock_data)


def fetch_news_api(company, days_back=30, api_key=None):
    """
    Fetch news about a company using NewsAPI
    
    Args:
        company (str): Company name
        days_back (int): Number of days to look back
        api_key (str): NewsAPI key (will use env var if None)
        
    Returns:
        pd.DataFrame: News data with headlines, dates, and URLs
    """
    # Use environment variable if no API key provided
    if api_key is None:
        api_key = os.environ.get("NEWS_API_KEY")
        if not api_key:
            logger.error("No NewsAPI key found in environment variables")
            raise ValueError("NewsAPI key is required. Set NEWS_API_KEY environment variable or pass it as a parameter.")
    
    base_url = "https://newsapi.org/v2/everything"
    
    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Format dates for API
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Parameters for API request
    params = {
        'q': company,
        'from': from_date,
        'to': to_date,
        'sortBy': 'publishedAt',
        'language': 'en',
        'apiKey': api_key
    }
    
    try:
        logger.info(f"Fetching news for '{company}' from {from_date} to {to_date}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the response
        data = response.json()
        
        if data['status'] == 'ok':
            articles = data['articles']
            
            # Process the articles
            processed_articles = []
            for article in articles:
                processed_articles.append({
                    'date': article['publishedAt'][:10],  # Extract date part
                    'headline': article['title'],
                    'snippet': article['description'],
                    'url': article['url'],
                    'source': article['source']['name']
                })
            
            return pd.DataFrame(processed_articles)
        else:
            logger.error(f"API Error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return pd.DataFrame()


def get_company_news(company, sources=['scrape', 'api'], days_back=30):
    """
    Get news from multiple sources and combine them
    
    Args:
        company (str): Company name
        sources (list): List of sources to use ('scrape', 'api')
        days_back (int): Number of days to look back
        
    Returns:
        pd.DataFrame: Combined news data
    """
    all_news = []
    
    if 'scrape' in sources:
        # Get scraped news
        scraped_news = scrape_economic_times_news(company)
        if not scraped_news.empty:
            all_news.append(scraped_news)
    
    if 'api' in sources:
        try:
            # Get news from API
            api_news = fetch_news_api(company, days_back=days_back)
            if not api_news.empty:
                all_news.append(api_news)
        except ValueError as e:
            logger.warning(f"Skipping API source: {e}")
    
    if not all_news:
        logger.warning(f"No news found for {company}")
        return pd.DataFrame()
    
    # Combine news from all sources
    combined_news = pd.concat(all_news, ignore_index=True)
    
    # Convert date to datetime
    combined_news['date'] = pd.to_datetime(combined_news['date'])
    
    # Sort by date (newest first)
    combined_news = combined_news.sort_values('date', ascending=False)
    
    # Drop duplicates based on headline
    combined_news = combined_news.drop_duplicates(subset=['headline'])
    
    return combined_news