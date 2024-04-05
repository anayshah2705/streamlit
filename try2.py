import streamlit as st
import pickle
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model
from keras.models import load_model
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.momentum import StochasticOscillator
import matplotlib.pyplot as plt
import webbrowser
import plotly.graph_objects as go
import pymongo
import requests
import os
import pathlib
import textwrap
from tensorflow.keras.optimizers import Adam
import google.generativeai as genai
import h5py
from textblob import TextBlob
from bs4 import BeautifulSoup

from IPython.display import display
from IPython.display import Markdown

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Update connection string as needed
db = client["streamlit_app"]
users_collection = db["users"]
portfolio_collection = db["portfolio"]

# Load the LSTM model
with open('nifty50_lstm_model.pkl', 'rb') as f:
    lstm_model = pickle.load(f)

# Load the algorithmic trading model
with open('nifty50_trading_signals.pkl', 'rb') as f:
    algo_trading_model = pickle.load(f)

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

genai.configure(api_key="AIzaSyCwNPhDLcR7-SFJi5Qq5UxpeOekStij6Ag")

## Function to load OpenAI model and get respones

def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(question)
    return response.text

# Function to fetch historical stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to prepare data for LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to generate trading signals based on RSI
def generate_rsi_signals(data):
    rsi_indicator = RSIIndicator(close=data, window=14)
    rsi_values = rsi_indicator.rsi()
    buy_signal = (rsi_values < 30)  # RSI below 30 indicates oversold condition (potential buy)
    sell_signal = (rsi_values > 70)  # RSI above 70 indicates overbought condition (potential sell)
    return buy_signal, sell_signal

def generate_sma_signals(data):
    sma_indicator = SMAIndicator(close=data, window=50)
    sma_values = sma_indicator.sma_indicator()
    buy_signal = (data < sma_values)  # Price below SMA indicates potential buy
    sell_signal = (data > sma_values)  # Price above SMA indicates potential sell
    return buy_signal, sell_signal

def generate_macd_signals(data):
    macd_indicator = MACD(close=data)
    macd_signal = macd_indicator.macd_signal()
    macd_diff = macd_indicator.macd_diff()
    buy_signal = (macd_diff > 0) & (macd_diff.shift(1) < 0)  # MACD line crosses above the signal line
    sell_signal = (macd_diff < 0) & (macd_diff.shift(1) > 0)  # MACD line crosses below the signal line
    return buy_signal, sell_signal

# Function to generate trading signals based on Stochastic Oscillator
# Function to generate trading signals based on Stochastic Oscillator
def generate_stochastic_signals(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    stochastic_indicator = StochasticOscillator(high=high, low=low, close=close)
    stochastic_signal = stochastic_indicator.stoch_signal()
    buy_signal = (stochastic_signal.iloc[-1] < 20)  # Stochastic signal crosses below the oversold threshold
    sell_signal = (stochastic_signal.iloc[-1] > 80)  # Stochastic signal crosses above the overbought threshold
    return buy_signal, sell_signal


def fetch_bse_data():
    bse_stocks = ['RELIANCE.BO', 'TCS.BO', 'INFY.BO', 'HINDUNILVR.BO', 'ICICIBANK.BO', 'KOTAKBANK.BO', 'BHARTIARTL.BO',
                  'ITC.BO', 'ASIANPAINT.BO', 'LT.BO', 'HCLTECH.BO', 'AXISBANK.BO', 'BAJAJ-AUTO.BO', 'MARUTI.BO', 'POWERGRID.BO', 'TITAN.BO',
                  'NESTLEIND.BO', 'NTPC.BO', 'ONGC.BO', 'SUNPHARMA.BO', 'M&M.BO', 'UPL.BO', 'SHREECEM.BO', 'BRITANNIA.BO',
                  'HINDALCO.BO', 'DRREDDY.BO', 'INDUSINDBK.BO', 'JSWSTEEL.BO', 'COALINDIA.BO', 'BAJAJFINSV.BO', 'GRASIM.BO', 'HEROMOTOCO.BO',
                  'DIVISLAB.BO', 'SBILIFE.BO', 'ULTRACEMCO.BO', 'CIPLA.BO', 'TECHM.BO', 'HDFCLIFE.BO', 'TATASTEEL.BO', 'IOC.BO', 'ADANIPORTS.BO',
                  'SBIN.BO', 'DIVISLAB.BO', 'TATAMOTORS.BO', 'WIPRO.BO', 'DRREDDY.BO', 'NESTLEIND.BO', 'TCS.BO', 'HCLTECH.BO', 'RELIANCE.BO',
                  'INFY.BO', 'HINDUNILVR.BO', 'ITC.BO', 'TITAN.BO', 'ASIANPAINT.BO', 'HDFCBANK.BO', 'BHARTIARTL.BO', 'MARUTI.BO',
                  'KOTAKBANK.BO', 'LT.BO', 'AXISBANK.BO', 'SUNPHARMA.BO', 'ONGC.BO', 'M&M.BO', 'POWERGRID.BO', 'JSWSTEEL.BO', 'UPL.BO', 'NTPC.BO',
                  'HINDALCO.BO', 'SHREECEM.BO', 'BAJAJ-AUTO.BO', 'TATASTEEL.BO', 'BAJFINANCE.BO', 'HEROMOTOCO.BO', 'CIPLA.BO', 'GRASIM.BO',
                  'COALINDIA.BO', 'TECHM.BO', 'ICICIBANK.BO', 'HDFCLIFE.BO', 'DIVISLAB.BO', 'SBILIFE.BO', 'DRREDDY.BO', 'BAJAJFINSV.BO', 'ULTRACEMCO.BO',
                  'IOC.BO', 'TATAMOTORS.BO', 'RELIANCE.BO', 'TCS.BO', 'INFY.BO', 'HINDUNILVR.BO', 'ITC.BO', 'ASIANPAINT.BO', 'TITAN.BO',
                  'HDFCBANK.BO', 'BHARTIARTL.BO', 'MARUTI.BO', 'KOTAKBANK.BO', 'LT.BO', 'AXISBANK.BO', 'SUNPHARMA.BO', 'ONGC.BO', 'M&M.BO', 'POWERGRID.BO',
                  'JSWSTEEL.BO', 'UPL.BO', 'NTPC.BO', 'HINDALCO.BO', 'SHREECEM.BO', 'BAJAJ-AUTO.BO', 'TATASTEEL.BO', 'BAJFINANCE.BO', 'HEROMOTOCO.BO',
                  'CIPLA.BO', 'GRASIM.BO', 'COALINDIA.BO', 'TECHM.BO', 'ICICIBANK.BO', 'HDFCLIFE.BO', 'DIVISLAB.BO', 'SBILIFE.BO', 'DRREDDY.BO',
                  'BAJAJFINSV.BO', 'ULTRACEMCO.BO', 'IOC.BO', 'TATAMOTORS.BO']
    
    bse_data = {}
    for stock_symbol in bse_stocks:
        try:
            stock = yf.Ticker(stock_symbol)
            data = stock.history(period="1d")
            price = data["Close"].iloc[-1]
            bse_data[stock_symbol] = price
        except Exception as e:
            print(f"Error fetching data for {stock_symbol}: {e}")
            continue
    
    return bse_data

def generate_marquee_html(bse_data):
    marquee_content = ""
    for stock_symbol, price in bse_data.items():
        marquee_content += f"{stock_symbol}: ₹{price:.2f} &emsp;&emsp;"
    marquee_html = f"""
    <marquee behavior="scroll" direction="left" style="font-size: 20px; color: #0D19AE;">
        {marquee_content}
    </marquee>
    """
    return marquee_html

# Function to open URL in a new tab
def open_url_in_new_tab(url):
    webbrowser.open_new_tab(url)

# Function for user authentication
def authenticate_user(username, password):
    user = users_collection.find_one({"username": username, "password": password})
    return user is not None

# Function to save portfolio to MongoDB
def save_portfolio(username, portfolio):
    portfolio_collection.update_one({"username": username}, {"$set": {"portfolio": portfolio}}, upsert=True)

# Function to retrieve portfolio from MongoDB
def get_portfolio(username):
    user = portfolio_collection.find_one({"username": username})
    if user:
        return user["portfolio"]
    return {}

def fetch_news(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        news = stock.news
        return news
    except Exception as e:
        print(f"Error fetching news for {ticker_symbol}: {e}")
        return None
    
def fetch_news_sentiment(stock_symbol):
    try:
        # Fetch news related to the provided stock symbol from yfinance
        stock = yf.Ticker(stock_symbol)
        news = stock.news

        if not news:
            st.warning(f"No news articles found for {stock_symbol}.")
            return

        # Configure Gemini API
        genai.configure(api_key="AIzaSyCwNPhDLcR7-SFJi5Qq5UxpeOekStij6Ag")

        sentiment_scores = []

        for i, article in enumerate(news, 1):
            title = article['title']
            link = article['link']

            # Get summary of the article using Gemini API
            summary_response = get_gemini_summary(link)
            if summary_response:
                summary_text = summary_response.text
            else:
                st.warning(f"Unable to fetch summary for '{title}'. Skipping.")
                continue

            # Perform sentiment analysis on the summary
            sentiment_score = TextBlob(summary_text).sentiment.polarity
            sentiment_scores.append(sentiment_score)

            # Display title, link, summary, and sentiment score
            st.subheader(f"Article {i}: {title}")
            st.write(f"Link: {link}")
            st.write(f"Summary: {summary_text}")
            st.write(f"Sentiment Score: {sentiment_score:.2f}")

        # Calculate average sentiment score
        if sentiment_scores:
            average_score = sum(sentiment_scores) / len(sentiment_scores)
            st.subheader("Overall Sentiment Analysis:")
            st.write(f"Average Sentiment Score: {average_score:.2f}")
            if average_score > 0:
                st.success("Recommendation: Buy")
            elif average_score < 0:
                st.error("Recommendation: Sell")
            else:
                st.info("Recommendation: Hold")

    except Exception as e:
        st.error(f"Error fetching or summarizing news data: {e}")

def get_gemini_summary(link):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"summarize {link}")
    return response
      
# Streamlit app
def main():
    st.set_page_config(page_title="ArthaGyani",
                       page_icon='💸',
                       layout='centered')
    st.title('Stock Price Prediction, Algorithmic Trading Signals, and Portfolio Analysis')
    
    # Page state
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    # Login page
    if st.session_state.page == "login":
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            if authenticate_user(username, password):
                st.success('Login successful!')
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.page = "portfolio"
            else:
                st.error('Invalid username or password. Please try again.')
        if st.button('Create Account'):
            st.session_state.page = "create_account"

    # Create account page
    elif st.session_state.page == "create_account":
        new_username = st.text_input('New Username')
        new_password = st.text_input('New Password', type='password')
        if st.button('Create Account'):
            if users_collection.find_one({"username": new_username}):
                st.error('Username already exists. Please choose a different username.')
            else:
                users_collection.insert_one({"username": new_username, "password": new_password, "portfolio": {}})
                st.success('Account created successfully! You can now login.')
                st.session_state.page = "login"
        st.button('Go to Login Page', on_click=lambda: st.session_state.update(page="login"))
    # Portfolio page
    elif st.session_state.page == "portfolio":
        username = st.session_state.username
        # User input for portfolio allocation
        st.sidebar.title('Portfolio Allocation')
        allocation = get_portfolio(username)
        for stock_symbol in ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
                      'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFC.NS',
                      'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'IOC.NS', 'INDUSINDBK.NS',
                      'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS',
                      'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS',
                      'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']:
            allocation[stock_symbol] = st.sidebar.number_input(f'How many stocks of {stock_symbol} do you have?', min_value=0, step=1, value=allocation.get(stock_symbol, 0))

        # Define start_date and end_date
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        # Button to generate portfolio analysis
        if st.sidebar.button('Generate Portfolio Analysis'):
            total_value = 0

            # Calculate portfolio value based on current prices
            for stock, quantity in allocation.items():
                if quantity > 0:
                    stock_data = fetch_stock_data(stock, start_date, end_date)
                    if not stock_data.empty:
                        current_price = stock_data['Close'].iloc[-1]
                        total_value += quantity * current_price

            # Save portfolio to MongoDB
            save_portfolio(username, allocation)
            
            # Display portfolio analysis
            st.subheader('Portfolio Analysis')
            st.write('Portfolio Value:', total_value)
            st.write('Individual Stock Allocation:')
            
            # Create pie chart using Plotly
            fig = go.Figure(data=[go.Pie(labels=list(allocation.keys()), values=list(allocation.values()))])
            fig.update_traces(textinfo='percent+label', pull=[0.1] * len(allocation.keys()), marker=dict(line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)
            fig.update_layout(title_text="Portfolio Allocation", title_x=0.5)
            st.plotly_chart(fig)

        # Streamlit app
        #st.title('Stock Price Prediction and Algorithmic Trading Signals')

        # User input
        stock_name = st.selectbox('Select Stock', ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS', 'BHARTIARTL.NS',
                          'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFC.NS',
                          'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'ITC.NS', 'IOC.NS', 'INDUSINDBK.NS',
                          'INFY.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS',
                          'RELIANCE.NS', 'SBILIFE.NS', 'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS',
                          'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'WIPRO.NS'])
        input_date_str = st.date_input('Select Date')

        # Sidebar for buttons

        st.header("Know More!(Powered by Gemini AI)")
        input=st.text_input("Input: ",key="input")
        submit=st.button("Ask the question")
        if submit:
            response=get_gemini_response(input)
            st.subheader("The Response is")
            st.write(response)

        st.sidebar.title('Actions')

        if st.sidebar.button('Predict Price'):
            # Fetch historical stock data
            start_date = '2010-01-01'
            end_date = datetime.strftime(input_date_str, '%Y-%m-%d')
            stock_data = fetch_stock_data(stock_name, start_date, end_date)['Close']

            # Predict the stock price using the LSTM model
            if not stock_data.empty:
                # Normalize the data
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(stock_data.values.reshape(-1, 1))

                # Prepare data for LSTM
                n_steps = 30
                X, _ = prepare_data(data_scaled, n_steps)

                # Reshape input data to match model input shape
                X = X.reshape((X.shape[0], X.shape[1], 1))

                # Predict using LSTM model
                predicted_price_scaled = lstm_model.predict(X)
                predicted_price = scaler.inverse_transform(predicted_price_scaled)[-1][0]

                st.write(f"Predicted price for {input_date_str}: ₹{predicted_price:.2f}")

                # Plot actual prices
                plt.figure(figsize=(10, 6))
                plt.plot(stock_data.index[-len(predicted_price_scaled):], stock_data[-len(predicted_price_scaled):])
                plt.title(f'Actual Prices for {stock_name}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                st.pyplot(plt)
            else:
                st.write("No data available for the selected date.")

        if st.sidebar.button('Trading Signals'):
            try:
                # Fetch historical stock data
                start_date = '2010-01-01'
                end_date = datetime.strftime(input_date_str, '%Y-%m-%d')
                stock_data = fetch_stock_data(stock_name, start_date, end_date)['Close']

                # Display trading signals
                if not stock_data.empty:
                    st.subheader('Trading Signals')
                    st.write('Date:', input_date_str)
                    st.write('Stock:', stock_name)

                    # Generate RSI signals
                    rsi_buy, rsi_sell = generate_rsi_signals(stock_data)
                    st.write('RSI Buy Signal:', rsi_buy[-1])
                    st.write('RSI Sell Signal:', rsi_sell[-1])

                    # Generate SMA signals
                    sma_buy, sma_sell = generate_sma_signals(stock_data)
                    st.write('SMA Buy Signal:', sma_buy[-1])
                    st.write('SMA Sell Signal:', sma_sell[-1])

                    macd_buy, macd_sell = generate_macd_signals(stock_data)
                    st.write('MACD Buy Signal:', macd_buy[-1])
                    st.write('MACD Sell Signal:', macd_sell[-1])

                    stochastic_buy, stochastic_sell = generate_stochastic_signals(stock_data)
                    st.write('Stochastic Buy Signal:', stochastic_buy[-1])
                    st.write('Stochastic Sell Signal:', stochastic_sell[-1])

                else:
                    st.write('No trading signals available for selected stock.')

            except KeyError:
                st.write('No data available for the selected date or stock.')

        if st.sidebar.button('Show Historical Data'):
            # Fetch historical stock data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            stock_data = fetch_stock_data(stock_name, start_date, end_date)['Close']

            # Plot historical stock prices
            if not stock_data.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(stock_data.index, stock_data)
                plt.title(f'Historical Prices for {stock_name}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                st.pyplot(plt)
            else:
                st.write("No data available for the selected stock.")


        if st.sidebar.button("Fetch News"):
        # Input for stock ticker symbol
            ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., MSFT):")

            if ticker_symbol:
            # Fetch news related to the provided stock ticker
                news = fetch_news(ticker_symbol)

                if news:
                    st.write(f"Latest news for {ticker_symbol}:")
                    for item in news:
                        st.markdown(f"**Title:** [{item['title']}]({item.get('link', '')})")
                        st.write(f"**Publisher:** {item.get('publisher', 'N/A')}")
                        if 'thumbnail' in item and 'resolutions' in item['thumbnail'] and item['thumbnail']['resolutions']:
                            st.image(item['thumbnail']['resolutions'][0]['url'], width=300)
                        else:
                            st.write("Thumbnail not available")
                        st.write("---")
                else:
                    st.write("No news found for the provided stock ticker.")
        
        if st.sidebar.button('Sentiment Analysis'):
            stock_symbol = st.text_input("Enter Stock Symbol (e.g., MSFT for Microsoft):")
            fetch_news_sentiment(stock_symbol)

        if st.sidebar.button('Buy/Sell'):
            open_url_in_new_tab('https://zerodha.com/')


    st.sidebar.title('BSE Data')
    bse_data = fetch_bse_data()
    marquee_html = generate_marquee_html(bse_data)
    st.sidebar.markdown(marquee_html, unsafe_allow_html=True)


if __name__ == '__main__':
    main()