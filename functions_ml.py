import streamlit as st

import ta
import talib as tal
import yfinance as yf
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# ML related imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

#from distutils.errors import LinkError
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream

from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream

# --- additional setup ---
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
# suppress 'SettingWithCopy' warning
pd.set_option("mode.chained_assignment", None)
# make pandas to print dataframes nicely
pd.set_option("expand_frame_repr", False)


# -------------------------- Function Definitions --------------------------------


def get_data(ticker, interval):
    # yahoo API
    connected = False
    while not connected:
        try:
            # 1 minute takes 5 day history by default, it is also max history
            # df = yf.download(ticker, interval=interval, start=start_time, end=end_time)
            df = yf.download(ticker, interval=interval) 
            connected = True
            print('connected to yahoo')
        except Exception as e:
            print("type error: " + str(e))
            time.sleep( 5 )
            pass   

    # use numerical integer index instead of date    
    df = df.reset_index()
    # print(df.head(5))

    # for one minute, we need to rename "Datetime" column to "Date"
    df.rename(columns = {'Datetime':'Date'}, inplace = True)

    return df



def get_alpaca_hist_data(symbol, data_type, API_KEY, SECRET_KEY):
    # function downloads minute data from alpaca api
    # examples
    # symbol = "AAPL"
    # data_type = "stocks" / "crypto"

    # example of crypto dataframe
    #                          exchange      open      high       low     close     volume  trade_count          vwap
    #timestamp                                                                                                       
    #2022-09-18 05:00:00+00:00     CBSE  20012.03  20013.17  20001.79  20002.47   1.726999          183  20007.886639
    #2022-09-18 05:00:00+00:00     FTXU  20007.00  20007.00  20007.00  20007.00   0.040800            1  20007.000000
    #2022-09-18 05:01:00+00:00     CBSE  20002.49  20003.71  19997.00  20002.77   0.821499          184  20000.362906
    #2022-09-18 05:01:00+00:00     FTXU  20001.00  20001.00  20001.00  20001.00   1.002000            4  20001.000000
    #2022-09-18 05:02:00+00:00     CBSE  20001.56  20010.79  20001.56  20008.20   1.060122          125  20007.932954


    # --- PREP STEPS ---
    # pick api data source
    #rest_api = REST(API_KEY, SECRET_KEY, "https://paper-api.alpaca.markets")     # paper trading data source
    #rest_api = REST(API_KEY, SECRET_KEY, "https://api.alpaca.markets")           # live trading API
    rest_api = REST(API_KEY, SECRET_KEY, "https://data.alpaca.markets")          # data endpoint


    # CRYPTO
    if data_type == "crypto":
        # Retrieve daily bar data for Bitcoin in a DataFrame
        #btc_bars = rest_api.get_crypto_bars("BTCUSD", TimeFrame.Minute, "2022-09-10", "2022-09-18").df
        df = rest_api.get_crypto_bars(symbol=symbol, timeframe=TimeFrame.Minute, limit=10000).df   

        #print("API data:")
        #print(df)

        #print(set(df.exchange.tolist()))     # currently there are 3 exchanges: {'FTXU', 'CBSE', 'ERSX'}
        # CBSE looks to have big trading volume

        # display only rows from df dataframe that contain exchange column equal to CBSE
        df = df[df['exchange'] == 'CBSE']
        df.reindex()    # reindex dataframe to start from 0
        #print(df)

        # other useful api options
        # Quote and trade data are also available for cryptocurrencies
        # btc_quotes = rest_api.get_crypto_quotes('BTCUSD', '2021-01-01', '2021-01-05').df
        # btc_trades = rest_api.get_crypto_trades('BTCUSD', '2021-01-01', '2021-01-05').df


    # STOCKS
    if data_type == "stocks":

        #stock_bars = rest_api.get_bars("AAPL", TimeFrame.Minute, start="2022-09-10", end="2022-09-17").df
        #stock_bars = rest_api.get_bars(symbol="AAPL", timeframe=TimeFrame.Minute, start="2022-09-10", end="2022-09-17T00:00:00.52Z", limit=10000).df

        now = datetime.datetime.now(datetime.timezone.utc)
        now.isoformat()

        end = now - datetime.timedelta(minutes=15.1)   # a bit more than 15 mins back
        start = now - datetime.timedelta(days=10)      # get minute data few days back

        #stock_bars = rest_api.get_bars(symbol="AAPL", timeframe=TimeFrame.Minute, start="2022-09-10", end="2022-09-17T16:49:57.055823+00:00", limit=10000).df
        df = rest_api.get_bars(symbol=symbol, timeframe=TimeFrame.Minute, start=start.isoformat(), end=end.isoformat(), limit=10000).df

        #print("Stock API data:")
        #print(df)

    # make the alpaca df look similar to df from Yahoo Finance
    # rename df columns "timestamp", 'open', 'high', 'low', 'close', 'volume' to "Date", 'Open', 'High', 'Low', 'Close', 'Volume'
    df.reset_index(inplace=True)
    df = df.rename(columns={"timestamp": "Date", 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df['Adj Close'] = df['Close']
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    print("FINAL DATAFRAME: ")
    print(df)

    #FINAL DATAFRAME: 
    #                      Date      Open      High     Low     Close  Volume
    #0    2022-09-08 18:32:00+00:00  154.0200  154.1100  153.99  154.1000  150359
    #1    2022-09-08 18:33:00+00:00  154.0907  154.2000  154.09  154.1500  170388
    #2    2022-09-08 18:34:00+00:00  154.1550  154.2100  154.12  154.1550  102371
    #3    2022-09-08 18:35:00+00:00  154.1500  154.2199  154.06  154.1199  146318
    #4    2022-09-08 18:36:00+00:00  154.1200  154.2181  154.10  154.2100   97022

    return df























def compute_technical_indicators(df):
    df['EMA5'] = tal.EMA(df['Adj Close'].values, timeperiod=5)
    df['EMA10'] = tal.EMA(df['Adj Close'].values, timeperiod=10)
    df['EMA15'] = tal.EMA(df['Adj Close'].values, timeperiod=15)
    df['EMA20'] = tal.EMA(df['Adj Close'].values, timeperiod=10)
    df['EMA30'] = tal.EMA(df['Adj Close'].values, timeperiod=30)
    df['EMA40'] = tal.EMA(df['Adj Close'].values, timeperiod=40)
    df['EMA50'] = tal.EMA(df['Adj Close'].values, timeperiod=50)

    df['EMA60'] = tal.EMA(df['Adj Close'].values, timeperiod=60)
    df['EMA70'] = tal.EMA(df['Adj Close'].values, timeperiod=70)
    df['EMA80'] = tal.EMA(df['Adj Close'].values, timeperiod=80)
    df['EMA90'] = tal.EMA(df['Adj Close'].values, timeperiod=90)
    
    df['EMA100'] = tal.EMA(df['Adj Close'].values, timeperiod=100)
    df['EMA150'] = tal.EMA(df['Adj Close'].values, timeperiod=150)
    df['EMA200'] = tal.EMA(df['Adj Close'].values, timeperiod=200)

    df['upperBB'], df['middleBB'], df['lowerBB'] = tal.BBANDS(df['Adj Close'].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['SAR'] = tal.SAR(df['High'].values, df['Low'].values, acceleration=0.02, maximum=0.2)
    df['RSI'] = tal.RSI(df['Adj Close'].values, timeperiod=14)

    df.tail()

    return df


def compute_features(df):
    # computes features for forest decisions
    df['aboveEMA5'] = np.where(df['Adj Close'] > df['EMA5'], 1, -1)
    df['aboveEMA10'] = np.where(df['Adj Close'] > df['EMA10'], 1, -1)
    df['aboveEMA15'] = np.where(df['Adj Close'] > df['EMA15'], 1, -1)
    df['aboveEMA20'] = np.where(df['Adj Close'] > df['EMA20'], 1, -1)
    df['aboveEMA30'] = np.where(df['Adj Close'] > df['EMA30'], 1, -1)
    df['aboveEMA40'] = np.where(df['Adj Close'] > df['EMA40'], 1, -1)
    
    df['aboveEMA50'] = np.where(df['Adj Close'] > df['EMA50'], 1, -1)
    df['aboveEMA60'] = np.where(df['Adj Close'] > df['EMA60'], 1, -1)
    df['aboveEMA70'] = np.where(df['Adj Close'] > df['EMA70'], 1, -1)
    df['aboveEMA80'] = np.where(df['Adj Close'] > df['EMA80'], 1, -1)
    df['aboveEMA90'] = np.where(df['Adj Close'] > df['EMA90'], 1, -1)
    
    df['aboveEMA100'] = np.where(df['Adj Close'] > df['EMA100'], 1, -1)
    df['aboveEMA150'] = np.where(df['Adj Close'] > df['EMA150'], 1, -1)
    df['aboveEMA200'] = np.where(df['Adj Close'] > df['EMA200'], 1, -1)

    df['aboveUpperBB'] = np.where(df['Adj Close'] > df['upperBB'], 1, -1)
    df['belowLowerBB'] = np.where(df['Adj Close'] < df['lowerBB'], 1, -1)
    
    df['aboveSAR'] = np.where(df['Adj Close'] > df['SAR'], 1, -1)
   
    df['oversoldRSI'] = np.where(df['RSI'] < 30, 1, -1)
    df['overboughtRSI'] = np.where(df['RSI'] > 70, 1, -1)


    # very important - cleanup NaN values, otherwise prediction does not work
    # but then it causes plot to look ugly
    # best would be to always skip first 200 datapoints
    # even for predictions
    # just wait until all features are computed so we dont cast them to zeros
    df=df.fillna(0).copy()

    
    #df.tail()

    return df


def plot_train_data(df, ticker):
    # plot price
    plt.figure(figsize=(15,2.5))
    plt.title('Stock data ' + str(ticker))
    plt.plot(df['Date'], df['Adj Close'])
    #plt.title('Price chart (Adj Close) ' + str(ticker))
    plt.show()
    return None


def define_target_condition(df, n_ticks = 55):
    # compares status with n ticks ahead in the future
    # the minus in front of n_ticks var actually means shift to future

    # price higher later - bad predictive results
    #df['target_cls'] = np.where(df['Adj Close'].shift(-34) > df['Adj Close'], 1, 0)    
    
    # price above trend multiple days later
    df['target_cls'] = np.where(df['Adj Close'].shift(-n_ticks) > df.EMA150.shift(-n_ticks), 1, 0)

    # important, remove NaN values
    df=df.fillna(0).copy()
    
    df.tail()
    
    return df



def save_model(clf, model_name = "./random_forest.joblib"):
    joblib.dump(clf, model_name)
    return None


