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


