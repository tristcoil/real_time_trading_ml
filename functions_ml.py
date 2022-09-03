import streamlit as st

import talib as ta
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


def get_data(ticker, start_time, end_time):
    # yahoo API
    connected = False
    while not connected:
        try:
            df = yf.download(ticker, start=start_time, end=end_time)
            connected = True
            print(f"{ticker}: connected to yahoo")
        except Exception as e:
            print("type error: " + str(e))
            time.sleep(5)
            pass

    # use numerical integer index instead of date
    df = df.reset_index()
    # print(df.head(5))
    return df


def compute_technical_indicators(df):
    df["EMA5"] = ta.EMA(df["Adj Close"].values, timeperiod=5)
    df["EMA10"] = ta.EMA(df["Adj Close"].values, timeperiod=10)
    df["EMA15"] = ta.EMA(df["Adj Close"].values, timeperiod=15)
    df["EMA20"] = ta.EMA(df["Adj Close"].values, timeperiod=10)
    df["EMA30"] = ta.EMA(df["Adj Close"].values, timeperiod=30)
    df["EMA40"] = ta.EMA(df["Adj Close"].values, timeperiod=40)
    df["EMA50"] = ta.EMA(df["Adj Close"].values, timeperiod=50)

    df["EMA60"] = ta.EMA(df["Adj Close"].values, timeperiod=60)
    df["EMA70"] = ta.EMA(df["Adj Close"].values, timeperiod=70)
    df["EMA80"] = ta.EMA(df["Adj Close"].values, timeperiod=80)
    df["EMA90"] = ta.EMA(df["Adj Close"].values, timeperiod=90)

    df["EMA100"] = ta.EMA(df["Adj Close"].values, timeperiod=100)
    df["EMA150"] = ta.EMA(df["Adj Close"].values, timeperiod=150)
    df["EMA200"] = ta.EMA(df["Adj Close"].values, timeperiod=200)

    df["upperBB"], df["middleBB"], df["lowerBB"] = ta.BBANDS(
        df["Adj Close"].values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df["SAR"] = ta.SAR(
        df["High"].values, df["Low"].values, acceleration=0.02, maximum=0.2
    )
    df["RSI"] = ta.RSI(df["Adj Close"].values, timeperiod=14)

    df.tail()

    return df


def compute_features(df):
    # computes features for forest decisions
    df["aboveEMA5"] = np.where(df["Adj Close"] > df["EMA5"], 1, -1)
    df["aboveEMA10"] = np.where(df["Adj Close"] > df["EMA10"], 1, -1)
    df["aboveEMA15"] = np.where(df["Adj Close"] > df["EMA15"], 1, -1)
    df["aboveEMA20"] = np.where(df["Adj Close"] > df["EMA20"], 1, -1)
    df["aboveEMA30"] = np.where(df["Adj Close"] > df["EMA30"], 1, -1)
    df["aboveEMA40"] = np.where(df["Adj Close"] > df["EMA40"], 1, -1)

    df["aboveEMA50"] = np.where(df["Adj Close"] > df["EMA50"], 1, -1)
    df["aboveEMA60"] = np.where(df["Adj Close"] > df["EMA60"], 1, -1)
    df["aboveEMA70"] = np.where(df["Adj Close"] > df["EMA70"], 1, -1)
    df["aboveEMA80"] = np.where(df["Adj Close"] > df["EMA80"], 1, -1)
    df["aboveEMA90"] = np.where(df["Adj Close"] > df["EMA90"], 1, -1)

    df["aboveEMA100"] = np.where(df["Adj Close"] > df["EMA100"], 1, -1)
    df["aboveEMA150"] = np.where(df["Adj Close"] > df["EMA150"], 1, -1)
    df["aboveEMA200"] = np.where(df["Adj Close"] > df["EMA200"], 1, -1)

    df["aboveUpperBB"] = np.where(df["Adj Close"] > df["upperBB"], 1, -1)
    df["belowLowerBB"] = np.where(df["Adj Close"] < df["lowerBB"], 1, -1)

    df["aboveSAR"] = np.where(df["Adj Close"] > df["SAR"], 1, -1)

    df["oversoldRSI"] = np.where(df["RSI"] < 30, 1, -1)
    df["overboughtRSI"] = np.where(df["RSI"] > 70, 1, -1)

    # very important - cleanup NaN values, otherwise prediction does not work
    df = df.fillna(0).copy()

    # df.tail()

    return df


def plot_train_data(df):
    # plot price
    plt.figure(figsize=(15, 2.5))
    # plt.title("Stock data " + str(ticker))
    plt.title("Stock data ")
    plt.plot(df["Date"], df["Adj Close"])
    # plt.title('Price chart (Adj Close) ' + str(ticker))
    plt.show()
    return None


def define_target_condition(df, n_ticks = 55):

    # price higher later - bad predictive results
    # df['target_cls'] = np.where(df['Adj Close'].shift(-34) > df['Adj Close'], 1, 0)

    # price above trend multiple days later
    df["target_cls"] = np.where(df["Adj Close"].shift(-n_ticks) > df.EMA150.shift(-n_ticks), 1, 0)

    # important, remove NaN values
    df = df.fillna(0).copy()

    df.tail()

    return df








def plot_stock_prediction(df, ticker):
    # plot  values and significant levels
    plt.figure(figsize=(20, 7))
    plt.title("Predictive model " + str(ticker))
    plt.plot(df["Date"], df["Adj Close"], label="High", alpha=0.2)

    plt.plot(df["Date"], df["EMA10"], label="EMA10", alpha=0.2)
    plt.plot(df["Date"], df["EMA20"], label="EMA20", alpha=0.2)
    plt.plot(df["Date"], df["EMA30"], label="EMA30", alpha=0.2)
    plt.plot(df["Date"], df["EMA40"], label="EMA40", alpha=0.2)
    plt.plot(df["Date"], df["EMA50"], label="EMA50", alpha=0.2)
    plt.plot(df["Date"], df["EMA100"], label="EMA100", alpha=0.2)
    plt.plot(df["Date"], df["EMA150"], label="EMA150", alpha=0.99)
    plt.plot(df["Date"], df["EMA200"], label="EMA200", alpha=0.2)

    plt.scatter(
        df["Date"],
        df["Buy"] * df["Adj Close"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    plt.legend()

    plt.show()

    return None


def plot_stock_prediction_zoom(df, ticker, ticks_back):
    # --- plot only Long trades and zoom in on last data ---

    # plot  values and significant levels
    # df.reset_index(inplace=True)

    # zoom in
    df = df.iloc[-ticks_back:]  # use eg. 50 for zooming in

    plt.figure(figsize=(20, 7))
    plt.title("Predictive model " + str(ticker))
    plt.plot(df.index, df["Adj Close"], label="High", alpha=0.4)

    plt.plot(df.index, df["EMA10"], label="EMA10", alpha=0.2)
    plt.plot(df.index, df["EMA20"], label="EMA20", alpha=0.2)
    plt.plot(df.index, df["EMA30"], label="EMA30", alpha=0.2)
    plt.plot(df.index, df["EMA40"], label="EMA40", alpha=0.2)
    plt.plot(df.index, df["EMA50"], label="EMA50", alpha=0.2)
    plt.plot(df.index, df["EMA100"], label="EMA100", alpha=0.2)
    plt.plot(df.index, df["EMA150"], label="EMA150", alpha=0.79)
    plt.plot(df.index, df["EMA200"], label="EMA200", alpha=0.99)

    # this dataobject plotting gives intraday gaps since data from non trading time is not there
    # plt.scatter(
    #    df["Date"],
    #    #df["Buy"] * df["Adj Close"],
    #    df['Long'],
    #    label="Buy",
    #    marker="^",
    #    color="magenta",
    #    alpha=0.55,
    # )

    # workaround with plotting over index

    plt.scatter(
        df.index,
        # df["Buy"] * df["Adj Close"],
        df["Long"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.55,
    )

    # avoid intraday gaps by overlaying timestamp values over index ticks
    plt.xticks(df.index, df["Date"], rotation="vertical")

    # make sure the x date ticks are not overlapping
    plt.locator_params(axis="x", nbins=15)

    # plt.xticks(x, labels, rotation='vertical')
    # Pad margins so that markers don't get clipped by the axes
    # plt.margins(0.2)
    # Tweak spacing to prevent clipping of tick-labels
    # plt.subplots_adjust(bottom=0.15)

    plt.legend()
    plt.show()

    return None


def save_model(clf):
    joblib.dump(clf, "./random_forest.joblib")
    return None


# --------- streamlit specific functions ----------
def plot_stock_prediction_streamlit(df, ticker):
    # plot  values and significant levels
    fig, ax = plt.subplots()
    # ax.figure(figsize=(20, 7))
    # ax.title("Predictive model " + str(ticker))
    ax.plot(df["Date"], df["Adj Close"], label="High", alpha=0.2)

    ax.plot(df["Date"], df["EMA10"], label="EMA10", alpha=0.2)
    ax.plot(df["Date"], df["EMA20"], label="EMA20", alpha=0.2)
    ax.plot(df["Date"], df["EMA30"], label="EMA30", alpha=0.2)
    ax.plot(df["Date"], df["EMA40"], label="EMA40", alpha=0.2)
    ax.plot(df["Date"], df["EMA50"], label="EMA50", alpha=0.2)
    ax.plot(df["Date"], df["EMA100"], label="EMA100", alpha=0.2)
    ax.plot(df["Date"], df["EMA150"], label="EMA150", alpha=0.99)
    ax.plot(df["Date"], df["EMA200"], label="EMA200", alpha=0.2)

    ax.scatter(
        df["Date"],
        # df["Buy"] * df["Adj Close"],
        df["Long"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    ax.legend()

    # plot matplotlib plt in streamlit
    # st.pyplot(fig)

    return st.pyplot(fig)


def plot_stock_prediction_zoom_streamlit(df, ticker):
    # --- plot only Long trades and zoom in on last data ---

    # plot  values and significant levels

    df = df.iloc[-20:]

    # plot  values and significant levels
    fig, ax = plt.subplots()
    # ax.figure(figsize=(20, 7))
    # ax.title("Predictive model " + str(ticker))
    ax.plot(df["Date"], df["Adj Close"], label="High", alpha=0.2)

    ax.plot(df["Date"], df["EMA10"], label="EMA10", alpha=0.2)
    ax.plot(df["Date"], df["EMA20"], label="EMA20", alpha=0.2)
    ax.plot(df["Date"], df["EMA30"], label="EMA30", alpha=0.2)
    ax.plot(df["Date"], df["EMA40"], label="EMA40", alpha=0.2)
    ax.plot(df["Date"], df["EMA50"], label="EMA50", alpha=0.2)
    ax.plot(df["Date"], df["EMA100"], label="EMA100", alpha=0.2)
    ax.plot(df["Date"], df["EMA150"], label="EMA150", alpha=0.99)
    ax.plot(df["Date"], df["EMA200"], label="EMA200", alpha=0.2)

    ax.scatter(
        df["Date"],
        # df["Buy"] * df["Adj Close"],
        df["Long"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    ax.legend()

    # plot matplotlib plt in streamlit
    # st.pyplot(fig)

    return st.pyplot(fig)
