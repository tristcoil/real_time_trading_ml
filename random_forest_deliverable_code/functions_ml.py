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


def define_target_condition(df):

    # price higher later - bad predictive results
    # df['target_cls'] = np.where(df['Adj Close'].shift(-34) > df['Adj Close'], 1, 0)

    # price above trend multiple days later
    df["target_cls"] = np.where(df["Adj Close"].shift(-55) > df.EMA150.shift(-55), 1, 0)

    # important, remove NaN values
    df = df.fillna(0).copy()

    df.tail()

    return df


def splitting_and_training(df):
    # __predictors__
    predictors_list = [
        "aboveSAR",
        "aboveUpperBB",
        "belowLowerBB",
        "RSI",
        "oversoldRSI",
        "overboughtRSI",
        "aboveEMA5",
        "aboveEMA10",
        "aboveEMA15",
        "aboveEMA20",
        "aboveEMA30",
        "aboveEMA40",
        "aboveEMA50",
        "aboveEMA60",
        "aboveEMA70",
        "aboveEMA80",
        "aboveEMA90",
        "aboveEMA100",
    ]

    # __features__
    X = df[predictors_list].fillna(0)
    # print('X.tail', X.tail())
    X = X.to_numpy()
    # print('X', X)

    # __targets__
    y_cls = df.target_cls.fillna(0)
    # print('y_cls.tail', y_cls.tail(10))
    y_cls = y_cls.to_numpy()
    # print('y_cls', y_cls)

    # __train test split__
    # from sklearn.model_selection import train_test_split
    y = y_cls
    X_cls_train, X_cls_test, y_cls_train, y_cls_test = train_test_split(
        X, y, test_size=0.3, random_state=432, stratify=y
    )

    # print (X_cls_train.shape, y_cls_train.shape)
    # print (X_cls_test.shape, y_cls_test.shape)

    # __RANDOM FOREST __       - retrainable - warm_start
    # from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier - incremental training - warm_start=True
    clf = RandomForestClassifier(
        n_estimators=500,
        criterion="gini",
        max_depth=20,
        min_samples_leaf=10,
        n_jobs=-1,
        warm_start=True,
    )

    # __ACTUAL TRAINING __
    clf = clf.fit(X_cls_train, y_cls_train)
    # clf

    # __making accuracy report__
    # ideally should be getting better with each round
    y_cls_pred = clf.predict(X_cls_test)

    # from sklearn.metrics import classification_report
    report = classification_report(y_cls_test, y_cls_pred)
    # print(report)

    return clf


def predict_timeseries(df, clf):

    # making sure we have good dimensions
    # column will be rewritten later
    df["Buy"] = np.nan

    print("df length: ", len(df))

    # for i in range(len(df)):
    #    print('above sar: ', df["aboveSAR"][i])

    # iterate over last 20 rows in a dataframe
    # use df.iterrows() to iterate over rows
    #for i, row in df.tail(
    #    20
    #).iterrows():  # predict for small subset of data, otherwise it takes too long

    for i, row in df.iterrows():    # predict for each row

        X_cls_valid = [
            [
                df["aboveSAR"][i],
                df["aboveUpperBB"][i],
                df["belowLowerBB"][i],
                df["RSI"][i],
                df["oversoldRSI"][i],
                df["overboughtRSI"][i],
                df["aboveEMA5"][i],
                df["aboveEMA10"][i],
                df["aboveEMA15"][i],
                df["aboveEMA20"][i],
                df["aboveEMA30"][i],
                df["aboveEMA40"][i],
                df["aboveEMA50"][i],
                df["aboveEMA60"][i],
                df["aboveEMA70"][i],
                df["aboveEMA80"][i],
                df["aboveEMA90"][i],
                df["aboveEMA100"][i],
            ]
        ]

        y_cls_pred_valid = clf.predict(X_cls_valid)
        df["Buy"][i] = y_cls_pred_valid[0].copy()

        print("step: ", i, "predicted class: ", df["Buy"][i])


    # add new column to better visualize Long only trades
    # graphs will look better, since no anchoring to zero for short trades
    df['Long'] = df['Buy'] * df['Adj Close'] 
    df['Long'].replace(0, np.nan, inplace=True) 

    print(df.tail())

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
        #df["Buy"] * df["Adj Close"],
        df["Long"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    plt.legend()

    plt.show()

    return None


def plot_stock_prediction_zoom(df, ticker):
    # --- plot only Long trades and zoom in on last data ---
    
    # plot  values and significant levels
    
    df = df.iloc[-20:]
    
    plt.figure(figsize=(20, 7))
    plt.title("Predictive model " + str(ticker))
    plt.plot(df["Date"], df["Adj Close"], label="High", alpha=0.4)

    plt.plot(df["Date"], df["EMA10"], label="EMA10", alpha=0.2)
    plt.plot(df["Date"], df["EMA20"], label="EMA20", alpha=0.2)
    plt.plot(df["Date"], df["EMA30"], label="EMA30", alpha=0.2)
    plt.plot(df["Date"], df["EMA40"], label="EMA40", alpha=0.2)
    plt.plot(df["Date"], df["EMA50"], label="EMA50", alpha=0.2)
    plt.plot(df["Date"], df["EMA100"], label="EMA100", alpha=0.2)
    plt.plot(df["Date"], df["EMA150"], label="EMA150", alpha=0.79)
    plt.plot(df["Date"], df["EMA200"], label="EMA200", alpha=0.99)

    plt.scatter(
        df["Date"],
        #df["Buy"] * df["Adj Close"],
        df['Long'],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.55,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

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
        #df["Buy"] * df["Adj Close"],
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
        #df["Buy"] * df["Adj Close"],
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







