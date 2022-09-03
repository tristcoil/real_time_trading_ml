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


# --------- random forest functions --------------

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
        "aboveEMA200",
        "LongSig",
        "ShortSig",
        "WLongSig",
        "WShortSig",
        "HH",
        "LL",
        "HL",
        "LH",
        "trend_conf",
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
    print(report)

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
    # for i, row in df.tail(
    #    20
    # ).iterrows():  # predict for small subset of data, otherwise it takes too long

    for i, row in df.iterrows():  # predict for each row

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
                df["aboveEMA200"][i],
                df["LongSig"][i],
                df["ShortSig"][i],
                df["WLongSig"][i],
                df["WShortSig"][i],
                df["HH"][i],
                df["LL"][i],
                df["HL"][i],
                df["LH"][i],
                df["trend_conf"][i],
            ]
        ]

        y_cls_pred_valid = clf.predict(X_cls_valid)
        df["Buy"][i] = y_cls_pred_valid[0].copy()

        print("step: ", i, "predicted class: ", df["Buy"][i])

    # add new column to better visualize Long only trades
    # graphs will look better, since no anchoring to zero for short trades
    df["Long"] = df["Buy"] * df["Adj Close"]
    df["Long"].replace(0, np.nan, inplace=True)

    print(df.tail())

    return df



def plot_forest_feature_importances(clf, predictors_list):
    # function plots feature importances of Random Forest classifier

    forest = clf

    start_time = time.time()
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    feature_names = predictors_list

    forest_importances = pd.Series(importances, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax, figsize=(15, 7))
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    return None





