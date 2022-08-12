import pandas as pd
import numpy as np
import ta
import talib
import os
from tqdm import tqdm
import time
from pathlib import Path
from time import time
from datetime import date
import time
import pickle
import argparse
import sys


def GetData(df, sym):

    if sym == "30Minutes":
        granularity = "30min"
    elif sym == "1Hour":
        granularity = "H"
    elif sym == "2Hours":
        granularity = "2H"
    elif sym == "3Hours":
        granularity = "3H"
    elif sym == "4Hours":
        granularity = "4H"
    elif sym == "5Hours":
        granularity = "5H"
    elif sym == "6Hours":
        granularity = "6H"
    elif sym == "Days":
        granularity = "D"
    elif sym == "Weeks":
        granularity = "W"
    elif sym == "Months":
        granularity = "M"
    # no need for Minute aggregation, data is olready in minute granularity
    # elif sym == 'Minutes':
    #    granularity = 'm'
    else:
        print("Time-Frame Not Implemented")

    # maybe volume should be sum, not mean
    agg_dict = {
        "Minutes_Open": "first",
        "Minutes_High": "max",
        "Minutes_Low": "min",
        "Minutes_Close": "last",
        "Minutes_Volume": "sum",
    }

    # resample dataframe to new time granularity
    # and give it better header
    r_df = df.resample(granularity, on="Minutes_Date Time").agg(agg_dict)
    r_df = r_df.reset_index()
    r_df.columns = [
        f"{sym}_Date Time",
        f"{sym}_Open",
        f"{sym}_High",
        f"{sym}_Low",
        f"{sym}_Close",
        f"{sym}_Volume",
    ]
    r_df = r_df.dropna()

    return r_df


def Create_Time_Frame(df, sym):
    df = df.copy()

    li = pd.DataFrame(
        columns=[
            f"{sym}_Date Time",
            f"{sym}_Open",
            f"{sym}_High",
            f"{sym}_Low",
            f"{sym}_Close",
            f"{sym}_Volume",
        ]
    )

    # quickly preprocess incoming dataframe
    # make date column real datetime object
    df["Minutes_Date Time"] = pd.to_datetime(df["Minutes_Date Time"])

    # process incoming dataframe
    li = GetData(df, sym)

    # ticker value cannot be time aggregated,
    # so put it back to resampled dataframe
    li[f"{sym}_Ticker"] = df["Minutes_Ticker"].values[0]

    return li


"""Calculates the indicator and append nthem to  the Dataset"""


def Calculate_Indicator(fd, sym):

    # to prevent pd copy warnings
    fd = fd.copy()

    try:
        # ADX works with talib
        fd[f"{sym}_ADX Trend"] = talib.ADX(
            high=fd[f"{sym}_High"].values,
            low=fd[f"{sym}_Low"].values,
            close=fd[f"{sym}_Close"].values,
        )

        fd[f"{sym}_RSI"] = ta.momentum.rsi(fd[f"{sym}_Close"])

        fd[f"{sym}_Stochastic"] = ta.momentum.stoch(
            low=fd[f"{sym}_Low"], close=fd[f"{sym}_Close"], high=fd[f"{sym}_High"]
        )
        fd[f"{sym}_Stochastic Signal"] = ta.momentum.stoch_signal(
            low=fd[f"{sym}_Low"], close=fd[f"{sym}_Close"], high=fd[f"{sym}_High"]
        )
        ###fd[f'{sym}_ADX Trend'] = ta.trend.adx(low = fd[f'{sym}_Low'], close = fd[f'{sym}_Close'], high = fd[f'{sym}_High'])
        fd[f"{sym}_Negative DI"] = ta.trend.adx_neg(
            low=fd[f"{sym}_Low"], close=fd[f"{sym}_Close"], high=fd[f"{sym}_High"]
        )
        fd[f"{sym}_Positive DI"] = ta.trend.adx_pos(
            low=fd[f"{sym}_Low"], close=fd[f"{sym}_Close"], high=fd[f"{sym}_High"]
        )

        fd[f"{sym}_MACD"] = ta.trend.macd(fd[f"{sym}_Close"])
        fd[f"{sym}_MACD Difference"] = ta.trend.macd_diff(fd[f"{sym}_Close"])
        fd[f"{sym}_MACD Signal"] = ta.trend.macd_signal(fd[f"{sym}_Close"])

    except:
        pass

    return fd


def create_dataframe_from_stream_file(stream_file):
    fd = pd.read_csv(stream_file, sep=" |,", header=None, engine="python")
    # rewrite columns to be in minutes
    fd.columns = [
        "Minutes_Date",
        "Minutes_Time",
        "Minutes_Open",
        "Minutes_High",
        "Minutes_Low",
        "Minutes_Close",
        "Minutes_Volume",
        "Minutes_Ticker",
    ]
    fd["Minutes_Date Time"] = fd[["Minutes_Date", "Minutes_Time"]].apply(
        lambda x: " ".join(x), axis=1
    )
    fd["Minutes_Date Time"] = pd.to_datetime(fd["Minutes_Date Time"])
    del fd["Minutes_Date"]
    del fd["Minutes_Time"]
    cols = [
        "Minutes_Date Time",
        "Minutes_Open",
        "Minutes_High",
        "Minutes_Low",
        "Minutes_Close",
        "Minutes_Volume",
        "Minutes_Ticker",
    ]
    # just rearrange colums to better order
    fd = fd[cols]

    return fd


def get_tickers(fd):
    # get list of unique tickers in the biig dataframe
    tickers_set = set(fd["Minutes_Ticker"].values)
    ticker_list = list(tickers_set)
    return ticker_list


def sort_file_by_date(file):
    # sorts file by date and rewrites the original file
    df = pd.read_csv(
        file, sep=",", header=None, engine="python", names=list("abcdefghijklmnop")
    )
    # date is marked as column 'a'
    df["a"] = pd.to_datetime(df["a"])
    df = df.sort_values(by="a", ascending=True)
    # print(df)
    df.to_csv(file, index=False, header=False)


def main(agg_timeframe, minute_indicators):
    path = "testData"
    today = date.today().strftime("%d-%m-%Y")
    # minute_indicators = True   # lets use minute indicators and csv saving only with one timeframe
    # I have reordered column order
    stream_file = "/home/AdminAccount/polygon/data_stream.csv"

    # agg_timeframe = '30Minutes'
    # agg_timeframe = '1Hour'
    # agg_timeframe = '2Hours'
    # agg_timeframe = '3Hours'
    # agg_timeframe = '4Hours'
    # agg_timeframe = '5Hours'
    # agg_timeframe = '6Hours'
    # agg_timeframe = 'Days'
    # agg_timeframe = 'Weeks'
    # agg_timeframe = 'Months'

    # final form of big fd dataframe
    fd = create_dataframe_from_stream_file(stream_file)

    ticker_list = get_tickers(fd)

    for ticker in ticker_list:
        # single ticker dataframe on minute granularity
        t_df = fd[fd["Minutes_Ticker"] == ticker]

        if minute_indicators == "True":
            # first calculate indicators on one minute timeframe (base timeframe)
            # dont calculate indicators for the big dataframe, makes no sense
            t_df = Calculate_Indicator(t_df, "Minutes")

            # save ticker minute dataframe with indicators to csv
            if not os.path.exists(f"{path}/minutes"):
                os.makedirs(f"{path}/minutes")
            t_df.to_csv(f"{path}/minutes/{ticker}.txt", index=False, header=False)
            # also append all minute tickers with indicators to temp file
            # (the t_df contains minute data with indicators for one stock)
            t_df.to_csv(
                f"{path}/minutes/all_tickers_{today}_temp.txt",
                index=False,
                mode="a",
                header=False,
            )

        # t_df can have indicators at this point as well
        agg_df = Create_Time_Frame(t_df, agg_timeframe)
        agg_df = Calculate_Indicator(agg_df, agg_timeframe)
        # save aggregated ticker dataframe with indicators to csv
        if not os.path.exists(f"{path}/{agg_timeframe}"):
            os.makedirs(f"{path}/{agg_timeframe}")
        agg_df.to_csv(f"{path}/{agg_timeframe}/{ticker}.txt", index=False, header=False)

        # also append all aggregated tickers with indicators to temp file
        agg_df.to_csv(
            f"{path}/{agg_timeframe}/all_tickers_{today}_temp.txt",
            index=False,
            mode="a",
            header=False,
        )

    if minute_indicators == "True":
        # MINUTE final aggregated file, file can already exist, in such case it gets rewritten, this is what we want
        sort_file_by_date(f"{path}/minutes/all_tickers_{today}_temp.txt")
        os.rename(
            f"{path}/minutes/all_tickers_{today}_temp.txt",
            f"{path}/minutes/all_tickers.txt",
        )

    # final aggregated file, file can already sexist, in such case it gets rewritten, this is what we want
    sort_file_by_date(f"{path}/{agg_timeframe}/all_tickers_{today}_temp.txt")
    os.rename(
        f"{path}/{agg_timeframe}/all_tickers_{today}_temp.txt",
        f"{path}/{agg_timeframe}/all_tickers.txt",
    )


if __name__ == "__main__":
    # call script like
    # python ./Concat.py 1Hour True     # True means we compute minute granularity indicators
    # python ./Concat.py 1Hour False    # without minute indicator computation
    # other time options: '30Minutes','1Hour','2Hours','3Hours','4Hours','5Hours','6Hours','Days','Weeks','Months'

    agg_timeframe = str(sys.argv[1])
    minute_indicators = str(sys.argv[2])
    main(agg_timeframe, minute_indicators)
