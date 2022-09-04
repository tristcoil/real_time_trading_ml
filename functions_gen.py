import pandas as pd
import numpy as np
import sqlite3


# ------------------- general functions -------------------


def resample_data(data, granularity="1Min"):
    # takes incoming data, converst it to dataframe
    # and resamples tick by tick 'price' column into new dataframe, returns resampled dataframe
    # granularity can be 1Min, 5Min, other granularities are also possible

    # load data to dataframe
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "symbol",
            "price",
            "size",
            "exchange",
            "conditions",
            "tape",
            "id",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
    df = df.set_index("timestamp")
    df_res = df["price"].resample(granularity).ohlc(_method="ohlc")

    # the neural net and Random Forest models expect df with columns called
    # 'Date','Open', 'High', 'Low', 'Close', 'Adj Close'
    # so we need to slightly mod our resampled df
    df_res.reset_index(inplace=True)
    df_res.rename(
        columns={
            "timestamp": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
        },
        inplace=True,
    )
    df_res["Adj Close"] = df_res["Close"]

    # data stream is not continuous, there are gaps between days, we need to remove the gaps
    # by taking only rows wit values that are not NaN
    # very important, otherwise indicators will compute wrong values
    df_res = df_res[df_res["Close"].notna()]

    # df_res.head()

    return df_res
