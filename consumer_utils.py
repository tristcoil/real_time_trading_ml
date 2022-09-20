import aiohttp
import asyncio
from collections import deque, defaultdict
from functools import partial

import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import datetime

import talib as ta
import yfinance as yf
import pandas as pd
import sqlite3

# custom function imports
from functions_gen import *  # general functions
from functions_ml import *  # machine learning
from functions_viz import *  # visualization
from functions_db import *  # database

# custom indicators moved to modules
from functions_superjump import *
from functions_HHLL import *
from functions_HHLL_conf import *
from functions_HHLL_channel import *
from functions_gator import *

from functions_forest import *  # Random Forest classifier

from concurrent.futures import ThreadPoolExecutor   # for parallel processing of blocking task with async

WS_CONN = "ws://localhost:8000/sample"  # for connection to stream

clf = joblib.load("./random_forest.joblib")  # load the trained model


#import nest_asyncio
#nest_asyncio.apply() 

#_executor = ThreadPoolExecutor(1)  


# var definition:
# symbol = 'AAPL'
# db_name = 'alpaca_websocket_stream_data.db'
# table_name= 'alpaca_websocket_stream_data'
# granularity = '1Min'
# interval = "1m" # for yahoo finance model training if needed


#def sync_blocking(df_short):
#    time.sleep(2)
#    predict_timeseries(df_short, clf)


#async def hello_world(loop, df_short):
#    # run blocking function in another thread,
#    # and wait for it's result:
#    await loop.run_in_executor(_executor, sync_blocking(df_short))




async def consumer(status, status2):

    async with aiohttp.ClientSession(trust_env=True) as session:
        status.subheader(f"Connecting to {WS_CONN}")
        async with session.ws_connect(WS_CONN) as websocket:
            status.subheader(f"Connected to: {WS_CONN}")
            async for message in websocket:
                # get the json payload from websocket
                data_dict = message.json()
                # print("WS data: ", data_dict)
                # creating a Dataframe object
                df = pd.DataFrame(data_dict)
                df["Date"] = pd.to_datetime(df["Date"])
                # st.write(df)
                # print(df.to_string())  # prints whole dataframe
                print(df)
                print(datetime.datetime.now())
                print(df.dtypes)

                # once we have this data in df, we can compute whatever indicators here
                # that is fast and also we can make like last 20 point prediction

                # df = get_data(symbol, interval)
                # df['Date'] = df['Date'].astype(str)
                # df['Date'] = pd.to_datetime(df['Date'])

                # print(df)
                # print(datetime.datetime.now())
                # print(df.dtypes)

                #    # prepare dfs with extra indicators
                out_df1 = superjumpTBB(df)  # superjumpTBB
                out_df1.replace({False: 0, True: 1}, inplace=True)
                out_df2 = HHLL_Strategy(df)  # HHHL indicator
                out_df2.replace({False: 0, True: 1}, inplace=True)
                out_df3 = HHLL_confirmation(df)  # HHLL confirmation
                # converting 'u','d', 'none' to integers for 'trend_conf' col
                out_df3.replace({"d": 0, "u": 1, "none": -1}, inplace=True)
                out_df4 = HHLL_Channel(df)
                out_df5 = rsi_strategy(df)
                #
                #    # compute general indicators, features and target
                df = compute_technical_indicators(df)
                df = compute_features(df)
                df = define_target_condition(df)
                print("targets computed")
                #
                #    # merge with custom indicators
                df = pd.merge(df, out_df1, how="inner", on="Date")
                df = pd.merge(df, out_df2, how="inner", on="Date")
                df = pd.merge(df, out_df3, how="inner", on="Date")
                df = pd.merge(df, out_df4, how="inner", on="Date")
                df = pd.merge(df, out_df5, how="inner", on="Date")

                #    # actual prediction
                #    # can take longer if the dataframe is big
                #    # so we are making shorter dataframe for this
                #    # bigger df is causing async conn to be unstable
                #    # we would need to use async loop for long running processes
                df_short = df.iloc[-10:]

                
                #loop.run_until_complete(hello_world(loop, df_short))

                predict_timeseries(df_short, clf)

                df = df_short

                # real time viz (indicators, predictions)
                with status2:

                    # possible solution to intra day gaps in plotly
                    # https://stackoverflow.com/questions/63780293/python-plotly-how-to-remove-datetime-gaps-in-candle-stick-chart

                    fig = go.Figure()
                    fig.add_trace(
                        go.Candlestick(
                            x=df["Date"],
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                            name="OHLC",
                        )
                    )

                    fig.add_trace(
                        go.Line(
                            x=df["Date"],
                            y=df["EMA20"],
                            name="EMA20",
                        )
                    )

                    fig.add_trace(
                        go.Line(
                            x=df["Date"],
                            y=df["EMA100"],
                            name="EMA100",
                        )
                    )

                    fig.add_trace(
                        go.Line(
                            x=df["Date"],
                            y=df["EMA100"],
                            name="EMA100",
                        )
                    )

                    fig.add_trace(
                        go.Line(
                            x=df["Date"],
                            y=df["EMA150"],
                            name="EMA150",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=df["Date"],
                            y=df["Long"],
                            mode="markers",
                            marker_color="magenta",
                            marker_symbol="triangle-up",
                            name="Buy",
                        )
                    )

                    st.plotly_chart(fig)
