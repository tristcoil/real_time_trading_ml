from fastapi import FastAPI, WebSocket
from random import choice, randint
import asyncio

import pandas as pd
import numpy as np

# -------
from functions_gen import *  # general functions
from functions_ml import *  # machine learning
from functions_viz import *  # visualization
from functions_db import *  # database


app = FastAPI()


# var definition:
symbol = "AAPL"
db_name = "alpaca_websocket_stream_data.db"
table_name = "alpaca_websocket_stream_data"
granularity = "1Min"
interval = "1m"  # for yahoo finance model training if needed


@app.websocket("/sample")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:

        # we need to:
        # - get data from db
        # - convert it to dataframe
        # - split OHLC df columns to json payload
        # - and send it to consumer

        # ---- get data from db ------

        # gets all data from db for given ticker
        ####data =  get_ticker_data_from_db(symbol, db_name, table_name)

        # ---- convert to df ---------
        # temp workaround with hardcoded dict
        # data_dict = {
        #    "DateTime": ["2022-10-01", "2022-10-02", "2022-10-03", "2022-10-04"],
        #    "Open": [23, 21, 22, 21],
        #    "High": [26 + np.random.randn(), 22, 23, 22],
        #    "Low": [
        #        22 - np.random.randn(),
        #        20 - np.random.randn(),
        #        21 - np.random.randn(),
        #        20 - np.random.randn(),
        #    ],
        #    "Close": [22, 21, 22, 21],
        #    "Adj Close": [22, 21, 22, 21],
        #    "Long": [26, 27, 28, 29],
        # }
        #
        # df = pd.DataFrame(data_dict)

        # resample tick by tick data from db to minute timeframe and save to df
        ####df =  resample_data(data, granularity=granularity)
        ####df['Date'] = df['Date'].astype(str)   # datetime object cannot be sent as JSON payload

        # TEMP WORKAROUND, TAKING DATA FROM YAHOO
        symbol = "BTC-USD"   # crypto streams all day new ticks, good for testing
        df = get_data(symbol, interval)
        df["Date"] = df["Date"].astype(str)

        # --- split dataframe to dictionary ---
        data_dict = df.to_dict()

        # --- send the json payload ---
        await websocket.send_json(data_dict)

        await asyncio.sleep(60)
