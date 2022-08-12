import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import joblib
import time

from functions_gen import *
from functions_ml import *






symbol = 'AAPL'
db_name = 'alpaca_websocket_stream_data.db'
table_name= 'alpaca_websocket_stream_data'
granularity = '1Min'


   

# load classifier, no need to initialize the loaded_rf
loaded_clf = joblib.load("./random_forest.joblib")
clf = loaded_clf



symbol = st.sidebar.selectbox("Select stock symbol: ", ["AAPL", "SPY", "IBM"])
granularity = st.sidebar.selectbox("Select stock symbol: ", ["1Min", "5Min"])

#my_fig = st.empty()


data = get_ticker_data_from_db(symbol, db_name, table_name)
df_res = resample_data(data, granularity=granularity)

df_res = compute_technical_indicators(df_res)
df_res = compute_features(df_res)
df_res = define_target_condition(df_res)

# streamline for pred and viz
df_res_cut = df_res.iloc[-202:].copy()
predict_timeseries(df_res_cut, clf)



with st.empty():
#    while True:     # turns out that streamlit really does not like to run with while True loops and sleeps

        #my_fig = plot_stock_prediction_streamlit(df_res_cut, symbol)        # zoom out
        my_fig = plot_stock_prediction_zoom_streamlit(df_res_cut, symbol)  # zoom in

        # streamlit does not like loops and sleeps
        #print('sleeping ...')
        #if granularity == '1Min':
        #    time.sleep(60)
        #elif granularity == '5Min':
        #    time.sleep(300)
        #else:
        #    raise ValueError('granularity must be 1Min or 5Min')



