import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import joblib
import time

from functions_gen import *
from functions_ml import *

# sqlite database structure is following:
#sqlite> .header on
#sqlite> .mode column
#sqlite> select * from alpaca_websocket_stream_data LIMIT 10;
#timestamp                            symbol  price   size  exchange  conditions  tape  id   
#-----------------------------------  ------  ------  ----  --------  ----------  ----  -----
#2022-07-19 15:49:25.477387108-04:00  AAPL    150.8   100   V         ['@']       C     10807
#2022-07-19 15:49:27.252579851-04:00  AAPL    150.81  3     V         ['@', 'I']  C     10808
#2022-07-19 15:49:27.252579851-04:00  AAPL    150.81  100   V         ['@']       C     10809
#2022-07-19 15:49:27.666163652-04:00  AAPL    150.81  100   V         ['@']       C     10810
#2022-07-19 15:49:27.666164795-04:00  AAPL    150.81  200   V         ['@']       C     10811
#2022-07-19 15:49:29.248316808-04:00  AAPL    150.79  100   V         ['@']       C     10812
#2022-07-19 15:49:32.963910211-04:00  AAPL    150.78  35    V         ['@', 'I']  C     10813
#2022-07-19 15:49:36.611092454-04:00  AAPL    150.77  2     V         ['@', 'I']  C     10814
#2022-07-19 15:49:36.612940345-04:00  AAPL    150.77  100   V         ['@']       C     10815
#2022-07-19 15:49:37.083678369-04:00  AAPL    150.76  100   V         ['@']       C     10816
#sqlite> 


# ------------------- variables -------------------
symbol = 'AAPL'
db_name = 'alpaca_websocket_stream_data.db'
table_name= 'alpaca_websocket_stream_data'
granularity = '1Min'


   

# load classifier, no need to initialize the loaded_rf
loaded_clf = joblib.load("./random_forest.joblib")
clf = loaded_clf






# main sequence:        
data =  get_ticker_data_from_db(symbol, db_name, table_name)
df_res =  resample_data(data, granularity=granularity)

df_res = compute_technical_indicators(df_res)
df_res = compute_features(df_res)
df_res =define_target_condition(df_res)

# streamline for pred and viz
df_res_cut = df_res.iloc[-202:].copy()
predict_timeseries(df_res_cut, clf)

