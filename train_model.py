# trains Random forest classifier  
# on yahoo finance stock historical data
# it learns from data on daily granularity, but this should generalize well to shorter time periods
# since we are evaluating trend behaviour

import datetime

import talib as ta
import yfinance as yf
import pandas as pd
import sqlite3

# custom function imports
from functions_gen import *        # general functions
from functions_ml import *         # machine learning
from functions_viz import *        # visualization
from functions_db import *         # database


# custom indicators moved to modules
from functions_superjump import *
from functions_HHLL import *
from functions_HHLL_conf import *
from functions_HHLL_channel import *
from functions_gator import *

# Random Forest specific functions
from functions_forest import *




# training stock data
tickers = ['SPY', 'F', 'IBM', 'GE', 'AAPL', 'ADM'] 

# other optional tickers for more training:
#           'XOM', 'GM','MMM','KO','PEP','SO','GS',           
#           'HAS','PEAK','HPE','HLT','HD','HON','HRL','HST','HPQ','HUM','ILMN',  
#           'INTC','ICE','INTU','ISRG','IVZ','IRM','JNJ','JPM','JNPR','K','KMB', 
#           'KIM', 'KMI','KSS','KHC', 'KR',  'LB', 'LEG', 'LIN', 'LMT','LOW', 
#           'MAR', 'MA','MCD','MDT', 'MRK', 'MET', 'MGM', 'MU','MSFT', 'MAA', 
#           'MNST', 'MCO','MS', 'MSI',
#           'MMM', 'ABT','ACN','ATVI','ADBE','AMD','A','AKAM','ARE','GOOG','AMZN','AAL', 
#           'AMT', 'AMGN','AIV','AMAT','ADM', 'AVB','BAC', 'BBY', 'BIIB',  'BLK', 'BA','BXP', 
#           'BMY', 'AVGO','CPB','COF','CAH', 'CCL', 'CAT', 'CBOE', 'CBRE','CNC', 'CNP', 'SCHW','CVX', 
#           'CMG', 'CI','CSCO','C','CLX', 'CME', 'KO',  'CTSH', 'CL',  'CMCSA',  'ED', 'COST','CCI', 
#           'CVS', 'DAL','DLR', 'D','DPZ', 'DTE', 'DUK', 'DRE', 'EBAY', 'EA', 'EMR', 'ETR', 'EFX', 'EQIX', 
#           'EQR', 'ESS', 'EL','EXC', 'EXPE','XOM', 'FFIV','FB','FRT', 'FDX', 'FE','GPS', 'GRMN',   
#           'IT', 'GD',  'GE','GIS', 'GM','GS',  'GWW', 'HAL'
#          ]


def training_sequence(tickers, interval="1m", model_name="./random_forest.joblib"):
    # initiates training sequence for random forest classifier

    for ticker in tickers:
        print('ticker: ', ticker)
        df = get_data(ticker, interval)
        #plot_train_data(df, ticker)


        #print(df)

        # custom indicator extension:
        # create extra features from new indicators into new dfs
        # and then join the dfs based on minute datetime with original df
        # our model also needs 1/0 instead of True/False
        # thrend_conf col needs conversion from 'u','d' to 1,0
        out_df1 = superjumpTBB(df)      # superjumpTBB
        out_df1.replace({False: 0, True: 1}, inplace=True)

        out_df2 = HHLL_Strategy(df)  # HHHL indicator
        out_df2.replace({False: 0, True: 1}, inplace=True)

        out_df3 = HHLL_confirmation(df)  # HHHL indicator
        out_df3.replace({'d': 0, 'u': 1, 'none': -1}, inplace=True)

        out_df4 = HHLL_Channel(df)
        
        out_df5 = rsi_strategy(df) # RSI gator indicator


        df = compute_technical_indicators(df)
        df = compute_features(df)
        df = define_target_condition(df)

        # TODO, verify that inner join is what we really need
        # merging with new dataframes with custom indicators
        df = pd.merge(df, out_df1, how='inner', on='Date')
        df = pd.merge(df, out_df2, how='inner', on='Date')
        df = pd.merge(df, out_df3, how='inner', on='Date')
        df = pd.merge(df, out_df4, how='inner', on='Date')
        df = pd.merge(df, out_df5, how='inner', on='Date')

        #print('regular df')
        #print(df)

        clf = splitting_and_training(df)

        save_model(clf, model_name)
        
        # commenting out saves time during training
        #df = predict_timeseries(df, clf)
        #plot_stock_prediction(df, ticker)

    return None




training_sequence(tickers, interval="1m", model_name="./random_forest.joblib")
