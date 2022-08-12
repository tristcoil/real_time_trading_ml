# trains Random forest classifier  
# on yahoo finance stock historical data
# it learns from data on daily granularity, but this should generalize well to shorter time periods
# since we are evaluating trend behaviour

import datetime

from functions_ml import *


start_time = datetime.datetime(1980, 1, 1)
#end_time = datetime.datetime(2019, 1, 20)
end_time = datetime.datetime.now().date().isoformat()         # today


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


for ticker in tickers:

    df = get_data(ticker, start_time, end_time)
    #plot_train_data(df)   # plot training data
    df = compute_technical_indicators(df)
    df = compute_features(df)
    df=define_target_condition(df)

    clf = splitting_and_training(df)

    save_model(clf)
    
    # commenting out saves time during training
    #df = predict_timeseries(df, clf)
    #plot_stock_prediction(df, ticker)





