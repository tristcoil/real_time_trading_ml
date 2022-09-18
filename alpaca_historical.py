import logging
import time
import asyncio

from configparser import ConfigParser

#from distutils.errors import LinkError
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream

from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream

import sqlite3
import datetime


# --- PREP STEPS ---
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
config = ConfigParser()
config.read("config.ini")
API_KEY = config.get("alpaca", "API_KEY")
SECRET_KEY = config.get("alpaca", "SECRET_KEY")

print(f"API_KEY:    ", API_KEY)
print(f"SECRET_KEY: ", SECRET_KEY)


# ------------------------------------------------------------------------
# HISTORICAL DATA
# docs:       https://github.com/alpacahq/alpaca-trade-api-python
# parameters: https://alpaca.markets/docs/api-references/market-data-api/stock-pricing-data/historical/#bars
# note: free stock subscription allows to get only data older than 15 minutes from now
# time has to be in RFC 3339 format 2022-09-17T00:00:00.52Z, get it like now.isoformat()
# when we specify just limit=10000 candles, it returns empty df
# ------------------------------------------------------------------------


def get_alpaca_hist_data(symbol, data_type):
    # function downloads minute data from alpaca api
    # examples
    # symbol = "AAPL"
    # data_type = "stocks" / "crypto"

    # example of crypto dataframe
    #                          exchange      open      high       low     close     volume  trade_count          vwap
    #timestamp                                                                                                       
    #2022-09-18 05:00:00+00:00     CBSE  20012.03  20013.17  20001.79  20002.47   1.726999          183  20007.886639
    #2022-09-18 05:00:00+00:00     FTXU  20007.00  20007.00  20007.00  20007.00   0.040800            1  20007.000000
    #2022-09-18 05:01:00+00:00     CBSE  20002.49  20003.71  19997.00  20002.77   0.821499          184  20000.362906
    #2022-09-18 05:01:00+00:00     FTXU  20001.00  20001.00  20001.00  20001.00   1.002000            4  20001.000000
    #2022-09-18 05:02:00+00:00     CBSE  20001.56  20010.79  20001.56  20008.20   1.060122          125  20007.932954


    # --- PREP STEPS ---
    # pick api data source
    #rest_api = REST(API_KEY, SECRET_KEY, "https://paper-api.alpaca.markets")     # paper trading data source
    #rest_api = REST(API_KEY, SECRET_KEY, "https://api.alpaca.markets")           # live trading API
    rest_api = REST(API_KEY, SECRET_KEY, "https://data.alpaca.markets")          # data endpoint


    # CRYPTO
    if data_type == "crypto":
        # Retrieve daily bar data for Bitcoin in a DataFrame
        #btc_bars = rest_api.get_crypto_bars("BTCUSD", TimeFrame.Minute, "2022-09-10", "2022-09-18").df
        df = rest_api.get_crypto_bars(symbol=symbol, timeframe=TimeFrame.Minute, limit=10000).df   

        #print("API data:")
        #print(df)

        #print(set(df.exchange.tolist()))     # currently there are 3 exchanges: {'FTXU', 'CBSE', 'ERSX'}
        # CBSE looks to have big trading volume

        # display only rows from df dataframe that contain exchange column equal to CBSE
        #df = df[df['exchange'] == 'CBSE']
        #print(df)

        # other useful api options
        # Quote and trade data are also available for cryptocurrencies
        # btc_quotes = rest_api.get_crypto_quotes('BTCUSD', '2021-01-01', '2021-01-05').df
        # btc_trades = rest_api.get_crypto_trades('BTCUSD', '2021-01-01', '2021-01-05').df


    # STOCKS
    if data_type == "stocks":

        #stock_bars = rest_api.get_bars("AAPL", TimeFrame.Minute, start="2022-09-10", end="2022-09-17").df
        #stock_bars = rest_api.get_bars(symbol="AAPL", timeframe=TimeFrame.Minute, start="2022-09-10", end="2022-09-17T00:00:00.52Z", limit=10000).df

        now = datetime.datetime.now(datetime.timezone.utc)
        now.isoformat()

        end = now - datetime.timedelta(minutes=15.1)   # a bit more than 15 mins back
        start = now - datetime.timedelta(days=10)      # get minute data few days back

        #stock_bars = rest_api.get_bars(symbol="AAPL", timeframe=TimeFrame.Minute, start="2022-09-10", end="2022-09-17T16:49:57.055823+00:00", limit=10000).df
        df = rest_api.get_bars(symbol=symbol, timeframe=TimeFrame.Minute, start=start.isoformat(), end=end.isoformat(), limit=10000).df

        #print("Stock API data:")
        #print(df)

    # make the alpaca df look similar to df from Yahoo Finance
    # rename df columns "timestamp", 'open', 'high', 'low', 'close', 'volume' to "Date", 'Open', 'High', 'Low', 'Close', 'Volume'
    df.reset_index(inplace=True)
    df = df.rename(columns={"timestamp": "Date", 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    print("FINAL DATAFRAME: ")
    print(df)

    #FINAL DATAFRAME: 
    #                      Date      Open      High     Low     Close  Volume
    #0    2022-09-08 18:32:00+00:00  154.0200  154.1100  153.99  154.1000  150359
    #1    2022-09-08 18:33:00+00:00  154.0907  154.2000  154.09  154.1500  170388
    #2    2022-09-08 18:34:00+00:00  154.1550  154.2100  154.12  154.1550  102371
    #3    2022-09-08 18:35:00+00:00  154.1500  154.2199  154.06  154.1199  146318
    #4    2022-09-08 18:36:00+00:00  154.1200  154.2181  154.10  154.2100   97022

    return df





# ------------------------------------------------------------------------
# WEBSOCKET
# ------------------------------------------------------------------------


# --- DB FUNCTION DEFINITIONS ---


# --- FUNCTION DEFINITIONS ---




if __name__ == "__main__":
    
    get_alpaca_hist_data("BTCUSD", "crypto")
    get_alpaca_hist_data("AAPL", "stocks")


















