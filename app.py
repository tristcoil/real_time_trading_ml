import logging
import time
import asyncio

from configparser import ConfigParser

from distutils.errors import LinkError
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream

from alpaca_trade_api.common import URL
from alpaca_trade_api.stream import Stream

import sqlite3


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
# ------------------------------------------------------------------------

# rest_api = REST(API_KEY, SECRET_KEY, "https://paper-api.alpaca.markets")

# Retrieve daily bar data for Bitcoin in a DataFrame
# btc_bars = rest_api.get_crypto_bars(
#    "BTCUSD", TimeFrame.Day, "2021-01-01", "2022-01-01"
# ).df

# print("API data:")
# print(btc_bars)

# Quote and trade data are also available for cryptocurrencies
# btc_quotes = rest_api.get_crypto_quotes('BTCUSD', '2021-01-01', '2021-01-05').df
# btc_trades = rest_api.get_crypto_trades('BTCUSD', '2021-01-01', '2021-01-05').df

# from alpaca_trade_api.rest import REST, TimeFrame
# api = REST()
#
# api.get_bars("AAPL", TimeFrame.Hour, "2021-06-08", "2021-06-08", adjustment='raw').df


# ------------------------------------------------------------------------
# WEBSOCKET
# ------------------------------------------------------------------------


# --- DB FUNCTION DEFINITIONS ---


def initiate_db():
    # create a database if it does not exist
    conn = sqlite3.connect("alpaca_websocket_stream_data.db")
    c = conn.cursor()
    # create table if it does not exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS alpaca_websocket_stream_data
                (timestamp INTEGER, symbol TEXT, price REAL, size INTEGER, 
                 exchange TEXT, conditions TEXT, tape TEXT, id INTEGER)"""
    )
    conn.commit()
    conn.close()

    return None


def save_websocket_stream_data_to_sqlite(data):
    # trade looks like:
    # quote Trade({   'conditions': [' ', 'T'],
    #    'exchange': 'V',
    #    'id': 58734536583767,
    #    'price': 378.13,
    #    'size': 100,
    #    'symbol': 'SPY',
    #    'tape': 'B',
    #    'timestamp': 1657828845257527097})

    # so its like a dictionary/class instance
    # we would like to parse it and then save it to sqlite database
    # timestamp is unix epoch in nanoseconds

    # access the object like
    # q.timestamp allows us to get a timestamp in nanoseconds from the quote

    # print('timestamp: ', data.timestamp)

    # create a database if it does not exist
    conn = sqlite3.connect("alpaca_websocket_stream_data.db")
    c = conn.cursor()

    # insert data
    c.execute(
        "INSERT INTO alpaca_websocket_stream_data VALUES (?,?,?,?,?,?,?,?)",
        (
            str(data.timestamp),
            str(data.symbol),
            float(data.price),
            int(data.size),
            str(data.exchange),
            str(data.conditions),
            str(data.tape),
            int(data.id),
        ),
    )
    # commit changes
    conn.commit()
    # close connection
    conn.close()


# --- FUNCTION DEFINITIONS ---

def run_connection(conn):
    try:
        conn.run()
    except KeyboardInterrupt:
        print("Interrupted execution by user")
        loop = asyncio.new_event_loop()
        loop.run_until_complete(conn.stop_ws())
        exit(0)
    except Exception as e:
        print(f"Exception from websocket connection: {e}")
    finally:
        print("Trying to re-establish connection")
        time.sleep(3)
        run_connection(conn)


async def trade_callback(t):
    print("trade", t)
    save_websocket_stream_data_to_sqlite(t)


async def quote_callback(q):
    print("quote", q)
    save_websocket_stream_data_to_sqlite(q)


async def print_trade_update(tu):
    print("trade update", tu)
    save_websocket_stream_data_to_sqlite(tu)


async def print_quote(q):
    print("quote", q)
    save_websocket_stream_data_to_sqlite(q)


if __name__ == "__main__":
    initiate_db()

    # Initiate stream Class Instance
    stream = Stream(
        API_KEY,
        SECRET_KEY,
        base_url=URL("https://paper-api.alpaca.markets"),
        data_feed="iex",
    )  # <- replace to 'sip' if you have PRO subscription

    # Add callbacks to specific ticker symbols
    # stream.subscribe_quotes(print_quote, "AAPL")
    # stream.subscribe_quotes(print_quote, "IBM")
    # stream.subscribe_trades(print_trade_update, "SPY")

    stream.subscribe_trades(print_quote, "SPY")
    stream.subscribe_trades(print_quote, "AAPL")
    stream.subscribe_trades(print_quote, "IBM")

    run_connection(stream)
