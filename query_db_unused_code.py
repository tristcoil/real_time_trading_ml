
# read data from sql database
conn = sqlite3.connect("alpaca_websocket_stream_data.db")
c = conn.cursor()
c.execute("SELECT * FROM alpaca_websocket_stream_data")
print(c.fetchall())
conn.close()





'''

# convert unix timestamps from nanoseconds to date
import datetime
import time

timestamp = 1657828845257527097
print(datetime.datetime.fromtimestamp(timestamp / 1000000000.0))
print(datetime.datetime.fromtimestamp(timestamp / 1000000000.0).strftime("%Y-%m-%d %H:%M:%S"))
print(datetime.datetime.fromtimestamp(timestamp / 1000000000.0).strftime("%Y-%m-%d %H:%M:%S").split(" ")[0])
print(datetime.datetime.fromtimestamp(timestamp / 1000000000.0).strftime("%Y-%m-%d %H:%M:%S").split(" ")[1])

'''




'''
# pandas to onnect to database and aggregate price data to 1 minute granularity in pandas
import pandas as pd
import sqlite3

conn = sqlite3.connect("alpaca_websocket_stream_data.db")
c = conn.cursor()
c.execute("SELECT * FROM alpaca_websocket_stream_data")
data = c.fetchall()
df = pd.DataFrame(data, columns=["timestamp", "symbol", "price", "size", "exchange", "conditions", "tape", "id"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns")
df["date"] = df["timestamp"].dt.date
df["time"] = df["timestamp"].dt.time
df["time"] = df["time"].apply(lambda x: x.strftime("%H:%M:%S"))
df["time"] = df["date"] + " " + df["time"]
df["time"] = pd.to_datetime(df["time"])
df = df.drop(columns=["timestamp"])
df = df.drop_duplicates(subset=["time", "symbol"])
df = df.set_index("time")
df = df.groupby(["symbol"]).mean()
df = df.reset_index()
df = df.sort_values(by=["symbol"])
df.to_csv("alpaca_websocket_stream_data_aggregated_to_1_minute.csv")
conn.close()


'''

resampled data to new dataframe
df_resampled = df.set_index("timestamp").resample("1Min").ohlc()
df_resampled.dropna(inplace=True)
df_resampled.reset_index(inplace=True)
df_resampled.drop(columns=["level_0"], inplace=True)
df_resampled.rename(columns={"timestamp": "time"}, inplace=True)



