import pandas as pd
import numpy as np

df = pd.read_sql_query(
    """SELECT * FROM alpaca_websocket_stream_data""", conn
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["timestamp"] = df["timestamp"].str.split(" ").str[0]
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d")
df = df.groupby(["timestamp", "symbol"]).agg({"price": "mean"}).reset_index()
df = df.sort_values(["timestamp", "symbol"])
df = df.reset_index(drop=True)
df.to_csv("alpaca_websocket_stream_data_aggregated.csv", index=False)

