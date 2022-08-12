import sqlite3


data = {
    "conditions": "some condition",
    "exchange": "V",
    "id": 58734536583767,
    "price": 378.13,
    "size": 100,
    "symbol": "SPY",
    "tape": "B",
    "timestamp": 1657828845257527097,
}


def save_websocket_stream_data_to_sqlite(data):
    # function saves data to sqlite database
    # data is a dictionary
    # data['timestamp'] is a timestamp
    # data['symbol'] is a symbol
    # data['price'] is a price
    # data['size'] is a size
    # data['exchange'] is a exchange
    # data['conditions'] is a conditions
    # data['tape'] is a tape
    # data['id'] is an id

    # create a database if it does not exist
    conn = sqlite3.connect("alpaca_websocket_stream_data.db")
    c = conn.cursor()
    # create table if it does not exist
    c.execute(
        """CREATE TABLE IF NOT EXISTS alpaca_websocket_stream_data
                (timestamp TEXT, symbol TEXT, price REAL, size INTEGER, exchange TEXT, conditions TEXT, tape TEXT, id TEXT)"""
    )
    # insert data
    c.execute(
        "INSERT INTO alpaca_websocket_stream_data VALUES (?,?,?,?,?,?,?,?)",
        (
            data["timestamp"],
            data["symbol"],
            data["price"],
            data["size"],
            data["exchange"],
            data["conditions"],
            data["tape"],
            data["id"],
        ),
    )
    # commit changes
    conn.commit()
    # close connection
    conn.close()



# save data dictionary to sql database
save_websocket_stream_data_to_sqlite(data)




