import sqlite3



def get_ticker_data_from_db(symbol, db_name, table_name):
    # connect to sqlite database and get all data where symbol is AAPL for example
    # symbol, database name, table name are external variables

    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    # c.execute("SELECT * FROM ? WHERE symbol = ?", (table_name, symbol,))
    c.execute(f"SELECT * FROM {table_name} WHERE symbol = ?", (symbol,))
    data = c.fetchall()
    conn.close()

    return data


def get_ticker_data_from_db_days_back(symbol, db_name, table_name):
    # load data n days back from db
    # connect to sqlite database and get all data where symbol is AAPL for example
    # timestamp is from 24 hours ago to now
    # symbol, database name, table name are external variables

    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute(
        f"SELECT * FROM  {table_name} WHERE symbol = ? AND timestamp BETWEEN datetime('now', '-7 days') AND datetime('now')",
        (symbol,),
    )
    data = c.fetchall()
    conn.close()

    return data





