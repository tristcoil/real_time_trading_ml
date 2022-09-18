import ta
# import ccxt
import yfinance as yf
import pandas as pd
import datetime

def crossover(x, y, i):
    if i == 0: return False
    return i > 0 and x[i] > y[i] and x[i-1] < y[i-1]

def crossunder(x, y, i):
    if i == 0: return False
    return i > 0 and x[i] < y[i] and x[i-1] > y[i-1]

def get_crypto_data(symbol="BTCUSDT", timeframe="1d", exchange="binance", since=None, limit=500):
    ex = getattr(ccxt, exchange)(); _ = ex.load_markets()
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=500)
    df = pd.DataFrame.from_records(data, columns=['date', "open", "high", "low", "close", "volume"])
    df.date = df.date.apply(lambda x: datetime.datetime.utcfromtimestamp(x//1000))
    return df

def rsi_strategy(df):
    close = df.Close
    overBought = 70
    overSold = 30
    jawPeriods = 5
    jawOffset = 0
    teethPeriods = 13
    teethOffset = 0
    lipsPeriods = 34
    lipsOffset = 0
    filterCross = False

    jaws = ta.momentum.rsi(close, jawPeriods).fillna(0)
    teeth = ta.momentum.rsi(close, teethPeriods).fillna(0)
    lips = ta.momentum.rsi(close, lipsPeriods).fillna(0)
    signal = []
    for i in range(len(close)):
        if filterCross:
            LONG_SIGNAL_BOOLEAN  = (crossover(teeth, lips, i) and jaws[i] > teeth[i]) 
        else: 
            LONG_SIGNAL_BOOLEAN  = (crossover(teeth, lips, i) and jaws[i] > lips[i]) or (crossover(jaws, lips, i) and teeth[i] > lips[i])
        if filterCross:
            SHORT_SIGNAL_BOOLEAN = (crossunder(teeth, lips, i) and jaws[i] < teeth[i]) 
        else:
            SHORT_SIGNAL_BOOLEAN = (crossunder(teeth, lips, i) and jaws[i] < lips[i]) or (crossunder(jaws, lips, i) and teeth[i] < lips[i])

        signal.append( 1 if LONG_SIGNAL_BOOLEAN else -1 if SHORT_SIGNAL_BOOLEAN else 0)
    
    return pd.DataFrame({'date': df.date,
                         'jaws': jaws,
                         'teeth': teeth,
                         'lips': lips,
                         'signal': signal})


if __name__ == "__main__":
    # df = get_crypto_data(symbol="BTCUSDT", timeframe='1h')
    df = yf.download("SPY", interval="1h", period='1y')
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True)
    out_df = rsi_strategy(df)
    print(out_df.tail())
    out_df.to_csv("rsi_strategy.csv", index=False)
    print("Saving results to rsi_strategy.csv")



