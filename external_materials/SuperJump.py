import pandas as pd 
import numpy as np
import ccxt
import ta
from datetime import datetime

def getData(symbol= "BTC/USDT", interval="15m"):
    exchange = getattr(ccxt, 'binance')(); _ = exchange.load_markets()
    data = exchange.fetch_ohlcv(symbol, timeframe=interval)
    df = {}
    for i, col in enumerate(['date','open','high','low','close','volume']):
        df[col] = []
        for row in data:
            if col == 'date':
                df[col].append(datetime.utcfromtimestamp(row[i]/1000))
            else:
                df[col].append(row[i])
    return  pd.DataFrame(df)

def stdev(src, length):
    out = [0]*length
    a = ta.trend.sma_indicator(np.power(src,2),length).fillna(0)
    for i in range(length, len(src)):
        b = np.power(np.sum(src[i-length+1:i+1]),2)/length**2
        out.append(np.sqrt(abs(a.iat[i] - b)))
    return pd.Series(out)

def rma(src, length):
    src = np.array(src)
    alpha = 1/length
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = len(src)
    
    r = np.arange(n)
    scale_arr = scale**r
    offset = src[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = src*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

def tr(high, low, close):
    out = [0]
    for i in range(1, len(close)):
        out.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    return out

def crossunder(x, y, i):
    return i > 0 and x[i] < y[i] and x[i-1] > y[i-1]

def crossover(x, y, i):
    return i > 0 and x[i] > y[i] and x[i-1] < y[i-1]

def indicator(df):
    open_, close, high, low = df.open, df.close, df.high, df.low
    length = 60
    src = close
    mult = 2.0
    b_mult = 2.5


    atrlength = 14
    smoothing = "RMA"  # options=["RMA", "SMA", "EMA", "WMA"]
    SLAtr = 1.0

    basis = ta.trend.sma_indicator(src, length).fillna(0)

    dev = mult * stdev(src, length)
    upper = basis + dev
    lower = basis - dev

    b_dev = b_mult * stdev(src, length)
    b_upper = basis + b_dev
    b_lower = basis - b_dev


    def ma_function(source, length, smoothing="RMA"):
        if smoothing == "RMA": 
            return rma(source, length)
        elif smoothing == "SMA": 
            return ta.trend.sma_indicator(source, length)
        elif smoothing == "EMA": 
            return ta.trend.ema_indicator(source, length)
        else: 
            return ta.trend.wma_indicator(source, length)

    atr = ma_function(tr(high, low, close), atrlength)

    LongSig = [crossunder(lower,src, i) and close[i] > open_[i] for i in range(len(src))]
    ShortSig = [crossover(upper,src, i) and close[i] < open_[i] for i in range(len(src))]
    WLongSig = [crossunder(b_lower,src, i) and close[i] > open_[i] for i in range(len(src))]
    WShortSig = [crossover(b_upper,src, i) and close[i] < open_[i] for i in range(len(src))]

    return pd.DataFrame({"date": df.date,
                         "Basis": basis,
                         "Upper": upper,
                         "Lower": lower,
                         "Wide Upper": b_upper,
                         "Wide Lower": b_lower,
                         "SL Upper": b_upper + atr*SLAtr,
                         "SL Lower": b_lower - atr*SLAtr,
                         "LongSig": LongSig,
                         "ShortSig": ShortSig,
                         "WLongSig": WLongSig, 
                         "WShortSig": WShortSig
                        })


if __name__ == "__main__":
    symbol = "BTC/USDT"
    interval = "1d"
    df = getData(symbol=symbol, interval=interval)
    
    out_df = indicator(df)
    print(out_df.tail())
    out_df.to_csv("./SuperJump_TBBB.csv", index=False)
    print("Results Saved to SuperJump_TBBB.csv")