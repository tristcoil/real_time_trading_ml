import numpy as np
import pandas as pd


def highest(data, length):
    return np.array(
        [0] * length
        + [np.max(data[i - length + 1 : i + 1]) for i in range(length, len(data))]
    )


def lowest(data, length):
    return np.array(
        [0] * length
        + [np.min(data[i - length + 1 : i + 1]) for i in range(length, len(data))]
    )


def HHLL_Channel(df):
    close, high, low = df.Close, df.High, df.Low
    length = 20
    reverse = False
    hh = highest(high, length)
    ll = lowest(low, length)

    pos, pos_prv = 0, 0
    possig = [0]
    for i in range(1, len(close)):
        iff_1 = -1 if close[i] < ll[i - 1] else pos_prv
        pos = 1 if close[i] > hh[i - 1] else iff_1
        pos_prv = pos
        iff_2 = 1 if reverse and pos == -1 else pos
        possig.append(-1 if reverse and pos == 1 else iff_2)

    return pd.DataFrame(
        {
            "Date": df.Date,
            # 1 for long -1 for short, 0 is additional state, assuming for none
            "HHLL_channel_sig": possig,
        }
    )
