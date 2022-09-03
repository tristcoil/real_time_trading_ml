import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def pivot_points(data, lb=5, rb=5, ph=False):
    src = data.High if ph else data.Low
    out = [np.nan] * (lb + rb)
    for x in range(lb + rb, len(src)):
        p, isFound = src[x - rb], True
        for i in range(x - lb - rb, x + 1):
            if i == x - rb:
                continue
            isFound = False if ph and src[i] > p else isFound
            isFound = False if not ph and src[i] < p else isFound
        out.append(src[x - rb] if isFound else np.nan)
    return out


def findprevious(hl, zz, i):
    loc1, loc2, loc3, loc4 = 0.0, 0.0, 0.0, 0.0

    def find(hl, zz, ehl, start):
        loc, xx = 0.0, 0
        for x in range(start, 10001):
            if i - x >= 0 and hl[i - x] == ehl and not np.isnan(zz[i - x]):
                loc = zz[i - x]
                xx = x + 1
                break
        return loc, xx

    xx = 0
    ehl = -1 if hl[i] == 1 else 1
    loc1, xx = find(hl, zz, ehl, xx)
    ehl = hl[i]
    loc2, xx = find(hl, zz, ehl, xx)
    ehl = -1 if hl[i] == 1 else 1
    loc3, xx = find(hl, zz, ehl, xx)
    ehl = hl[i]
    loc4, xx = find(hl, zz, ehl, xx)

    return loc1, loc2, loc3, loc4


def HHLL_Strategy(df, lb=5, rb=5, showsupres=True):

    ph = pivot_points(df, lb, rb, True)
    pl = pivot_points(df, lb, rb, False)

    prv_zz, prv_hl = np.nan, np.nan
    _hh, _ll, _hl, _lh = [], [], [], []
    hl, zz = [], []
    res, sup, trend = [], [], []

    prv_res, prv_sup, prv_trend = 0, 0, 0
    a, b, c, d, e = np.nan, np.nan, np.nan, np.nan, np.nan
    for i in range(0, len(ph)):
        zz_ = ph[i] if not np.isnan(ph[i]) else pl[i] if not np.isnan(pl[i]) else np.nan
        hl_ = 1 if not np.isnan(ph[i]) else -1 if not np.isnan(pl[i]) else np.nan

        zz_ = (
            np.nan
            if not np.isnan(pl[i]) and hl_ == -1 and prv_hl == -1 and pl[i] > prv_zz
            else zz_
        )
        zz_ = (
            np.nan
            if not np.isnan(ph[i]) and hl_ == 1 and prv_hl == 1 and ph[i] < prv_zz
            else zz_
        )

        hl_ = np.nan if hl_ == -1 and prv_hl == 1 and zz_ > prv_zz else hl_
        hl_ = np.nan if hl_ == 1 and prv_hl == -1 and zz_ < prv_zz else hl_
        zz_ = np.nan if np.isnan(hl_) else zz_

        prv_hl, prv_zz = (
            hl_ if not np.isnan(hl_) else prv_hl,
            zz_ if not np.isnan(zz_) else prv_zz,
        )

        hl.append(hl_)
        zz.append(zz_)

        if not np.isnan(hl[-1]):
            b, c, d, e = findprevious(hl, zz, i)
            a = zz[-1]

        _hh.append(not np.isnan(zz[-1]) and (a > b and a > c and c > b and c > d))
        _ll.append(not np.isnan(zz[-1]) and (a < b and a < c and c < b and c < d))
        _hl.append(
            not np.isnan(zz[-1])
            and (
                (a >= c and (b > c and b > d and d > c and d > e))
                or (a < b and a > c and b < d)
            )
        )
        _lh.append(
            not np.isnan(zz[-1])
            and (
                (a <= c and (b < c and b < d and d < c and d < e))
                or (a > b and a < c and b > d)
            )
        )

        res_ = zz[-1] if _lh[-1] else prv_res
        sup_ = zz[-1] if _hl[-1] else prv_sup

        trend_ = 1 if df.Close[i] > res_ else -1 if df.Close[i] < sup_ else prv_trend

        res_ = (
            zz[-1] if (trend_ == 1 and _hh[-1]) or (trend_ == -1 and _lh[-1]) else res_
        )
        sup_ = (
            zz[-1] if (trend_ == 1 and _hl[-1]) or (trend_ == -1 and _ll[-1]) else sup_
        )

        prv_res, prv_sup, prv_trend = res_, sup_, trend_
        trend.append(trend_)
        res.append(res_)
        sup.append(sup_)

    # adding offset of -rb as in pinescript plotshape function
    # plotshape(_hl, text="HL", title="Higher Low", style=shape.labelup, color=color.lime, textcolor=color.black, location=location.belowbar, "offset = -rb")

    def offset(x, rb):
        return x[rb:] + [False] * rb

    _hh = offset(_hh, rb)
    _ll = offset(_ll, rb)
    _hl = offset(_hl, rb)
    _lh = offset(_lh, rb)

    return pd.DataFrame(
        {
            "Date": df.Date,
            "HH": _hh,
            "LL": _ll,
            "HL": _hl,
            "LH": _lh,
            "support": sup,
            "resistance": res,
            "trend": trend,
        }
    )


if __name__ == "__main__":

    data = yf.download("SPY", interval="1d")
    data["Date"] = data.index
    data.reset_index(drop=True, inplace=True)

    out_df = HHLL_Strategy(data)

    print("Saving output to HHLL_Strategy.csv")
    out_df.to_csv("./HHLL_Strategy.csv", index=False)
