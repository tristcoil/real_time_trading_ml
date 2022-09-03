import pandas as pd


def isHigh_Low(close, open_, bar, high=True):
    flag = False
    if bar < 11:
        return flag
    if close[bar] < open_[bar] and high or close[bar] > open_[bar] and not high:
        for i in range(bar - 1, bar - 11, -1):
            if close[i] == open_[i]:
                continue
            elif close[i] > open_[i] and high:
                flag = True
                break
            elif close[i] < open_[i] and not high:
                flag = True
                break
            else:
                break
    return flag


def HHLL_confirmation(df):
    open_, close, high, low = df.Open, df.Close, df.High, df.Low
    hBar, hPrice, hCPrice, prevHBar, prevHPrice, prevHCPrice = [0.0] * 6
    lBar, lPrice, lCPrice, prevLBar, prevLPrice, prevLCPrice = [0.0] * 6

    hHigh, lLow = high, low

    firstLBar, firstHBar, firstLPrice, firstHPrice = [0] * 4
    trend = "none"
    lastTrend = trend
    Trend = [trend]

    latestLBar, latestHBar, latestLPrice, latestHPrice = [0], [0], [0], [0]
    firstHBar_, firstHPrice_, firstLBar_, firstLPrice_ = [0], [0], [0], [0]

    for bar_index in range(1, len(close)):
        if isHigh_Low(close, open_, bar_index, high=True):
            prevHBar = hBar
            prevHPrice = hPrice
            prevHCPrice = hCPrice
            hBar = bar_index
            hPrice = hHigh[bar_index]
            hCPrice = close[bar_index - 1]

        if isHigh_Low(close, open_, bar_index, high=False):
            prevLBar = lBar
            prevLPrice = lPrice
            prevLCPrice = lCPrice
            lBar = bar_index
            lPrice = lLow[bar_index]
            lCPrice = close[bar_index - 1]

        if (
            prevLPrice < lPrice
            and prevLCPrice < lCPrice
            and hPrice < high[bar_index]
            and hCPrice < close[bar_index]
        ):
            if lastTrend != "u":
                firstLBar = prevLBar
                firstLPrice = prevLPrice
            trend = "u"
        elif (
            prevHPrice > hPrice
            and prevHCPrice > hCPrice
            and lPrice > low[bar_index]
            and close[bar_index]
        ):
            if lastTrend != "d":
                firstHBar = prevHBar
                firstHPrice = prevHPrice
            trend = "d"
        elif (
            trend == "u"
            and (
                lCPrice > close[bar_index]
                or (
                    isHigh_Low(close, open_, bar_index, high=True)
                    and prevHCPrice > hCPrice
                )
            )
        ) or (
            trend == "d"
            and (
                hCPrice < close[bar_index]
                or (
                    isHigh_Low(close, open_, bar_index, high=False)
                    and prevLCPrice < lCPrice
                )
            )
        ):
            trend = "none"

        firstHBar_.append(firstHBar)
        firstHPrice_.append(firstHPrice)
        firstLBar_.append(firstLBar)
        firstLPrice_.append(firstLPrice)

        Trend.append(trend)

        if trend == "u" and (
            (lPrice - firstLPrice) / (lBar - firstLBar + 1e-7)
            < (latestLPrice[-1] - firstLPrice) / (latestLBar[-1] - firstLBar + 1e-7)
            or prevLBar == firstLBar
        ):
            latestLBar.append(lBar)
            latestLPrice.append(lPrice)
        else:
            latestLBar.append(latestLBar[-1])
            latestLPrice.append(latestLPrice[-1])

        if trend == "d" and (
            (hPrice - firstHPrice) / (hBar - firstHBar + 1e-7)
            > (latestHPrice[-1] - firstHPrice) / (latestHBar[-1] - firstHBar + 1e-7)
            or prevHBar == firstHBar
        ):
            latestHBar.append(hBar)
            latestHPrice.append(hPrice)
        else:
            latestHBar.append(latestHBar[-1])
            latestHPrice.append(latestHPrice[-1])

    return pd.DataFrame(
        {
            "Date": df.Date,
            "trend_conf": Trend,
            "firstLBar": firstLBar_,
            "firstLPrice": firstLPrice_,
            "latestLBar": latestLBar,
            "latestLPrice": latestLPrice,
            "firstHBar": firstHBar_,
            "firstHPrice": firstHPrice_,
            "latestHBar": latestHBar,
            "latestHPrice": latestHPrice,
        }
    )
