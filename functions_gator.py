import numpy as np
import pandas as pd
import ta

# RSI Gator indicator

def gator_crossover(x, y, i):
    if i == 0: return False
    return i > 0 and x[i] > y[i] and x[i-1] < y[i-1]


def gator_crossunder(x, y, i):
    if i == 0: return False
    return i > 0 and x[i] < y[i] and x[i-1] > y[i-1]


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
            LONG_SIGNAL_BOOLEAN  = (gator_crossover(teeth, lips, i) and jaws[i] > teeth[i]) 
        else: 
            LONG_SIGNAL_BOOLEAN  = (gator_crossover(teeth, lips, i) and jaws[i] > lips[i]) or (gator_crossover(jaws, lips, i) and teeth[i] > lips[i])
        if filterCross:
            SHORT_SIGNAL_BOOLEAN = (gator_crossunder(teeth, lips, i) and jaws[i] < teeth[i]) 
        else:
            SHORT_SIGNAL_BOOLEAN = (gator_crossunder(teeth, lips, i) and jaws[i] < lips[i]) or (gator_crossunder(jaws, lips, i) and teeth[i] < lips[i])

        signal.append( 1 if LONG_SIGNAL_BOOLEAN else -1 if SHORT_SIGNAL_BOOLEAN else 0)
    
    return pd.DataFrame({'Date': df.Date,
                         'jaws': jaws,
                         'teeth': teeth,
                         'lips': lips,
                         'rsi_gator_sig': signal})


