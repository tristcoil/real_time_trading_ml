
# split long list of strings into chunks of size n
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

lst = [1,2,3,4,5,6,7,8,9,10,11]


sub_lst = list(chunks(lst, 3))
print(sub_lst)
print(len(sub_lst))

# iterate over sub_lst and print each sublist
for sub_lst in sub_lst:
    print(sub_lst)
    print(len(sub_lst))
    print("\n")

# get AAPL price data OHLC with yfinance
# save to pandas dataframe
import yfinance as yf
import pandas as pd

aapl = yf.Ticker("AAPL")
aapl_data = aapl.history(period="1d")
print(aapl_data)



