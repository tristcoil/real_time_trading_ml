import matplotlib.pyplot as plt
import streamlit as st



def plot_stock_prediction(df, ticker):
    # plot  values and significant levels
    plt.figure(figsize=(20,7))
    plt.title('Predictive model ' + str(ticker))
    plt.plot(df['Date'], df['Adj Close'], label='High', alpha=0.2)

    plt.plot(df['Date'], df['EMA10'], label='EMA10', alpha=0.2)
    plt.plot(df['Date'], df['EMA20'], label='EMA20', alpha=0.2)
    plt.plot(df['Date'], df['EMA30'], label='EMA30', alpha=0.2)
    plt.plot(df['Date'], df['EMA40'], label='EMA40', alpha=0.2)
    plt.plot(df['Date'], df['EMA50'], label='EMA50', alpha=0.2)
    plt.plot(df['Date'], df['EMA100'], label='EMA100', alpha=0.2)
    plt.plot(df['Date'], df['EMA150'], label='EMA150', alpha=0.99)
    plt.plot(df['Date'], df['EMA200'], label='EMA200', alpha=0.2)


    plt.scatter(df['Date'], df['Buy']*df['Adj Close'], label='Buy', marker='^', color='magenta', alpha=0.15)
    #lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    plt.legend()

    plt.show()

    return None 


def plot_stock_prediction_tb(df, ticker, ticks_back):
    # plot stock prediction with ticks back
    # so it will plot subset of the dataframe
    # this is more flexible approach to plotting

    # zoom in
    df = df.iloc[-ticks_back:]   # use eg. 50 for zooming in
    
    # plot  values and significant levels
    plt.figure(figsize=(30,7))
    plt.title('Predictive model ' + str(ticker))
    plt.plot(df['Date'], df['Adj Close'], label='Adj Close', alpha=0.2)

    plt.plot(df['Date'], df['EMA10'], label='EMA10', alpha=0.2)
    plt.plot(df['Date'], df['EMA20'], label='EMA20', alpha=0.2)
    plt.plot(df['Date'], df['EMA30'], label='EMA30', alpha=0.2)
    plt.plot(df['Date'], df['EMA40'], label='EMA40', alpha=0.2)
    plt.plot(df['Date'], df['EMA50'], label='EMA50', alpha=0.2)
    plt.plot(df['Date'], df['EMA100'], label='EMA100', alpha=0.2)
    plt.plot(df['Date'], df['EMA150'], label='EMA150', alpha=0.99)
    plt.plot(df['Date'], df['EMA200'], label='EMA200', alpha=0.2)


    #plt.scatter(df['Date'], df['Buy']*df['Adj Close'], label='Buy', marker='^', color='magenta', alpha=0.15)
    #lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    plt.scatter(
        df['Date'],
        #df["Buy"] * df["Adj Close"],
        df['Long'],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.55,
    )    


    plt.legend()
    plt.show()

    return None




def plot_stock_prediction_zoom(df, ticker, ticks_back):
    # --- plot only Long trades and zoom in on last data ---
    
    # plot  values and significant levels
    #df.reset_index(inplace=True)    
    
    # zoom in
    df = df.iloc[-ticks_back:]   # use eg. 50 for zooming in
    
    plt.figure(figsize=(20, 7))
    plt.title("Predictive model " + str(ticker))
    plt.plot(df.index, df["Adj Close"], label="High", alpha=0.4)

    plt.plot(df.index, df["EMA10"], label="EMA10", alpha=0.2)
    plt.plot(df.index, df["EMA20"], label="EMA20", alpha=0.2)
    plt.plot(df.index, df["EMA30"], label="EMA30", alpha=0.2)
    plt.plot(df.index, df["EMA40"], label="EMA40", alpha=0.2)
    plt.plot(df.index, df["EMA50"], label="EMA50", alpha=0.2)
    plt.plot(df.index, df["EMA100"], label="EMA100", alpha=0.2)
    plt.plot(df.index, df["EMA150"], label="EMA150", alpha=0.79)
    plt.plot(df.index, df["EMA200"], label="EMA200", alpha=0.99)

    
    # workaround with plotting over index

    plt.scatter(
        df.index,
        #df["Buy"] * df["Adj Close"],
        df['Long'],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.55,
    )    
    

    # for stocks on one minute timeframe
    # avoid intraday gaps by overlaying timestamp values over index ticks
    # plt.xticks(df.index, df['Date'], rotation='vertical')
    n=20 # show every n-th x-axis tick on plot
    plt.xticks(df.index[::n], df['Date'][::n], rotation='vertical')


    plt.legend()
    plt.show()

    return None




# ----------------------------------- streamlit specific functions -------------------------------------

def plot_stock_prediction_streamlit(df, ticker):
    # plot  values and significant levels
    fig, ax = plt.subplots()
    # ax.figure(figsize=(20, 7))
    # ax.title("Predictive model " + str(ticker))
    ax.plot(df["Date"], df["Adj Close"], label="High", alpha=0.2)

    ax.plot(df["Date"], df["EMA10"], label="EMA10", alpha=0.2)
    ax.plot(df["Date"], df["EMA20"], label="EMA20", alpha=0.2)
    ax.plot(df["Date"], df["EMA30"], label="EMA30", alpha=0.2)
    ax.plot(df["Date"], df["EMA40"], label="EMA40", alpha=0.2)
    ax.plot(df["Date"], df["EMA50"], label="EMA50", alpha=0.2)
    ax.plot(df["Date"], df["EMA100"], label="EMA100", alpha=0.2)
    ax.plot(df["Date"], df["EMA150"], label="EMA150", alpha=0.99)
    ax.plot(df["Date"], df["EMA200"], label="EMA200", alpha=0.2)

    ax.scatter(
        df["Date"],
        # df["Buy"] * df["Adj Close"],
        df["Long"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    ax.legend()

    # plot matplotlib plt in streamlit
    # st.pyplot(fig)

    return st.pyplot(fig)


def plot_stock_prediction_zoom_streamlit(df, ticker):
    # --- plot only Long trades and zoom in on last data ---

    # plot  values and significant levels

    df = df.iloc[-20:]

    # plot  values and significant levels
    fig, ax = plt.subplots()
    # ax.figure(figsize=(20, 7))
    # ax.title("Predictive model " + str(ticker))
    ax.plot(df["Date"], df["Adj Close"], label="High", alpha=0.2)

    ax.plot(df["Date"], df["EMA10"], label="EMA10", alpha=0.2)
    ax.plot(df["Date"], df["EMA20"], label="EMA20", alpha=0.2)
    ax.plot(df["Date"], df["EMA30"], label="EMA30", alpha=0.2)
    ax.plot(df["Date"], df["EMA40"], label="EMA40", alpha=0.2)
    ax.plot(df["Date"], df["EMA50"], label="EMA50", alpha=0.2)
    ax.plot(df["Date"], df["EMA100"], label="EMA100", alpha=0.2)
    ax.plot(df["Date"], df["EMA150"], label="EMA150", alpha=0.99)
    ax.plot(df["Date"], df["EMA200"], label="EMA200", alpha=0.2)

    ax.scatter(
        df["Date"],
        # df["Buy"] * df["Adj Close"],
        df["Long"],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )
    # lt.scatter(df.index, df['sell_sig'], label='Sell', marker='v')

    ax.legend()

    # plot matplotlib plt in streamlit
    # st.pyplot(fig)

    return st.pyplot(fig)
