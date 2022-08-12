import websocket, json, pprint
from csv import writer
from datetime import datetime
import os


API_KEY = "qqqqqqqqqqqqqqqqqqqqqqqqqq"


# get list of tickers from a file, one ticker per line
with open(os.path.join(os.getcwd(), '/home/user/polygon/stock_list.csv'), 'r') as f:
    tickers = [line.split(',')[0] for line in f]

tickers = ['AM.' + str(i) for i in tickers]
tickers_str = ','.join(tickers)
params = tickers_str

def on_open(ws):
    print("opened")
    auth_data = {
        "action": "auth",
        "params": API_KEY
    }

    ws.send(json.dumps(auth_data))

    channel_data = {
        "action": "subscribe",
        "params": params
    }

    ws.send(json.dumps(channel_data))


def on_error(ws, error):
    print(error)


def on_message(ws, message):
    import pandas as pd
    print("received a message")
    # print(message)

    json_message = json.loads(message)
    # print(json_message)
    # pprint.pprint(json_message)
    pprint.pprint(json_message[0])

    # access the payload itself:
    data = json_message[0]
    # print('data is: ', data)

    # how to acces given element in payload, for example closing price
    # close = data['c']
    # print('close is:', close)

    # IMPORTANT
    # convert directly from epoch to normal date
    # divide by 1000 so we get seconds from miliseconds (as the datetime library expects)
    data['e'] = datetime.fromtimestamp(float(data['e']) / 1000.0).strftime('%Y-%m-%d %H:%M:%S')
    data['e'] = str(data['e'])
    # print(data['e'])

    list_of_elem = [data['e'], data['o'], data['h'], data['l'], data['c'], data['v'], data['sym']]

    file_name = 'data_stream.csv'

    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)



def on_close(ws):
    print("closed connection")

socket = "wss://socket.polygon.io/stocks"

ws = websocket.WebSocketApp(socket, on_open=on_open, on_message=on_message, on_close=on_close, on_error=on_error)

# before starting a stream, prepare header in the csv file
# if you rerun the stream capture from this point
# you will end up with 2 headers in csv file
#file_name = 'data_stream.csv'
#header_elements = ['date','openP', 'high', 'low', 'close', 'vol', 'tic']

#with open(file_name, 'a+', newline='') as write_obj:
#    # Create a writer object from csv module
#    csv_writer = writer(write_obj)
#    # Add contents of list as last row in the csv file
#    csv_writer.writerow(header_elements)

data = ws.run_forever()
