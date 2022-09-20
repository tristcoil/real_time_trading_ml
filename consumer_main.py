import asyncio

import streamlit as st
from consumer_utils import consumer

#import nest_asyncio
#nest_asyncio.apply() 


st.set_page_config(page_title="stream", layout="wide")

status = st.empty()

connect = st.checkbox("Connect to WS Server")

status2 = st.empty()

#loop = asyncio.get_event_loop()
#loop = asyncio.new_event_loop()
#asyncio.set_event_loop(loop)

if connect:
    asyncio.run(consumer(status, status2))
else:
    status.subheader(f"Disconnected.")
