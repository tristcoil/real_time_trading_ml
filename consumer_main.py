import asyncio

import streamlit as st
from consumer_utils import consumer


st.set_page_config(page_title="stream", layout="wide")

status = st.empty()

connect = st.checkbox("Connect to WS Server")

status2 = st.empty()



if connect:
    asyncio.run(consumer(status, status2))
else:
    status.subheader(f"Disconnected.")
