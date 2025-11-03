import requests
import streamlit as st
from pandas import to_datetime
from .ethereum_functions import CryptoInfoHelper
import plotly.graph_objects as go
import pandas as pd

ETH_API_URL = "https://at3-api-ethereum.onrender.com/predict/ethereum"


def display_ethereum_front():
    st.title("Ethereum")
    # fetch the data
    helper = CryptoInfoHelper()
    # helper.fetch_ohlc(periods = days)
    # helper.fetch_additional_info(periods = days)
    # helper.modify_raw_prices()
    # data = helper.generate_input()
    # data['timestamp'] = to_datetime(data['timestamp'])
    # data.set_index(keys = 'timestamp', inplace = True)
    data = helper.fetch_ohlc_data(interval = 1440)
    data.index = to_datetime(data.index)
    if data is not None:
        st.subheader(f"Ethereum - Last {(len(data.index))} days")
        fig = go.Figure(data=[
            go.Candlestick(
                x = data.index,
                open = data['open'],
                high = data['high'],
                low = data['low'],
                close = data['close']
            )
        ])
        fig.update_layout(xaxis_rangeslider_visible = True, height = 500)
        st.plotly_chart(fig, use_container_width = True)
    # fetch market data
    market_data = helper.get_market_data()
    if market_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{market_data.get('current_price', 0):.2f} USD")
        col2.metric("Market Cap", f"{market_data.get('market_cap', 0) / 1e9:.2f} B")
        col3.metric("24h Volume", f"{market_data.get('volume', 0) / 1e6:.2f} M")
        col4.metric("Change (24h)", f"{market_data.get('24h_price_change', 0):.2f}%")
    
    if data is not None:
        data = helper.create_tecnical_indicators(data)
        data = helper.create_returns(data)
        st.markdown("#### Technical indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(data[['close', 'sma_7', 'ema_7']])
        with col2:
            st.line_chart(data[['daily_returns', 'weekly_returns', 'monthly_returns']])
    
    st.markdown("### Price prediction")
    st.info("This section helps you predict the highest price of the next day from the available information")
    if st.button("Generate prediction:"):
        try:
            response = requests.get(ETH_API_URL)
            if response.status_code == 200:
                info = response.json()
                st.success(f"Prediction for {info.get('prediction_date', '')} is: {float(info.get('prediction', 0)):.2f} USD")
            else:
                st.warning(f"Failed to get prediction — API Response: {response.json()}")
        except Exception as e:
            st.error(f"Error consulting API predictions: {e}")
                
    
        