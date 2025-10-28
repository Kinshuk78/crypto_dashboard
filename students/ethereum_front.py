import requests
import streamlit as st
from pandas import to_datetime
from .ethereum_functions import CriptoInfo, create_tecnical_indicators, create_daily_returns
import plotly.graph_objects as go

ETH_API_URL = "https://at3-api-ethereum.onrender.com/predict/ethereum"

def display_ethereum_front():
    st.title("Ethereum")
    # fetch the data
    days = 30
    helper = CriptoInfo(token = "ethereum")
    helper.fetch_ohlc(periods = days)
    helper.fetch_additional_info(periods = days)
    helper.modify_raw_prices()
    data = helper.generate_input()
    data['timestamp'] = to_datetime(data['timestamp'])
    data.set_index(keys = 'timestamp', inplace = True)
    if data is not None:
        st.subheader(f"Ethereum - Last {days} days")
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
    helper.get_market_data()
    if helper.market_data is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{helper.market_data.get('current_price', 0):.2f} USD")
        col2.metric("Market Cap", f"{helper.market_data.get('market_cap', 0) / 1e9:.2f} B")
        col3.metric("24h Volume", f"{helper.market_data.get('volume', 0) / 1e6:.2f} M")
        col4.metric("Change (24h)", f"{helper.market_data.get('24h_price_change', 0):.2f}%")
    
    if data is not None:
        data = create_tecnical_indicators(data)
        data = create_daily_returns(data)
        st.markdown("#### Technical indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(data[['close', 'sma_7', 'ema_7']])
        with col2:
            st.line_chart(data[['daily_returns']], y_label = 'daily returns')
    
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
                
    
        