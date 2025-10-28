import requests
import streamlit as st
from pandas import to_datetime
from ethereum_functions import CriptoInfo, create_tecnical_indicators
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
    data.set_index(keys = 'timestamp', inplace = True)
    data.index = to_datetime(input.index)
    if data is not None:
        st.subheader(f"Ethereum - Last {days} days")
        fig = go.Figure(data=[
            go.Candlestick(
                x = data['timestamp'],
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
        st.markdown("#### Technical indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(data[['close', 'sma_7', 'ema_7']])
        with col2:
            st.line_chart(data[['rsi_14']])
    
    st.markdown("### Price prediction")
    st.info("This section helps you predict the highest price of the next day from the available information")
    if st.button("Generate prediction:"):
        try:
            response = requests.get(ETH_API_URL)
            if response.status_code == 200:
                info = response.json()
                text = (
                    f"The latest available date is: {info.get("latest_date", "")}\n"
                    f"The prediction date is: {info.get("prediction_date", "")}\n"
                    f"The predicted highest price is: {float(info.get("prediction", 0)):.2f}"
                )
                st.success(text)
            else:
                st.warning(f"Failed to get prediction — API Response: {response.json()}")
        except Exception as e:
            st.error(f"Error consulting API predictions: {e}")
                
    
        