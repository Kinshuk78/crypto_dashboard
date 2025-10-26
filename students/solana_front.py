import streamlit as st
import pandas as pd
import requests
import datetime as dt
import plotly.graph_objects as go
import numpy as np

API_URL = "https://adv-mlaa-at3-api.onrender.com/predict/solana"

def display_solana_front():
    st.title("Solana")
    crypto_symbol = "solana"
    vs_currency = "usd"
    days = 30

    # Fetch Coingecko OHLC
    @st.cache_data
    def fetch_coingecko_ohlc(crypto_symbol, vs_currency="usd", days=30):
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/ohlc"
        params = {"vs_currency": vs_currency, "days": days}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            st.error("Failed to fetch CoinGecko data")
            return None
        df = pd.DataFrame(resp.json(), columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df

    df = fetch_coingecko_ohlc(crypto_symbol, vs_currency, days)

    if df is not None:
        st.subheader(f"{crypto_symbol.capitalize()} — Last {days} Days")
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["time"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name=f"{crypto_symbol} price",
                )
            ]
        )
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Market data
    @st.cache_data
    def fetch_market_data(crypto_symbol, vs_currency="usd"):
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": vs_currency, "ids": crypto_symbol}
        resp = requests.get(url, params=params)
        if resp.status_code == 200 and len(resp.json()) > 0:
            data = resp.json()[0]
            return {
                "current_price": data["current_price"],
                "market_cap": data["market_cap"],
                "volume": data["total_volume"],
                "price_change_24h": data["price_change_percentage_24h"],
            }
        return None

    market_data = fetch_market_data(crypto_symbol, vs_currency)
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"{market_data['current_price']:.2f} {vs_currency.upper()}")
        col2.metric("Market Cap", f"{market_data['market_cap'] / 1e9:.2f} B")
        col3.metric("24h Volume", f"{market_data['volume'] / 1e6:.2f} M")
        col4.metric("Change (24h)", f"{market_data['price_change_24h']:.2f}%")

    # Technical indicators
    def add_indicators(df):
        df["SMA_7"] = df["close"].rolling(7).mean()
        df["EMA_7"] = df["close"].ewm(span=7, adjust=False).mean()
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI_14"] = 100 - (100 / (1 + rs))
        return df

    if df is not None:
        df = add_indicators(df)
        st.markdown("### 📈 Technical Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df.set_index("time")[["close", "SMA_7", "EMA_7"]])
        with col2:
            st.line_chart(df.set_index("time")[["RSI_14"]])

    # Kraken OHLC data
    @st.cache_data
    def fetch_kraken_data(pair="SOLUSD", interval=1440):
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": pair, "interval": interval}
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            data = resp.json()["result"][pair]
            df = pd.DataFrame(
                data,
                columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
            )
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
            return df
        return None

    st.markdown("### 🪙 Kraken Exchange Data")
    df_kraken = fetch_kraken_data(f"{crypto_symbol[:3].upper()}USD")
    if df_kraken is not None:
        st.line_chart(df_kraken.set_index("time")[["close"]])

    # Prediction section
    st.markdown("## 🤖 Price Prediction")
    st.info("This section helps you to predict the next-day HIGH price of Solana.")

    prediction_date = st.date_input("Select input date for prediction", dt.date.today())

    if st.button("Generate Prediction"):
        params = {"date": prediction_date.strftime("%Y-%m-%d")}
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted next-day high price: ${result['prediction']['high_price']:.2f}")
            else:
                st.warning(f"Failed to get prediction — API Response: {response.json()}")
        except Exception as e:
            st.error(f"Error contacting prediction API: {e}")
