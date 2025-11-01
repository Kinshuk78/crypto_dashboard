import streamlit as st
import pandas as pd
import requests
import datetime as dt
import plotly.graph_objects as go
import numpy as np

API_BASE_URL = "https://adv-mlaa-at3-api.onrender.com"

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
            st.markdown("#### Simple Moving Average and Exponential Moving Average 7 days")
            st.line_chart(df.set_index("time")[["close", "SMA_7", "EMA_7"]])
        with col2:
            st.markdown("#### RSI: Relative Strength Index (14-day)")
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
    st.markdown("---")
    st.markdown("## 🤖 Price Prediction")
    today_str = dt.datetime.now().strftime("%Y-%m-%d")
    st.info(f"This section helps you to predict tomorrow's HIGH price of Solana using market data and engineered features from last 14 days.")

    st.write(f"Actual date for prediction: {dt.date.today()}")

    if st.button(
        "🤖 Get Prediction for Tomorrow", type="primary", use_container_width=True
    ):
        with st.spinner("Calling prediction API…"):
            try:
                response = requests.get(f"{API_BASE_URL}/predict/solana")
                if response.status_code != 200:
                    try:
                        msg = response.json()
                    except Exception:
                        msg = response.text
                    st.warning(f"API Error ({response.status_code}): {msg}")
                else:
                    result = response.json()
                    if result.get("error", False):
                        msg = result.get("message", "No error message provided. Check API for details")
                        st.warning(f"API Error: {msg}")
                    else:
                        pred_value = result.get("high_price")
                        todays_high = float(df["high"].iloc[-1])
                        direction_up = pred_value >= todays_high
                        delta = pred_value - todays_high
                        delta_pct = (delta / todays_high) * 100

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Today's High", f"${todays_high:,.2f}")
                        c2.metric(
                            "Predicted Tomorrow High",
                            f"${pred_value:,.2f}",
                            delta=f"{delta:+.2f} ({delta_pct:+.2f}%)",
                            delta_color="normal",
                        )
                        # Color the direction text
                        direction_text = "📈 UP" if direction_up else "📉 DOWN"
                        direction_color = "green" if direction_up else "red"
                        c3.markdown(
                            f"<div style='text-align: center; padding-top: 15px;'><span style='color: {direction_color}; font-size: 18px; font-weight: bold;'>{direction_text}</span></div>",
                            unsafe_allow_html=True,
                        )
            except Exception as e:
                st.error(f"Error contacting prediction API: {e}")
    st.markdown("---")
    st.caption(
        f"API: {API_BASE_URL} • Data: Kraken (analytics), CoinGecko (overview)"
    )
