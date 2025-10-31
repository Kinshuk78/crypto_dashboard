import streamlit as st
import pandas as pd
import requests
import datetime as dt
import plotly.graph_objects as go
import numpy as np

API_BASE_URL = "https://adv-mlaa-at3-api-bitcoin.onrender.com"
TOKEN = "bitcoin"


def display_bitcoin_front():
    st.title("Bitcoin")

    crypto_symbol = "bitcoin"
    vs_currency = "usd"
    days = 30

    # Fetch Coingecko OHLC
    @st.cache_data
    def fetch_coingecko_ohlc(
        crypto_symbol: str, vs_currency: str = "usd", days: int = 30
    ):
        url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/ohlc"
        params = {"vs_currency": vs_currency, "days": days}
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            st.error("Failed to fetch CoinGecko OHLC data")
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
    def fetch_market_data(crypto_symbol: str, vs_currency: str = "usd"):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": vs_currency, "ids": crypto_symbol}
        resp = requests.get(url, params=params)
        if resp.status_code == 200 and len(resp.json()) > 0:
            data = resp.json()[0]
            return {
                "current_price": data.get("current_price"),
                "market_cap": data.get("market_cap"),
                "volume": data.get("total_volume"),
                "price_change_24h": data.get("price_change_percentage_24h"),
            }
        return None

    market_data = fetch_market_data(crypto_symbol, vs_currency)
    if market_data:
        col1, col2, col3, col4 = st.columns(4)
        if market_data.get("current_price") is not None:
            col1.metric(
                "Current Price",
                f"{market_data['current_price']:.2f} {vs_currency.upper()}",
            )
        if market_data.get("market_cap") is not None:
            col2.metric("Market Cap", f"{market_data['market_cap'] / 1e9:.2f} B")
        if market_data.get("volume") is not None:
            col3.metric("24h Volume", f"{market_data['volume'] / 1e6:.2f} M")
        if market_data.get("price_change_24h") is not None:
            col4.metric("Change (24h)", f"{market_data['price_change_24h']:.2f}%")

    # Technical indicators
    def add_indicators(df_prices: pd.DataFrame) -> pd.DataFrame:
        df_prices = df_prices.copy()
        df_prices["SMA_7"] = df_prices["close"].rolling(7).mean()
        df_prices["EMA_7"] = df_prices["close"].ewm(span=7, adjust=False).mean()
        delta = df_prices["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / avg_loss
        df_prices["RSI_14"] = 100 - (100 / (1 + rs))
        return df_prices

    if df is not None:
        df_ind = add_indicators(df)
        st.markdown("### 📈 Technical Indicators")
        col1, col2 = st.columns(2)
        with col1:
            st.line_chart(df_ind.set_index("time")[["close", "SMA_7", "EMA_7"]])
        with col2:
            st.line_chart(df_ind.set_index("time")[["RSI_14"]])

    # Kraken OHLC data (Kraken uses XBT for Bitcoin)
    @st.cache_data
    def fetch_kraken_data(pair: str = "XBTUSD", interval: int = 1440):
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": pair, "interval": interval}
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            payload = resp.json()
            # Kraken returns dict with a key equal to pair
            data = payload.get("result", {}).get(pair)
            if not data:
                return None
            df_k = pd.DataFrame(
                data,
                columns=[
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "vwap",
                    "volume",
                    "count",
                ],
            )
            df_k["time"] = pd.to_datetime(df_k["time"], unit="s")
            df_k[["open", "high", "low", "close", "volume"]] = df_k[
                ["open", "high", "low", "close", "volume"]
            ].astype(float)
            return df_k
        return None

    st.markdown("### 🪙 Kraken Exchange Data")
    df_kraken = fetch_kraken_data("XBTUSD")
    if df_kraken is not None:
        st.line_chart(df_kraken.set_index("time")[["close"]])

    # Prediction section
    st.markdown("## 🤖 Price Prediction")
    st.info("Predict the next-day HIGH price for Bitcoin using the deployed model.")

    prediction_date = st.date_input(
        "Select input date for prediction (optional)", dt.date.today()
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        trigger_with_date = st.button("Generate Prediction for Selected Date")
    with col_b:
        trigger_today = st.button("Generate Prediction for Today")

    def call_prediction_api(date_str: str | None):
        url = f"{API_BASE_URL}/predict/{TOKEN}"
        params = {"date": date_str} if date_str else None
        try:
            resp = requests.get(url, params=params)
            if resp.status_code == 200:
                result = resp.json()
                # API returns { token, prediction: { t_plus_1_high } } (and input_date optionally)
                pred_value = result.get("prediction", {}).get("t_plus_1_high")
                if pred_value is not None:
                    st.success(f"Predicted next-day HIGH price: ${pred_value:,.2f}")
                    if result.get("input_date"):
                        st.caption(
                            f"Input date acknowledged by API: {result['input_date']}"
                        )
                else:
                    st.warning("Prediction field missing in API response.")
            else:
                # Try to show message body if any
                try:
                    st.warning(
                        f"Failed to get prediction — API Response: {resp.status_code} {resp.json()}"
                    )
                except Exception:
                    st.warning(f"Failed to get prediction — Status: {resp.status_code}")
        except Exception as ex:
            st.error(f"Error contacting prediction API: {ex}")

    if trigger_with_date:
        call_prediction_api(
            prediction_date.strftime("%Y-%m-%d") if prediction_date else None
        )
    if trigger_today:
        call_prediction_api(None)
