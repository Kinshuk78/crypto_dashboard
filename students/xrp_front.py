# students/xrp_front.py
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import numpy as np

API_BASE = "https://fastapi-at3.onrender.com"
API_URL = f"{API_BASE}/predict/xrp"


def display_xrp_front():
    st.title("XRP")

    #Controls 
    colc1, colc2 = st.columns([1, 3])
    with colc1:
        days = st.selectbox("Window", options=[30, 60, 90], index=0)
    coin_id = "ripple"
    vs_currency = "usd"

    # CoinGecko OHLC 
    @st.cache_data
    def fetch_coingecko_ohlc(coin_id="ripple", vs_currency="usd", days=30):
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {"vs_currency": vs_currency, "days": int(days)}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        df = pd.DataFrame(r.json(), columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df

    df = fetch_coingecko_ohlc(coin_id, vs_currency, days)

    if df is not None and len(df) > 0:
        st.subheader(f"XRP — last {days} days")

        # Bollinger Bands on close
        bb_len, bb_k = 20, 2.0
        df["bb_mid"] = df["close"].rolling(bb_len).mean()
        df["bb_std"] = df["close"].rolling(bb_len).std()
        df["bb_upper"] = df["bb_mid"] + bb_k * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - bb_k * df["bb_std"]

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df["time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="XRP price",
            )
        )
        fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], mode="lines", name="BB Upper", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"],   mode="lines", name="BB Mid",   line=dict(width=1)))
        fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], mode="lines", name="BB Lower", line=dict(width=1)))

        fig.update_layout(xaxis_rangeslider_visible=False, height=520)
        st.plotly_chart(fig, use_container_width=True)

    #  Market snapshot
    @st.cache_data
    def fetch_market_data(coin_id="ripple", vs_currency="usd"):
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": vs_currency, "ids": coin_id}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200 and len(r.json()) > 0:
            d = r.json()[0]
            return {
                "current_price": d.get("current_price"),
                "market_cap": d.get("market_cap"),
                "volume": d.get("total_volume"),
                "price_change_24h": d.get("price_change_percentage_24h"),
            }
        return None

    md = fetch_market_data(coin_id, vs_currency)
    if md:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"{md['current_price']:.4f} {vs_currency.upper()}")
        c2.metric("Market Cap", f"{(md['market_cap'] or 0) / 1e9:.2f} B")
        c3.metric("24h Volume", f"{(md['volume'] or 0) / 1e6:.2f} M")
        c4.metric("Change 24h", f"{(md['price_change_24h'] or 0):.2f}%")

    # Technical indicators 
    def add_indicators(df_):
        out = df_.copy()
        out["SMA_7"] = out["close"].rolling(7).mean()
        out["EMA_7"] = out["close"].ewm(span=7, adjust=False).mean()

        # RSI 14
        delta = out["close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = pd.Series(gain).rolling(14).mean()
        avg_loss = pd.Series(loss).rolling(14).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        out["RSI_14"] = 100 - (100 / (1 + rs))

        # MACD 12,26 with signal 9
        ema12 = out["close"].ewm(span=12, adjust=False).mean()
        ema26 = out["close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = ema12 - ema26
        out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
        return out

    if df is not None and len(df) > 0:
        ind = add_indicators(df)

        st.markdown("### Technical indicators")
        a, b = st.columns(2)
        with a:
            st.line_chart(ind.set_index("time")[["close", "SMA_7", "EMA_7"]])
        with b:
            st.line_chart(ind.set_index("time")[["RSI_14"]])

        # MACD panel
        st.markdown("### MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=ind["time"], y=ind["MACD"], name="MACD", mode="lines"))
        fig_macd.add_trace(go.Scatter(x=ind["time"], y=ind["MACD_signal"], name="Signal", mode="lines"))
        fig_macd.add_trace(go.Bar(x=ind["time"], y=ind["MACD_hist"], name="Histogram"))
        fig_macd.update_layout(height=300, barmode="relative", showlegend=True)
        st.plotly_chart(fig_macd, use_container_width=True)

        # Return stats and risk block
        ret = ind["close"].pct_change()
        ann_vol = (ret.rolling(30).std() * np.sqrt(365)).iloc[-1]
        ret_7d = (ind["close"].iloc[-1] / ind["close"].iloc[-7] - 1) if len(ind) >= 7 else np.nan
        ret_30d = (ind["close"].iloc[-1] / ind["close"].iloc[0] - 1) if len(ind) > 1 else np.nan

        # Max drawdown
        roll_max = ind["close"].cummax()
        drawdown = ind["close"] / roll_max - 1.0
        dd_min_idx = drawdown.idxmin()
        dd_min = drawdown.min()
        peak_idx = ind["close"][:dd_min_idx].idxmax() if not np.isnan(dd_min) else None

        st.markdown("### Return and risk")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("7d return", f"{ret_7d*100:,.2f}%" if pd.notna(ret_7d) else "n/a")
        s2.metric("30d return", f"{ret_30d*100:,.2f}%" if pd.notna(ret_30d) else "n/a")
        s3.metric("Ann. vol (30d)", f"{ann_vol*100:,.2f}%" if pd.notna(ann_vol) else "n/a")
        if peak_idx is not None and pd.notna(dd_min):
            peak_date = ind.loc[peak_idx, "time"].date()
            trough_date = ind.loc[dd_min_idx, "time"].date()
            s4.metric("Max drawdown", f"{dd_min*100:,.2f}%")
            st.caption(f"Peak {peak_date} to trough {trough_date}")
        else:
            s4.metric("Max drawdown", "n/a")

    # Kraken OHLC daily 
    @st.cache_data
    def fetch_kraken(pair="XRPUSD", interval=1440):
        url = "https://api.kraken.com/0/public/OHLC"
        r = requests.get(url, params={"pair": pair, "interval": interval}, timeout=30)
        if r.status_code != 200:
            return None
        res = r.json().get("result", {})
        keys = [k for k in res.keys() if k != "last"]
        if not keys:
            return None
        series = res.get(keys[0], [])
        if not series:
            return None
        cols = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
        dfk = pd.DataFrame(series, columns=cols)
        dfk["time"] = pd.to_datetime(dfk["time"], unit="s")
        for c in ["open", "high", "low", "close", "vwap", "volume"]:
            dfk[c] = dfk[c].astype(float)
        return dfk

    st.markdown("### Kraken exchange data")
    df_k = fetch_kraken("XRPUSD", 1440)
    if df_k is not None and len(df_k) > 0:
        # price line
        st.line_chart(df_k.set_index("time")[["close"]])

        # volume with 7d average
        df_k["vol_ma7"] = df_k["volume"].rolling(7).mean()
        st.markdown("Kraken volume")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=df_k["time"], y=df_k["volume"], name="Volume"))
        fig_vol.add_trace(go.Scatter(x=df_k["time"], y=df_k["vol_ma7"], name="Vol MA7", mode="lines"))
        fig_vol.update_layout(height=300, barmode="overlay")
        st.plotly_chart(fig_vol, use_container_width=True)

    #  Prediction via FastAPI 
    st.markdown("## Price prediction")
    st.info("Returns the next day HIGH from the deployed XRP model.")
    asof = st.date_input("Feature date optional. Leave empty to use the latest available", value=None)

    if st.button("Get prediction"):
        try:
            url = f"{API_URL}?asof={asof.isoformat()}" if asof else API_URL
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                result = r.json()
                st.success(f"Predict date: {result['predict_date']}")
                st.success(f"High price: {result['high_price']}")
                st.json(result)
            else:
                st.warning(f"API returned {r.status_code}")
                with st.expander("Response body"):
                    st.write(r.text)
        except Exception as e:
            st.error(f"Error contacting API: {e}")
