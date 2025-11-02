# students/xrp_front.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

API_BASE = "https://fastapi-at3.onrender.com"
API_URL = f"{API_BASE}/predict/xrp"

def _fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{x*100:,.2f}%"

def _delta(value):
    # returns tuple for st.metric delta text
    if pd.isna(value):
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value*100:,.2f}%"

def _sparkline(series: pd.Series):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index, y=series.values, mode="lines",
            line=dict(width=2), fill="tozeroy", name="close"
        )
    )
    fig.update_layout(
        height=80, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    return fig

def display_xrp_front():
    st.title("XRP")

    # layout controls
    top_left, top_right = st.columns([1, 1])
    with top_left:
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
        df = df.sort_values("time").reset_index(drop=True)
        return df

    df = fetch_coingecko_ohlc(coin_id, vs_currency, days)

    tabs = st.tabs(["Overview", "Indicators", "Exchange", "Prediction"])

    #  Overview 
    with tabs[0]:
        if df is None or df.empty:
            st.warning("No recent OHLC data available.")
        else:
       
            bb_len, bb_k = 20, 2.0
            df["bb_mid"] = df["close"].rolling(bb_len).mean()
            df["bb_std"] = df["close"].rolling(bb_len).std()
            df["bb_upper"] = df["bb_mid"] + bb_k * df["bb_std"]
            df["bb_lower"] = df["bb_mid"] - bb_k * df["bb_std"]

            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=df["time"], open=df["open"], high=df["high"],
                    low=df["low"], close=df["close"], name="XRP"
                )
            )
            fig.add_trace(go.Scatter(x=df["time"], y=df["bb_upper"], mode="lines", name="BB Upper", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"],   mode="lines", name="BB Mid",   line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df["time"], y=df["bb_lower"], mode="lines", name="BB Lower", line=dict(width=1)))
            fig.update_layout(xaxis_rangeslider_visible=False, height=520)
            st.plotly_chart(fig, use_container_width=True)

            # market snapshot
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
            m1, m2, m3, m4 = st.columns(4)
            if md:
                m1.metric("Current Price", f"{md['current_price']:.4f} {vs_currency.upper()}")
                m2.metric("Market Cap", f"{(md['market_cap'] or 0) / 1e9:.2f} B")
                m3.metric("24h Volume", f"{(md['volume'] or 0) / 1e6:.2f} M")
                m4.metric("24h Change", f"{(md['price_change_24h'] or 0):.2f}%")

            # windowed performance tiles
            perf = df.set_index("time")["close"]
            pct_24h = perf.pct_change().iloc[-1] if len(perf) >= 2 else np.nan
            pct_7d = (perf.iloc[-1] / perf.iloc[-7] - 1) if len(perf) >= 7 else np.nan
            pct_30d = (perf.iloc[-1] / perf.iloc[-30] - 1) if len(perf) >= 30 else np.nan

            st.subheader("Performance")
            c1, c2, c3 = st.columns(3)
            c1.metric("24h Change", _fmt_pct(pct_24h), delta=_delta(pct_24h))
            c2.metric("7d Change", _fmt_pct(pct_7d), delta=_delta(pct_7d))
            c3.metric("30d Change", _fmt_pct(pct_30d), delta=_delta(pct_30d))

            # sparkline
            st.caption("Close price sparkline")
            st.plotly_chart(_sparkline(perf.tail(90)), use_container_width=True)

    # Indicators
    with tabs[1]:
        if df is None or df.empty:
            st.info("Indicators will appear once price data loads.")
        else:
            def add_indicators(df_):
                out = df_.copy()
                out["SMA_7"] = out["close"].rolling(7).mean()
                out["EMA_7"] = out["close"].ewm(span=7, adjust=False).mean()
                # RSI
                delta = out["close"].diff()
                gain = np.where(delta > 0, delta, 0.0)
                loss = np.where(delta < 0, -delta, 0.0)
                avg_gain = pd.Series(gain).rolling(14).mean()
                avg_loss = pd.Series(loss).rolling(14).mean()
                rs = avg_gain / (avg_loss.replace(0, np.nan))
                out["RSI_14"] = 100 - (100 / (1 + rs))
                # MACD
                ema12 = out["close"].ewm(span=12, adjust=False).mean()
                ema26 = out["close"].ewm(span=26, adjust=False).mean()
                out["MACD"] = ema12 - ema26
                out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
                out["MACD_hist"] = out["MACD"] - out["MACD_signal"]
                return out

            ind = add_indicators(df)

            a, b = st.columns(2)
            with a:
                st.markdown("Price with SMA and EMA")
                st.line_chart(ind.set_index("time")[["close", "SMA_7", "EMA_7"]])
            with b:
                st.markdown("RSI 14")
                st.line_chart(ind.set_index("time")[["RSI_14"]])

            st.markdown("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=ind["time"], y=ind["MACD"], name="MACD", mode="lines"))
            fig_macd.add_trace(go.Scatter(x=ind["time"], y=ind["MACD_signal"], name="Signal", mode="lines"))
            fig_macd.add_trace(go.Bar(x=ind["time"], y=ind["MACD_hist"], name="Histogram"))
            fig_macd.update_layout(height=300, barmode="relative", showlegend=True)
            st.plotly_chart(fig_macd, use_container_width=True)

            # Risk block
            ret = ind["close"].pct_change()
            ann_vol = (ret.rolling(30).std() * np.sqrt(365)).iloc[-1]
            roll_max = ind["close"].cummax()
            drawdown = ind["close"] / roll_max - 1.0
            dd_min_idx = drawdown.idxmin()
            dd_min = drawdown.min()

            s1, s2 = st.columns(2)
            s1.metric("Annualized volatility 30d", _fmt_pct(ann_vol))
            if pd.notna(dd_min):
                s2.metric("Max drawdown", f"{dd_min*100:,.2f}%")
            else:
                s2.metric("Max drawdown", "n/a")

    #  Exchange data
    with tabs[2]:
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

        df_k = fetch_kraken("XRPUSD", 1440)
        if df_k is None or df_k.empty:
            st.info("No Kraken data available.")
        else:
            st.line_chart(df_k.set_index("time")[["close"]])
            df_k["vol_ma7"] = df_k["volume"].rolling(7).mean()
            st.markdown("Kraken volume")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df_k["time"], y=df_k["volume"], name="Volume"))
            fig_vol.add_trace(go.Scatter(x=df_k["time"], y=df_k["vol_ma7"], name="Vol MA7", mode="lines"))
            fig_vol.update_layout(height=300, barmode="overlay")
            st.plotly_chart(fig_vol, use_container_width=True)

    #  Prediction 
    with tabs[3]:
        st.subheader("Price prediction")
        st.info("Returns the next day HIGH from the deployed XRP model.")
        if st.button("Get prediction"):
            try:
                r = requests.get(API_URL, timeout=30)
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
