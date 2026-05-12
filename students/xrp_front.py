# students/xrp_front.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

API_BASE = "https://fastapi-at3.onrender.com"
API_URL = f"{API_BASE}/predict/xrp"
API_HEALTH_URL = f"{API_BASE}/health/"

def _fmt_price(x):
    return "n/a" if x is None or pd.isna(x) else f"${x:,.4f}"

def _fmt_pct(x):
    return "n/a" if pd.isna(x) else f"{x*100:,.2f}%"

def _fmt_pct_points(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:,.2f}%"

def _fmt_market_value(x):
    if x is None or pd.isna(x):
        return "n/a"
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.2f}M"
    return f"${x:,.0f}"

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

def _add_indicators(df_):
    out = df_.copy()
    out["SMA_7"] = out["close"].rolling(7).mean()
    out["EMA_7"] = out["close"].ewm(span=7, adjust=False).mean()
    out["SMA_20"] = out["close"].rolling(20).mean()
    out["ret_1d"] = out["close"].pct_change()

    delta = out["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    out["bb_mid"] = out["close"].rolling(20).mean()
    out["bb_std"] = out["close"].rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + 2.0 * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - 2.0 * out["bb_std"]
    return out

def _latest_value(series):
    cleaned = series.dropna()
    return np.nan if cleaned.empty else cleaned.iloc[-1]

def _signal_labels(ind):
    latest_close = float(ind["close"].iloc[-1])
    latest_rsi = _latest_value(ind["RSI_14"])
    latest_macd = _latest_value(ind["MACD"])
    latest_signal = _latest_value(ind["MACD_signal"])
    latest_sma20 = _latest_value(ind["SMA_20"])

    trend = "Bullish" if latest_close >= latest_sma20 else "Bearish"
    momentum = "Bullish" if latest_macd >= latest_signal else "Bearish"
    if pd.isna(latest_rsi):
        rsi_label = "Building"
    elif latest_rsi >= 70:
        rsi_label = "Overbought"
    elif latest_rsi <= 30:
        rsi_label = "Oversold"
    else:
        rsi_label = "Neutral"
    return trend, momentum, rsi_label

def _risk_label(ann_vol, max_drawdown):
    if pd.isna(ann_vol):
        return "n/a"
    if ann_vol >= 0.70 or max_drawdown <= -0.20:
        return "High"
    if ann_vol >= 0.35 or max_drawdown <= -0.10:
        return "Medium"
    return "Low"

def _request_xrp_prediction():
    """
    Render free-tier services can sleep and need extra time on the first request.
    Warm the health endpoint, then retry the prediction once with a longer read timeout.
    """
    last_error = None
    for attempt in range(2):
        try:
            if attempt == 0:
                try:
                    requests.get(API_HEALTH_URL, timeout=(5, 20))
                except requests.exceptions.RequestException:
                    pass
            return requests.get(API_URL, timeout=(10, 90))
        except requests.exceptions.RequestException as exc:
            last_error = exc
    raise last_error

def display_xrp_front():
    st.title("XRP")
    st.caption("XRP market intelligence dashboard for price action, technical signals, exchange liquidity, and next-day HIGH prediction.")

    # layout controls
    top_left, top_right = st.columns([1, 1])
    with top_left:
        days = st.selectbox("Window", options=[30, 60, 90], index=0)
    coin_id = "ripple"
    vs_currency = "usd"

    @st.cache_data(ttl=900)
    def fetch_kraken_ohlc(pair="XRPUSD", interval=1440):
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
        return dfk.sort_values("time").reset_index(drop=True)

    # CoinGecko OHLC only accepts specific day windows. Fetch the next supported
    # window and trim locally so the app can still offer a 60-day view.
    @st.cache_data(ttl=900)
    def fetch_coingecko_ohlc(coin_id="ripple", vs_currency="usd", days=30):
        supported_days = [1, 7, 14, 30, 90, 180, 365]
        request_days = next((d for d in supported_days if d >= int(days)), 365)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {"vs_currency": vs_currency, "days": request_days}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return None
        df = pd.DataFrame(r.json(), columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for c in ["open", "high", "low", "close"]:
            df[c] = df[c].astype(float)
        return df.sort_values("time").reset_index(drop=True)

    def filter_window(df_, days_):
        if df_ is None or df_.empty:
            return df_
        cutoff = df_["time"].max() - pd.Timedelta(days=int(days_))
        filtered = df_.loc[df_["time"] >= cutoff].copy()
        return filtered.reset_index(drop=True)

    @st.cache_data(ttl=900)
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

    df = filter_window(fetch_coingecko_ohlc(coin_id, vs_currency, days), days)
    if df is None or df.empty:
        df = filter_window(fetch_kraken_ohlc("XRPUSD", 1440), days)

    md = fetch_market_data(coin_id, vs_currency)
    ind = _add_indicators(df) if df is not None and not df.empty else None
    perf = df.set_index("time")["close"] if df is not None and not df.empty else pd.Series(dtype=float)
    pct_24h = perf.pct_change().iloc[-1] if len(perf) >= 2 else np.nan
    pct_7d = (perf.iloc[-1] / perf.iloc[-7] - 1) if len(perf) >= 7 else np.nan
    pct_30d = (perf.iloc[-1] / perf.iloc[-30] - 1) if len(perf) >= 30 else np.nan
    latest_close = float(perf.iloc[-1]) if len(perf) else np.nan
    ann_vol = (ind["close"].pct_change().rolling(30).std() * np.sqrt(365)).iloc[-1] if ind is not None else np.nan
    drawdown = ind["close"] / ind["close"].cummax() - 1.0 if ind is not None else pd.Series(dtype=float)
    dd_min = drawdown.min() if not drawdown.empty else np.nan
    trend, momentum, rsi_label = _signal_labels(ind) if ind is not None else ("n/a", "n/a", "n/a")

    st.markdown("#### Executive snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current price", _fmt_price(md.get("current_price") if md else latest_close))
    m2.metric("24h change", _fmt_pct_points(md.get("price_change_24h") if md else pct_24h * 100))
    m3.metric("Market cap", _fmt_market_value(md.get("market_cap") if md else None))
    m4.metric("24h volume", _fmt_market_value(md.get("volume") if md else None))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("30d performance", _fmt_pct(pct_30d), delta=_delta(pct_30d))
    s2.metric("Trend", trend)
    s3.metric("Momentum", momentum)
    s4.metric("Risk level", _risk_label(ann_vol, dd_min))

    st.caption("Market data: CoinGecko. OHLC and liquidity fallback: Kraken. Forecast service: deployed XRP FastAPI model.")

    tabs = st.tabs(["Overview", "Indicators", "Exchange", "Prediction"])

    #  Overview 
    with tabs[0]:
        if df is None or df.empty:
            st.warning("No recent OHLC data available.")
        else:
            st.subheader(f"XRP price structure over the last {days} days")
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=ind["time"], open=ind["open"], high=ind["high"],
                    low=ind["low"], close=ind["close"], name="XRP"
                )
            )
            fig.add_trace(go.Scatter(x=ind["time"], y=ind["bb_upper"], mode="lines", name="BB Upper", line=dict(width=1, color="#3b82f6")))
            fig.add_trace(go.Scatter(x=ind["time"], y=ind["bb_mid"], mode="lines", name="BB Mid", line=dict(width=1, color="#f3b6b6")))
            fig.add_trace(go.Scatter(x=ind["time"], y=ind["bb_lower"], mode="lines", name="BB Lower", line=dict(width=1, color="#ef4444")))
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                height=460,
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified",
                yaxis_title="Price (USD)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Candles show daily open, high, low and close. Bollinger Bands show whether price is trading near the top, middle or bottom of its recent volatility range.")

            st.subheader("Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("24h Change", _fmt_pct(pct_24h), delta=_delta(pct_24h))
            c2.metric("7d Change", _fmt_pct(pct_7d), delta=_delta(pct_7d))
            c3.metric("30d Change", _fmt_pct(pct_30d), delta=_delta(pct_30d))
            c4.metric("Latest close", _fmt_price(latest_close))

            with st.expander("How to read this overview"):
                st.markdown(
                    """
                    - **Candlestick body**: daily open-to-close movement.
                    - **Wicks**: daily high and low range.
                    - **Bollinger Bands**: wider bands indicate higher volatility; price near the upper band suggests strength, while price near the lower band suggests weakness or stress.
                    - **Performance tiles**: short-window returns help compare immediate movement against the broader 30-day trend.
                    """
                )

            st.caption("Close price sparkline")
            st.plotly_chart(_sparkline(perf.tail(90)), use_container_width=True)

    # Indicators
    with tabs[1]:
        if df is None or df.empty:
            st.info("Indicators will appear once price data loads.")
        else:
            st.subheader("Technical signal monitor")
            i1, i2, i3 = st.columns(3)
            i1.metric("Trend signal", trend, help="Close price compared with the 20-day simple moving average.")
            i2.metric("Momentum signal", momentum, help="MACD compared with its signal line.")
            i3.metric("RSI state", rsi_label, help="RSI above 70 is commonly treated as overbought; below 30 is commonly treated as oversold.")

            a, b = st.columns(2)
            with a:
                st.markdown("Price with SMA and EMA")
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=ind["time"], y=ind["close"], name="Close", mode="lines", line=dict(color="#f3b6b6")))
                fig_ma.add_trace(go.Scatter(x=ind["time"], y=ind["SMA_7"], name="SMA 7", mode="lines", line=dict(color="#2563eb")))
                fig_ma.add_trace(go.Scatter(x=ind["time"], y=ind["EMA_7"], name="EMA 7", mode="lines", line=dict(color="#93c5fd")))
                fig_ma.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified", yaxis_title="USD")
                st.plotly_chart(fig_ma, use_container_width=True)
            with b:
                st.markdown("RSI 14")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=ind["time"], y=ind["RSI_14"], name="RSI 14", mode="lines"))
                fig_rsi.add_hrect(y0=70, y1=100, fillcolor="#ef4444", opacity=0.12, line_width=0)
                fig_rsi.add_hrect(y0=0, y1=30, fillcolor="#22c55e", opacity=0.12, line_width=0)
                fig_rsi.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 100]), hovermode="x unified")
                st.plotly_chart(fig_rsi, use_container_width=True)

            st.markdown("MACD")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=ind["time"], y=ind["MACD"], name="MACD", mode="lines"))
            fig_macd.add_trace(go.Scatter(x=ind["time"], y=ind["MACD_signal"], name="Signal", mode="lines"))
            fig_macd.add_trace(go.Bar(x=ind["time"], y=ind["MACD_hist"], name="Histogram"))
            fig_macd.update_layout(height=300, barmode="relative", showlegend=True, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified")
            st.plotly_chart(fig_macd, use_container_width=True)

            s1, s2, s3 = st.columns(3)
            s1.metric("Annualized volatility 30d", _fmt_pct(ann_vol))
            s2.metric("Max drawdown", f"{dd_min*100:,.2f}%" if pd.notna(dd_min) else "n/a")
            s3.metric("Risk level", _risk_label(ann_vol, dd_min))

            with st.expander("What these signals mean"):
                st.markdown(
                    """
                    - **SMA/EMA**: moving averages smooth price action and show whether short-term price is above or below trend.
                    - **RSI 14**: momentum oscillator. Above 70 can mean overheated; below 30 can mean oversold.
                    - **MACD**: momentum crossover indicator. MACD above signal line is usually interpreted as bullish momentum.
                    - **Volatility and drawdown**: risk measures that show how unstable the price has been and how far it fell from a recent peak.
                    """
                )

    #  Exchange data
    with tabs[2]:
        df_k = filter_window(fetch_kraken_ohlc("XRPUSD", 1440), days)
        if df_k is None or df_k.empty:
            st.info("No Kraken data available.")
        else:
            st.subheader("Kraken spot market liquidity")
            e1, e2, e3 = st.columns(3)
            e1.metric("Kraken latest close", _fmt_price(float(df_k["close"].iloc[-1])))
            e2.metric("Latest daily volume", f"{float(df_k['volume'].iloc[-1]):,.0f} XRP")
            e3.metric("Daily trade count", f"{int(df_k['count'].iloc[-1]):,}" if "count" in df_k else "n/a")

            fig_ex = go.Figure()
            fig_ex.add_trace(go.Scatter(x=df_k["time"], y=df_k["close"], name="Close", mode="lines"))
            fig_ex.update_layout(height=330, margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified", yaxis_title="USD")
            st.plotly_chart(fig_ex, use_container_width=True)
            df_k["vol_ma7"] = df_k["volume"].rolling(7).mean()
            st.markdown("Kraken volume")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=df_k["time"], y=df_k["volume"], name="Volume"))
            fig_vol.add_trace(go.Scatter(x=df_k["time"], y=df_k["vol_ma7"], name="Vol MA7", mode="lines"))
            fig_vol.update_layout(height=320, barmode="overlay", margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified", yaxis_title="XRP")
            st.plotly_chart(fig_vol, use_container_width=True)
            st.caption("Kraken OHLC provides daily exchange price and volume, useful for validating CoinGecko market data against a live exchange feed.")

    #  Prediction 
    with tabs[3]:
      st.subheader("Price prediction")
      st.info("Forecast target: next-day HIGH price for XRP from the deployed FastAPI model.")


      latest_close = None
      latest_time = None
      if df is not None and not df.empty:
        latest_close = float(df["close"].iloc[-1])
        latest_time = pd.to_datetime(df["time"].iloc[-1]).date()

      if st.button("Get prediction"):
        try:
            with st.spinner("Waking XRP prediction service and generating forecast..."):
                r = _request_xrp_prediction()
            if r.status_code == 200:
                res = r.json()
                predict_date = res.get("predict_date")
                pred_high = float(res.get("high_price"))

                if latest_close:
                    pct_move = (pred_high - latest_close) / latest_close
                    # headline metrics
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Latest close", f"{latest_close:.4f} USD", help=f"As of {latest_time}")
                    m2.metric("Predicted next day HIGH", f"{pred_high:.4f} USD")
                    m3.metric("Expected change", f"{pct_move*100:,.2f}%", delta=f"{pct_move*100:,.2f}%")

                    st.progress(
                        value=float(np.clip((pct_move + 0.1) / 0.2, 0, 1)),
                        text="Predicted upside vs latest close"
                    )

                    if pct_move >= 0:
                        st.success(
                            f"Model expects the next day HIGH on {predict_date} to be about "
                            f"{pct_move*100:,.2f}% above the latest close."
                        )
                    else:
                        st.warning(
                            f"Model expects the next day HIGH on {predict_date} to be about "
                            f"{pct_move*100:,.2f}% below the latest close."
                        )
                else:
                    st.metric("Predicted next day HIGH", f"{pred_high:.4f} USD")
                    st.caption("Could not compute percent change because no recent close was available.")

            else:
                st.warning(f"API returned {r.status_code}")
                with st.expander("Response body"):
                    st.write(r.text)
        except requests.exceptions.Timeout as e:
            st.warning(
                "The XRP prediction API took too long to respond. This usually happens when the Render service is waking up. "
                "Wait 30-60 seconds and click Get prediction again."
            )
            with st.expander("Technical details"):
                st.write(str(e))
        except requests.exceptions.RequestException as e:
            st.error("Could not reach the XRP prediction API. Check whether the Render service is online and try again.")
            with st.expander("Technical details"):
                st.write(str(e))
        except Exception as e:
            st.error(f"Error contacting API: {e}")

      with st.expander("Model and deployment details"):
        st.markdown(
            f"""
            - **Prediction endpoint**: `{API_URL}`
            - **Target variable**: next-day XRP `HIGH` price.
            - **Live input source**: Kraken daily OHLC candles.
            - **Output**: prediction date and predicted high price in USD.
            - **Important limitation**: this is an ML forecast for analysis only, not financial advice.
            """
        )
