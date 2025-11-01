# bitcoin_front.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from math import sqrt
from datetime import datetime
import plotly.graph_objects as go

API_BASE_URL = "https://adv-mlaa-at3-api-bitcoin.onrender.com"
TOKEN = "bitcoin"


# ---------- Indicators ----------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _realized_vol(log_ret: pd.Series, window: int = 14, periods_per_year: int = 365):
    return log_ret.rolling(window).std() * sqrt(periods_per_year)


def _bollinger(series: pd.Series, window=20, num_std=2.0):
    ma = series.rolling(window).mean()
    sd = series.rolling(window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower


# ---------- Data ----------
@st.cache_data(ttl=300)
def _fetch_market_data(crypto_symbol: str = "bitcoin", vs_currency: str = "usd"):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": vs_currency, "ids": crypto_symbol}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200 and len(r.json()) > 0:
            d = r.json()[0]
            return {
                "current_price": d.get("current_price"),
                "market_cap": d.get("market_cap"),
                "volume": d.get("total_volume"),
                "price_change_24h": d.get("price_change_percentage_24h"),
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def _fetch_coingecko_ohlc(crypto_symbol="bitcoin", vs_currency="usd", days=365):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/ohlc"
    params = {"vs_currency": vs_currency, "days": days}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df[["open", "high", "low", "close"]] = df[
            ["open", "high", "low", "close"]
        ].astype(float)
        df["volume"] = 0.0
        return df
    except Exception:
        return None


@st.cache_data(ttl=300)
def _fetch_kraken_ohlc(pair="XBTUSD", interval=1440):
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        result = r.json().get("result", {})
        data = None
        for k in result.keys():
            if isinstance(result[k], list):
                data = result[k]
                break
        if not data:
            return None
        df = pd.DataFrame(
            data,
            columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"],
        )
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        return df
    except Exception:
        return None


# ---------- Page ----------
def display_bitcoin_front():
    st.title("₿ Bitcoin Analysis & Prediction")
    st.markdown(
        """
        This dashboard contains:
        1. **Price Analysis**: Interactive candlestick or OHLC line charts with customizable time ranges and technical overlays (MA20, Bollinger Bands)
        2. **Technical Analytics**: Real-time calculations of realized volatility, momentum indicators, RSI(14), MACD(12,26,9), and log returns of daily high
        3. **ML Prediction**: Today-only prediction model for tomorrow's HIGH price with directional indicators and visual comparison
        4. **Market Overview**: Current market metrics including price, market cap, 24h volume, and price change
        """
    )
    st.markdown("---")

    market = _fetch_market_data("bitcoin", "usd")
    if market:
        c1, c2, c3, c4 = st.columns(4)
        if market.get("current_price") is not None:
            c1.metric("Current Price", f"${market['current_price']:,.2f}")
        if market.get("market_cap") is not None:
            c2.metric("Market Cap", f"${market['market_cap']/1e12:.2f}T")
        if market.get("volume") is not None:
            c3.metric("24h Volume", f"${market['volume']/1e9:.2f}B")
        if market.get("price_change_24h") is not None:
            c4.metric(
                "24h Change",
                f"{market['price_change_24h']:.2f}%",
                delta=f"{market['price_change_24h']:.2f}%",
            )
        st.markdown("---")

    st.subheader("📈 Price Analysis")
    col_controls = st.columns([2, 2, 2])
    with col_controls[0]:
        range_choice = st.radio(
            "Time Range",
            options=["7D", "1M", "3M", "6M", "1Y", "All"],
            horizontal=True,
            index=1,
        )
    with col_controls[1]:
        chart_mode = st.radio(
            "Chart Mode",
            options=["Candlestick (Full)", "Lines (OHLC)"],
            horizontal=True,
            index=0,
        )
    with col_controls[2]:
        show_overlays = st.checkbox(
            "MA20 & Bollinger Bands", value=True, help="Only for Candlestick mode"
        )

    # Short explainer right before the chart
    with st.expander("ℹ️ What this chart shows"):
        st.markdown(
            "- **Candles**: Each candle represents one day. Green candle = close > open (price rose); Red candle = close < open (price fell). "
            "The wicks show the high/low range, and the body shows open-to-close movement.\n"
            "- **OHLC**: Open/High/Low/Close — the four key prices that define each candle and show full intraday movement.\n"
            "- **MA20**: 20-day moving average of the close price — smooths out short-term fluctuations to show the trend.\n"
            "- **Bollinger Bands**: MA20 ± 2×standard deviation — visualizes price volatility; wider bands = more volatile market."
        )

    days_map = {"7D": 7, "1M": 30, "3M": 90, "6M": 180, "1Y": 365, "All": 9999}
    days = days_map[range_choice]
    fetch_days = max(days, 365) if days != 9999 else 365

    df_raw = _fetch_coingecko_ohlc("bitcoin", "usd", days=fetch_days)
    if df_raw is None or df_raw.empty:
        st.warning("CoinGecko unavailable, trying Kraken…")
        df_raw = _fetch_kraken_ohlc("XBTUSD", interval=1440)
    if df_raw is None or df_raw.empty:
        st.error("Unable to fetch OHLC data right now.")
        st.stop()

    # Fill missing volume from Kraken if needed
    if df_raw["volume"].sum() == 0:
        df_k = _fetch_kraken_ohlc("XBTUSD", interval=1440)
        if df_k is not None and not df_k.empty:
            df_raw["time_day"] = df_raw["time"].dt.date
            df_k["time_day"] = df_k["time"].dt.date
            vol_map = df_k[["time_day", "volume"]].set_index("time_day")
            for i in df_raw.index:
                d = df_raw.loc[i, "time_day"]
                if d in vol_map.index:
                    df_raw.loc[i, "volume"] = vol_map.loc[d, "volume"]
            df_raw.drop(columns=["time_day"], inplace=True, errors="ignore")

    max_needed = max(days, 365) if days != 9999 else 365
    df_full = df_raw.tail(max_needed) if len(df_raw) >= max_needed else df_raw.copy()

    # Analytics basis
    df_full["log_return"] = np.log(df_full["close"]).diff()
    df_full["log_return_high"] = np.log(df_full["high"]).diff()
    df_full["realized_vol_14"] = _realized_vol(df_full["log_return"], 14, 365)
    df_full["momentum"] = df_full["close"].pct_change(14) * 100
    df_full["rsi_14"] = _rsi(df_full["close"], 14)
    macd_line, signal_line, hist = _macd(df_full["close"], 12, 26, 9)
    df_full["macd"] = macd_line
    df_full["macd_signal"] = signal_line
    df_full["macd_hist"] = hist
    ma20, bb_up, bb_dn = _bollinger(df_full["close"], 20, 2.0)
    df_full["bb_ma20"] = ma20
    df_full["bb_up"] = bb_up
    df_full["bb_dn"] = bb_dn
    df_full["log_return_high_pct"] = df_full["log_return_high"] * 100.0

    df_view = df_full.copy() if days == 9999 else df_full.tail(days).copy()

    # Chart
    if chart_mode == "Candlestick (Full)":
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df_view["time"],
                open=df_view["open"],
                high=df_view["high"],
                low=df_view["low"],
                close=df_view["close"],
                name="OHLC",
            )
        )
        if show_overlays:
            fig.add_trace(
                go.Scatter(
                    x=df_view["time"], y=df_view["bb_ma20"], name="MA20", mode="lines"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_view["time"], y=df_view["bb_up"], name="BB Upper", mode="lines"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df_view["time"], y=df_view["bb_dn"], name="BB Lower", mode="lines"
                )
            )
        fig.add_trace(
            go.Bar(
                x=df_view["time"],
                y=df_view["volume"],
                name="Volume",
                opacity=0.35,
                marker_color="rgba(100,150,200,0.6)",
                yaxis="y2",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.2f} BTC<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Bitcoin (XBTUSD) — {range_choice}",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            yaxis2=dict(
                title="Volume (BTC)", overlaying="y", side="right", showgrid=False
            ),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df_view["time"], y=df_view["open"], mode="lines", name="Open")
        )
        fig.add_trace(
            go.Scatter(x=df_view["time"], y=df_view["high"], mode="lines", name="High")
        )
        fig.add_trace(
            go.Scatter(x=df_view["time"], y=df_view["low"], mode="lines", name="Low")
        )
        fig.add_trace(
            go.Scatter(
                x=df_view["time"], y=df_view["close"], mode="lines", name="Close"
            )
        )
        fig.add_trace(
            go.Bar(
                x=df_view["time"],
                y=df_view["volume"],
                name="Volume",
                opacity=0.25,
                yaxis="y2",
            )
        )
        fig.update_layout(
            title=f"Bitcoin (XBTUSD) — {range_choice} (Lines)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            yaxis2=dict(
                title="Volume (BTC)", overlaying="y", side="right", showgrid=False
            ),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Today's quick metrics
    if not df_view.empty:
        latest = df_view.iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Open (Today)", f"${latest['open']:,.2f}")
        m2.metric("High (Today)", f"${latest['high']:,.2f}")
        m3.metric("Low (Today)", f"${latest['low']:,.2f}")
        m4.metric("Close (Today)", f"${latest['close']:,.2f}")

        # ---------- Analytics ----------
        st.markdown("---")
    st.subheader("📊 Analytics")
    with st.expander("ℹ️ What these analytics mean"):
        st.markdown(
            "- **Realized Volatility (14d)**: Annualized variability of daily log-returns (higher = choppier market).\n"
            "- **Momentum (14d)**: % change in close over 14 days (green up / red down).\n"
            "- **RSI(14)**: Relative Strength Index measures how fast and far prices have moved recently. "
            ">70 suggests the asset may be overbought (too many buyers, potential pullback). <30 suggests oversold (too many sellers, potential bounce). "
            "It helps identify when price movements might reverse.\n"
            "- **MACD(12,26,9)**: Moving Average Convergence Divergence compares two moving averages to spot trend changes. "
            "When the MACD line crosses above the signal line, it suggests upward momentum (bullish). "
            "When it crosses below, it suggests downward momentum (bearish). "
            "The histogram shows the strength of that momentum—positive bars mean bullish momentum is strengthening, negative bars mean bearish momentum.\n"
            "- **Log Return of High**: Daily log change of the **High** price—this is the target variable your ML model predicts (transform helps normalize the data)."
        )

    c1, c2 = st.columns(2)
    with c1:
        fig_rv = go.Figure()
        if not df_view["realized_vol_14"].isna().all():
            fig_rv.add_trace(
                go.Scatter(
                    x=df_view["time"],
                    y=df_view["realized_vol_14"],
                    mode="lines",
                    name="Realized Volatility",
                    line=dict(color="purple", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(128,0,128,0.1)",
                )
            )
            avg_vol = df_view["realized_vol_14"].mean()
            fig_rv.add_hline(
                y=avg_vol,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Avg: {avg_vol:.2f}",
            )
        fig_rv.update_layout(
            title="Realized Volatility (14-day, annualized)",
            xaxis_title="Date",
            yaxis_title="σ (annualized)",
        )
        st.plotly_chart(fig_rv, use_container_width=True)
        st.caption("Annualized variability of daily log-returns (close).")

    with c2:
        fig_mom = go.Figure()
        if not df_view["momentum"].isna().all():
            colors = ["green" if x > 0 else "red" for x in df_view["momentum"]]
            fig_mom.add_trace(
                go.Bar(
                    x=df_view["time"],
                    y=df_view["momentum"],
                    name="Momentum (14d %)",
                    marker_color=colors,
                    opacity=0.7,
                )
            )
            fig_mom.add_hline(y=0, line_dash="dash", line_color="black")
        fig_mom.update_layout(
            title="Momentum (14-day % change)",
            xaxis_title="Date",
            yaxis_title="Momentum (%)",
        )
        st.plotly_chart(fig_mom, use_container_width=True)
        st.caption("% change of close over 14 days.")

    fig_rsi = go.Figure()
    if not df_view["rsi_14"].isna().all():
        fig_rsi.add_trace(
            go.Scatter(
                x=df_view["time"],
                y=df_view["rsi_14"],
                mode="lines",
                name="RSI(14)",
                line=dict(color="blue", width=2),
            )
        )
        fig_rsi.add_hrect(
            y0=70,
            y1=100,
            line_width=0,
            fillcolor="red",
            opacity=0.15,
            annotation_text="Overbought",
        )
        fig_rsi.add_hrect(
            y0=0,
            y1=30,
            line_width=0,
            fillcolor="green",
            opacity=0.15,
            annotation_text="Oversold",
        )
        fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray")
    fig_rsi.update_layout(
        title="RSI(14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.caption("Speed/magnitude of moves. 70≈stretched, 30≈depressed.")

    fig_macd = go.Figure()
    if not df_view["macd"].isna().all():
        fig_macd.add_trace(
            go.Scatter(x=df_view["time"], y=df_view["macd"], name="MACD", mode="lines")
        )
        fig_macd.add_trace(
            go.Scatter(
                x=df_view["time"], y=df_view["macd_signal"], name="Signal", mode="lines"
            )
        )
        hist_colors = ["green" if x > 0 else "red" for x in df_view["macd_hist"]]
        fig_macd.add_trace(
            go.Bar(
                x=df_view["time"],
                y=df_view["macd_hist"],
                name="Histogram",
                marker_color=hist_colors,
                opacity=0.5,
            )
        )
        fig_macd.add_hline(y=0, line_dash="dash", line_color="black")
    fig_macd.update_layout(
        title="MACD (12,26,9)",
        xaxis_title="Date",
        yaxis_title="Value",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_macd, use_container_width=True)
    st.caption("Line vs signal for bias; histogram shows momentum of that bias.")

    fig_lrh = go.Figure()
    if not df_view["log_return_high_pct"].isna().all():
        colors = ["green" if x > 0 else "red" for x in df_view["log_return_high_pct"]]
        fig_lrh.add_trace(
            go.Bar(
                x=df_view["time"],
                y=df_view["log_return_high_pct"],
                name="Log Return (High, %)",
                marker_color=colors,
                opacity=0.7,
            )
        )
        fig_lrh.add_hline(y=0, line_dash="dash", line_color="black")
        avg_ret = df_view["log_return_high_pct"].mean()
        fig_lrh.add_hline(
            y=avg_ret,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Avg: {avg_ret:.2f}%",
        )
    fig_lrh.update_layout(
        title="Log Returns — Daily High (%)",
        xaxis_title="Date",
        yaxis_title="Log Return (%)",
    )
    st.plotly_chart(fig_lrh, use_container_width=True)
    st.caption("Daily log change of the High (target transform).")

    # ---------- Prediction ----------
    st.markdown("---")
    st.subheader("🤖 ML Prediction: Next-Day HIGH (Today Only)")
    today_str = datetime.now().strftime("%Y-%m-%d")
    st.info(
        f"Predicts tomorrow’s HIGH using today’s ({today_str}) market data and engineered features."
    )

    if st.button(
        "🔮 Get Prediction for Today", type="primary", use_container_width=True
    ):
        url = f"{API_BASE_URL}/predict/{TOKEN}"
        with st.spinner("Calling prediction API…"):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code != 200:
                    try:
                        msg = r.json()
                    except Exception:
                        msg = r.text
                    st.warning(f"API Error ({r.status_code}): {msg}")
                else:
                    result = r.json()
                    pred_value = result.get("prediction", {}).get("t_plus_1_high")
                    if pred_value is None:
                        st.warning("Prediction not found in API response.")
                    else:
                        todays_high = float(df_raw["high"].iloc[-1])
                        direction_up = pred_value >= todays_high
                        delta = pred_value - todays_high
                        delta_pct = (delta / todays_high) * 100

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Today's High", f"${todays_high:,.2f}")
                        # Use "normal" so negative delta shows red, positive shows green
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

                        fig_pred = go.Figure()
                        # Baseline = today's high
                        fig_pred.add_hline(
                            y=todays_high,
                            line_dash="dot",
                            line_color="gray",
                            annotation_text="Today's High (baseline)",
                        )
                        # Bars (text moved to annotations for better contrast)
                        fig_pred.add_trace(
                            go.Bar(
                                x=["Today"],
                                y=[todays_high],
                                name="Today High",
                                marker_color="rgba(200,200,200,0.9)",
                            )
                        )
                        bar_color = "#27ae60" if direction_up else "#e74c3c"
                        fig_pred.add_trace(
                            go.Bar(
                                x=["Tomorrow (Predicted)"],
                                y=[pred_value],
                                name="Tomorrow (Pred.)",
                                marker_color=bar_color,
                            )
                        )
                        # Change line
                        fig_pred.add_trace(
                            go.Scatter(
                                x=["Today", "Tomorrow (Predicted)"],
                                y=[todays_high, pred_value],
                                mode="lines+markers",
                                name="Change",
                                line=dict(color=bar_color, width=3, dash="dot"),
                                marker=dict(size=9, color=bar_color),
                                showlegend=False,
                            )
                        )
                        # Value annotations (dark label for readability)
                        fig_pred.add_annotation(
                            x="Today",
                            y=todays_high,
                            yshift=10,
                            text=f"${todays_high:,.2f}",
                            showarrow=False,
                            font=dict(color="black"),
                        )
                        fig_pred.add_annotation(
                            x="Tomorrow (Predicted)",
                            y=pred_value,
                            yshift=10,
                            text=f"${pred_value:,.2f} {'▲' if direction_up else '▼'}",
                            showarrow=False,
                            font=dict(color="black"),
                        )
                        # Delta chip
                        fig_pred.add_annotation(
                            x=0.5,
                            y=max(todays_high, pred_value) * 1.002,
                            text=f"{'UP' if direction_up else 'DOWN'}  ({delta:+.2f} USD | {delta_pct:+.2f}%)",
                            showarrow=False,
                            font=dict(color="white"),
                            bgcolor="rgba(0,0,0,0.6)",
                            xanchor="center",
                        )
                        fig_pred.update_layout(
                            title="Price Prediction",
                            xaxis_title="",
                            yaxis_title="Price (USD)",
                            barmode="group",
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.1,
                                xanchor="center",
                                x=0.5,
                            ),
                            height=400,
                            plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(
                                range=[
                                    min(todays_high, pred_value) * 0.995,
                                    max(todays_high, pred_value) * 1.01,
                                ]
                            ),
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)

                        st.caption(
                            "Model output is informational only and not financial advice."
                        )
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as ex:
                st.error(f"Error contacting prediction API: {ex}")

    st.markdown("---")
    st.caption(
        f"API: {API_BASE_URL} • Data: Kraken (analytics), CoinGecko (overview) • Page: bitcoin_front.py"
    )


if __name__ == "__main__":
    display_bitcoin_front()
