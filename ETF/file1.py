import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

st.set_page_config(page_title="US Sector Dashboard", layout="wide")

# ------------------------
# Setup
# ------------------------
SECTOR_ETFS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Information Technology": "XLK",
}
BENCHMARK = "SPY"

# ------------------------
# Streamlit UI
# ------------------------
st.title("📈 US Sector Dashboard")
st.caption("Displays SPDR sector ETF metrics — trend, volatility, and returns.")

# ------------------------
# Load Data
# ------------------------
lookback_period = "2y"  # fixed 2-year default
st.write(f"Fetched last {lookback_period} of data")

tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
data = yf.download(tickers, period=lookback_period, interval="1d", auto_adjust=True, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    data = data["Close"]
prices = data.dropna(how="all")

st.write(f"Loaded {len(prices)} daily bars from {prices.index[0].date()} to {prices.index[-1].date()}.")

# ------------------------
# Compute metrics
# ------------------------
metrics = []
spy = prices[BENCHMARK]
spy_ratio = prices.div(spy, axis=0)

for sector, ticker in SECTOR_ETFS.items():
    s = prices[ticker].dropna()
    if s.empty:
        continue

    def pct_return(series, days):
        if len(series) < days + 1:
            return np.nan
        return series.iloc[-1] / series.iloc[-(days + 1)] - 1

    def moving_average(series, window):
        return series.rolling(window).mean()

    rets = s.pct_change().dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252)

    row = {
        "Sector": sector,
        "Ticker": ticker,
        "1W": pct_return(s, 5),
        "1M": pct_return(s, 21),
        "3M": pct_return(s, 63),
        "6M": pct_return(s, 126),
        "YTD": (
            (s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1)
            if (s.index.year == s.index[-1].year).any()
            else np.nan
        ),
        "1Y": pct_return(s, 252),
        "Above 50DMA": float(s.iloc[-1] > moving_average(s, 50).iloc[-1]) if len(s) >= 50 else np.nan,
        "Above 100DMA": float(s.iloc[-1] > moving_average(s, 100).iloc[-1]) if len(s) >= 100 else np.nan,
        "20D Vol (ann)": vol,
        "RS 3M vs SPY": pct_return(spy_ratio[ticker].dropna(), 63),
        "RS 1M vs SPY": pct_return(spy_ratio[ticker].dropna(), 21),
    }
    metrics.append(row)

df = pd.DataFrame(metrics)

# ------------------------
# Display table
# ------------------------
st.subheader("Sector Metrics")
st.dataframe(
    df[
        [
            "Sector",
            "Ticker",
            "1W",
            "1M",
            "3M",
            "6M",
            "YTD",
            "1Y",
            "Above 50DMA",
            "Above 100DMA",
            "20D Vol (ann)",
            "RS 1M vs SPY",
            "RS 3M vs SPY",
        ]
    ],
    width="stretch",
)

st.download_button(
    "Download table (CSV)", df.to_csv(index=False).encode(), file_name="sector_metrics.csv", mime="text/csv"
)

# ------------------------
# Plot each sector
# ------------------------
st.subheader("📊 Sector Price Trends (Interactive)")

for t in df["Ticker"]:
    ma50 = prices[t].rolling(50).mean()
    ma100 = prices[t].rolling(100).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices[t], mode="lines", name=f"{t} Price"))
    fig.add_trace(go.Scatter(x=prices.index, y=ma50, mode="lines", name="50 DMA", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=prices.index, y=ma100, mode="lines", name="100 DMA", line=dict(dash="dot")))

    fig.update_layout(
        title=f"{t} — {df.loc[df['Ticker'] == t, 'Sector'].values[0]}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# ------------------------
# Notes
# ------------------------
st.markdown("---")
st.markdown(
    "**How to use:** Review sector trends, volatility, and relative strength vs S&P 500. "
    "You can optionally filter to only show sectors above their 100-day moving average."
)
