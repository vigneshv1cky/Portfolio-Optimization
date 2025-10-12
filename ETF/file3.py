import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="US Sector → Stock Buy/Sell Dashboard", layout="wide")

# ------------------------
# ETF setup
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
# Hardcoded Sector Holdings (Top 10)
# ------------------------
SECTOR_HOLDINGS = {
    "XLC": ["GOOGL", "META", "NFLX", "DIS", "TMUS", "CHTR", "VZ", "CMCSA", "TTWO", "T"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "MO", "KMB", "STZ"],
    "XLE": ["XOM", "CVX", "SLB", "EOG", "COP", "MPC", "PSX", "HAL", "VLO", "BKR"],
    "XLF": ["JPM", "BAC", "GS", "MS", "WFC", "BLK", "C", "AXP", "SPGI", "SCHW"],
    "XLV": ["UNH", "LLY", "JNJ", "MRK", "ABBV", "TMO", "DHR", "BMY", "PFE", "AMGN"],
    "XLI": ["HON", "CAT", "GE", "UPS", "RTX", "DE", "LMT", "BA", "MMM", "NSC"],
    "XLB": ["LIN", "SHW", "APD", "ECL", "NEM", "CTVA", "DD", "FCX", "MLM", "VMC"],
    "XLRE": ["PLD", "AMT", "EQIX", "SPG", "O", "CBRE", "CCI", "PSA", "WELL", "VICI"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PCG", "PEG"],
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "CSCO", "AMD", "ORCL", "INTC"],
}

# ------------------------
# Header
# ------------------------
st.title("📈 US Sector → Stock Buy/Sell Dashboard")
st.caption("Analyze sector momentum, choose ranking period, and see BUY/SELL signals for top holdings.")

st.markdown("---")
st.markdown(
    """
**How to use:**
1. Select a ranking period (1W–1Y).
2. The app picks top-performing sectors based on that metric.
3. It analyzes the top 10 holdings in each chosen sector.
4. Each stock gets a **BUY / HOLD / SELL** signal based on trend & momentum.
5. Visualize trends or download CSVs for deeper analysis.
"""
)
st.markdown("---")

# ------------------------
# Load ETF data (robust)
# ------------------------
lookback_period = "2y"
st.write(f"Fetching {lookback_period} of ETF price data...")

tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]


@st.cache_data(ttl=3600)
def load_prices_safe(tickers, period="2y"):
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    data = data.dropna(how="all")

    # If SPY (benchmark) missing, fetch separately
    for t in tickers:
        if t not in data.columns:
            st.warning(f"⚠️ {t} missing, retrying individually...")
            try:
                single = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False)["Close"]
                data[t] = single
            except Exception as e:
                st.error(f"❌ Could not fetch {t}: {e}")
    return data.dropna(how="all")


prices = load_prices_safe(tickers, lookback_period)


# ------------------------
# Helper functions
# ------------------------
def pct_return(series, days):
    if len(series) < days + 1:
        return np.nan
    return series.iloc[-1] / series.iloc[-(days + 1)] - 1


def moving_average(series, window):
    return series.rolling(window).mean()


# ------------------------
# Compute sector metrics
# ------------------------

metrics = []
spy = prices[BENCHMARK]
spy_ratio = prices.div(spy, axis=0)

for sector, ticker in SECTOR_ETFS.items():
    s = prices[ticker].dropna()
    if s.empty:
        continue
    rets = s.pct_change().dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252)

    row = {
        "Sector": sector,
        "Ticker": ticker,
        "1W": pct_return(s, 5),
        "1M": pct_return(s, 21),
        "3M": pct_return(s, 63),
        "6M": pct_return(s, 126),
        "YTD": (s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1),
        "1Y": pct_return(s, 252),
        "Above 100DMA": float(s.iloc[-1] > moving_average(s, 100).iloc[-1]),
        "Vol (20D, ann)": vol,
        "RS 3M vs SPY": pct_return(spy_ratio[ticker].dropna(), 63),
    }
    metrics.append(row)

df = pd.DataFrame(metrics)
st.subheader("📊 Sector Metrics")
st.dataframe(df, width="stretch")

# ------------------------
# Identify top sectors (user-selectable ranking)
# ------------------------
top_n = st.slider("Select number of top sectors to analyze", 1, 10, 3)

ranking_choice = st.selectbox("Select ranking period for top sectors:", ["1W", "1M", "3M", "6M", "YTD", "1Y"], index=2)

top_sectors = df.nlargest(top_n, ranking_choice)["Ticker"].tolist()
st.info(f"Ranking sectors by **{ranking_choice}** performance.")
st.write(f"Top {top_n} sectors: {', '.join(top_sectors)}")

# ------------------------
# Get stocks for top sectors
# ------------------------
stocks = sorted(set(sum([SECTOR_HOLDINGS[t] for t in top_sectors if t in SECTOR_HOLDINGS], [])))
st.write(f"Analyzing {len(stocks)} stocks from top sectors...")

stock_data = yf.download(stocks, period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]

# ------------------------
# Compute stock metrics + signals
# ------------------------
stock_metrics = []
for ticker in stock_data.columns:
    s = stock_data[ticker].dropna()
    if len(s) < 100:
        continue
    rets = s.pct_change().dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252)
    ma100 = moving_average(s, 100).iloc[-1]

    sector = next((k for k, v in SECTOR_HOLDINGS.items() if ticker in v), "Unknown")

    m1 = pct_return(s, 21)
    m3 = pct_return(s, 63)
    ytd = s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1
    above = float(s.iloc[-1] > ma100)

    # Simple buy/sell logic
    if above == 1 and m3 > 0.05 and m1 > 0:
        signal = "BUY ✅"
    elif above == 0 and m3 < -0.05:
        signal = "SELL ❌"
    else:
        signal = "HOLD ➖"

    stock_metrics.append(
        {
            "Sector": sector,
            "Ticker": ticker,
            "1M": m1,
            "3M": m3,
            "YTD": ytd,
            "Vol (20D, ann)": vol,
            "Above 100DMA": above,
            "Signal": signal,
        }
    )

stock_df = pd.DataFrame(stock_metrics)

# ------------------------
# Display stock dashboard
# ------------------------
st.subheader("📈 Stock Buy/Sell Signals")
st.dataframe(stock_df.sort_values(["Sector", "Signal"], ascending=[True, False]), width="stretch")
st.download_button(
    "Download Stock Signals (CSV)",
    stock_df.to_csv(index=False).encode(),
    file_name="stock_signals.csv",
    mime="text/csv",
)

# ------------------------
# Visualization per sector
# ------------------------
st.subheader("📉 Stock Price Charts")
selected_sector = st.selectbox("Select sector to view stock trends", top_sectors)
sector_stocks = SECTOR_HOLDINGS.get(selected_sector, [])
for t in sector_stocks:
    if t not in stock_data.columns:
        continue
    s = stock_data[t]
    ma50 = s.rolling(50).mean()
    ma100 = s.rolling(100).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=f"{t} Price"))
    fig.add_trace(go.Scatter(x=s.index, y=ma50, mode="lines", name="50 DMA", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=s.index, y=ma100, mode="lines", name="100 DMA", line=dict(dash="dot")))
    fig.update_layout(
        title=f"{t} — {selected_sector}",
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
    "**App Summary:** Top sectors are ranked by the selected period (1W–1Y). "
    "Stocks in those sectors are scored for trend strength and short-term momentum "
    "to generate BUY, HOLD, or SELL signals."
)
