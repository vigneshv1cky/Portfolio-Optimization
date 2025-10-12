import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------------------
# Streamlit setup
# ------------------------
st.set_page_config(page_title="US Sector → Stock Buy/Sell Dashboard", layout="wide")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("📊 Control Panel")
top_n = st.sidebar.slider("Top sectors to analyze", 1, 10, 3)
ranking_choice = st.sidebar.selectbox("Ranking period", ["1W", "1M", "3M", "6M", "YTD", "1Y"], index=2)
st.sidebar.markdown("---")
st.sidebar.info("Built using Yahoo Finance data via yfinance.")

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
# Load Data
# ------------------------
@st.cache_data(ttl=3600)
def load_prices(tickers, period="2y"):
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    return data.dropna(how="all")


tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
prices = load_prices(tickers)


# ------------------------
# Metrics Calculation
# ------------------------
def pct_return(series, days):
    if len(series) < days + 1:
        return np.nan
    return series.iloc[-1] / series.iloc[-(days + 1)] - 1


spy = prices[BENCHMARK]
spy_ratio = prices.div(spy, axis=0)
metrics = []

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
        "Vol (20D, ann)": vol,
        "RS 3M vs SPY": pct_return(spy_ratio[ticker], 63),
    }
    metrics.append(row)

df = pd.DataFrame(metrics)

# ------------------------
# Dashboard Header
# ------------------------
st.title("📈 US Sector → Stock Buy/Sell Dashboard")
st.markdown("### Sector Overview")

# KPI Cards
cols = st.columns(3)
cols[0].metric("Best Sector (3M)", df.loc[df["3M"].idxmax(), "Sector"], f"{df['3M'].max()*100:.1f}%")
cols[1].metric("Worst Sector (3M)", df.loc[df["3M"].idxmin(), "Sector"], f"{df['3M'].min()*100:.1f}%")
cols[2].metric("Benchmark (SPY, 3M)", f"{pct_return(spy, 63)*100:.1f}%", "")

st.markdown("---")

# ------------------------
# Tabs
# ------------------------
tab1, tab2, tab3 = st.tabs(["📊 Sector Ranking", "📈 Stock Signals", "📉 Charts"])

with tab1:
    st.subheader(f"Sector Ranking ({ranking_choice})")

    # Multiply by 100 to show percentages
    df_plot = df.copy()
    df_plot[ranking_choice] = df_plot[ranking_choice] * 100

    fig = px.bar(
        df_plot.sort_values(ranking_choice, ascending=False),
        x="Sector",
        y=ranking_choice,
        color=ranking_choice,
        color_continuous_scale="RdYlGn",
        labels={ranking_choice: f"{ranking_choice} (%)"},  # update axis label
    )

    fig.update_layout(
        xaxis=dict(tickangle=90), height=500, template="plotly_white", margin=dict(l=40, r=40, t=40, b=120)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Numeric formatting for table
    num_cols = df.select_dtypes(include=[np.number]).columns

    st.markdown("---")
    st.subheader(f"Returns")
    st.dataframe(df.style.format({col: "{:.2%}" for col in num_cols}), use_container_width=True)


with tab2:
    st.subheader("📈 Stock Buy/Sell Signals")

    # Determine top sectors based on user selection
    top_sectors = df.nlargest(top_n, ranking_choice)["Ticker"].tolist()
    st.info(f"Analyzing top {top_n} sectors ranked by **{ranking_choice}**: {', '.join(top_sectors)}")

    # ------------------------
    # Hardcoded Sector Holdings (top 10 per ETF)
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

    # Collect all unique stocks from top sectors
    stocks = sorted(set(sum([SECTOR_HOLDINGS[t] for t in top_sectors if t in SECTOR_HOLDINGS], [])))
    st.write(f"Fetching 1-year data for {len(stocks)} stocks...")

    stock_data = yf.download(stocks, period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]

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
    # Compute metrics and signals
    # ------------------------
    stock_metrics = []
    for ticker in stock_data.columns:
        s = stock_data[ticker].dropna()
        if len(s) < 100:
            continue
        rets = s.pct_change().dropna()
        vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252)
        ma100 = moving_average(s, 100).iloc[-1]

        # Identify which sector it belongs to
        sector = next((k for k, v in SECTOR_HOLDINGS.items() if ticker in v), "Unknown")

        m1 = pct_return(s, 21)
        m3 = pct_return(s, 63)
        ytd = s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1
        above = float(s.iloc[-1] > ma100)

        # Simple Buy/Sell logic
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
                "Above 100DMA": bool(above),
                "Signal": signal,
            }
        )

    stock_df = pd.DataFrame(stock_metrics)

    # ------------------------
    # Display in dashboard
    # ------------------------
    if stock_df.empty:
        st.warning("No stock data available.")
    else:
        # Color-code by Signal
        def highlight_signal(val):
            if "BUY" in val:
                color = "#C8E6C9"  # light green
            elif "SELL" in val:
                color = "#FFCDD2"  # light red
            else:
                color = "#F5F5F5"  # gray
            return f"background-color: {color}"

        st.dataframe(
            stock_df.style.format(
                {"1M": "{:.2%}", "3M": "{:.2%}", "YTD": "{:.2%}", "Vol (20D, ann)": "{:.2f}"}
            ).applymap(highlight_signal, subset=["Signal"]),
            use_container_width=True,
        )

        st.download_button(
            "Download Stock Signals (CSV)",
            stock_df.to_csv(index=False).encode(),
            file_name="stock_signals.csv",
            mime="text/csv",
        )

with tab3:
    st.subheader("📉 Sector & Stock Charts")

    # Choose sector first
    chosen_sector = st.selectbox("Select Sector", df["Sector"])
    etf_ticker = SECTOR_ETFS[chosen_sector]
    sector_price = prices[etf_ticker]

    # Moving averages for ETF
    ma50 = sector_price.rolling(50).mean()
    ma100 = sector_price.rolling(100).mean()

    # Plot sector ETF
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sector_price.index, y=sector_price, name=f"{etf_ticker} Price", mode="lines"))
    fig.add_trace(go.Scatter(x=sector_price.index, y=ma50, name="50 DMA", mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=sector_price.index, y=ma100, name="100 DMA", mode="lines", line=dict(dash="dot")))
    fig.update_layout(height=400, template="plotly_white", title=f"{chosen_sector} Sector ETF ({etf_ticker})")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Individual Stock Trend")

    # Choose stock within selected sector
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

    stocks = SECTOR_HOLDINGS.get(etf_ticker, [])
    selected_stock = st.selectbox("Select Stock", stocks)

    stock_data = yf.download(selected_stock, period="1y", interval="1d", auto_adjust=True, progress=False)["Close"]
    ma50_stock = stock_data.rolling(50).mean()
    ma100_stock = stock_data.rolling(100).mean()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data, name=f"{selected_stock} Price", mode="lines"))
    fig2.add_trace(go.Scatter(x=stock_data.index, y=ma50_stock, name="50 DMA", mode="lines", line=dict(dash="dash")))
    fig2.add_trace(go.Scatter(x=stock_data.index, y=ma100_stock, name="100 DMA", mode="lines", line=dict(dash="dot")))
    fig2.update_layout(height=400, template="plotly_white", title=f"{selected_stock} — {chosen_sector}")
    st.plotly_chart(fig2, use_container_width=True)


st.markdown("---")
st.caption("Data source: Yahoo Finance | Updated hourly | © 2025 TradeFit Scan")
