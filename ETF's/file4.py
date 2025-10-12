import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(page_title="US Sector → Stock Buy/Sell Dashboard", layout="wide")

# =====================================================
# Sidebar Controls
# =====================================================
st.sidebar.title("📊 Control Panel")
top_n = st.sidebar.slider("Top sectors to analyze", 1, 10, 3)
ranking_choice = st.sidebar.selectbox("Sector Performance period", ["1W", "1M", "3M", "6M", "YTD", "1Y"], index=2)
st.sidebar.markdown("---")
st.sidebar.info("Built using Yahoo Finance data via yfinance.")

# =====================================================
# ETF setup
# =====================================================
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

# =====================================================
# Hardcoded Sector Holdings (Top 10)
# =====================================================
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


# =====================================================
# Load Data
# =====================================================
@st.cache_data(ttl=3600)
def load_prices(tickers, period="2y"):
    """Downloads historical stock prices for a list of tickers."""
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        # Handle multi-index if fetching single ticker, otherwise select 'Close'
        data = data["Close"]
    return data.dropna(how="all")


tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
prices = load_prices(tickers)


# =====================================================
# Helper Functions
# =====================================================
def pct_return(series, days):
    """Calculates percentage return over a specified number of trading days."""
    if len(series) < days + 1:
        return np.nan
    return series.iloc[-1] / series.iloc[-(days + 1)] - 1


def moving_average(series, window):
    """Calculates the simple moving average."""
    return series.rolling(window).mean()


# =====================================================
# Sector Metrics
# =====================================================
spy = prices[BENCHMARK]
# Calculate relative strength (RS) ratio of sector ETF price vs SPY price
spy_ratio = prices.div(spy, axis=0)
metrics = []

for sector, ticker in SECTOR_ETFS.items():
    s = prices.get(ticker)
    if s is None or s.empty:
        continue

    s = s.dropna()

    # Calculate annualized volatility based on the last 20 trading days
    rets = s.pct_change().dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252) if len(rets) >= 20 else np.nan

    # Calculate YTD return
    ytd_return = np.nan
    current_year_data = s[s.index.year == s.index[-1].year]
    if not current_year_data.empty:
        ytd_return = (s.iloc[-1] / current_year_data.iloc[0] - 1) if current_year_data.iloc[0] != 0 else np.nan

    row = {
        "Sector": sector,
        "Ticker": ticker,
        "1W": pct_return(s, 5),
        "1M": pct_return(s, 21),
        "3M": pct_return(s, 63),
        "6M": pct_return(s, 126),
        "YTD": ytd_return,
        "1Y": pct_return(s, 252),
        "Vol (20D, ann)": vol,
        "RS 3M vs SPY": pct_return(spy_ratio[ticker], 63),
    }
    metrics.append(row)

df = pd.DataFrame(metrics).dropna(subset=["1W", "1M", "3M", "6M", "1Y"], how="all")

# =====================================================
# Dashboard Header
# =====================================================
st.title("📈 US Sector → Stock Buy/Sell Dashboard")
st.markdown("### Sector Overview")

# Calculate header metrics dynamically based on ranking period
period_days = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "YTD": None, "1Y": 252}  # handled separately

cols = st.columns(3)


def shorten_name(name, max_len=18):
    """Shorten long sector names for metric display."""
    return (
        name
        if len(name) <= max_len
        else name.replace("Communication", "Comm.")
        .replace("Discretionary", "Disc.")
        .replace("Information", "Info.")
        .replace("Technology", "Tech")
        .replace("Consumer", "Cons.")
        .replace("Financials", "Fin.")
        .replace("Industrials", "Indus.")
        .replace("Utilities", "Utils")
        .replace("Materials", "Mat.")
    )


if ranking_choice == "YTD":
    current_year_data = spy[spy.index.year == spy.index[-1].year]
    spy_return = (spy.iloc[-1] / current_year_data.iloc[0] - 1) * 100 if not current_year_data.empty else np.nan
else:
    days = period_days.get(ranking_choice, 63)
    spy_return = pct_return(spy, days) * 100 if len(spy) >= days else np.nan

if not df.empty:
    max_val = df[ranking_choice].max()
    min_val = df[ranking_choice].min()

    cols[0].metric(
        f"Best Sector ({ranking_choice})",
        shorten_name(df.loc[df[ranking_choice].idxmax(), "Sector"]) if not df[ranking_choice].isnull().all() else "N/A",
        f"{max_val*100:.1f}%" if not np.isnan(max_val) else "N/A",
    )
    cols[1].metric(
        f"Worst Sector ({ranking_choice})",
        shorten_name(df.loc[df[ranking_choice].idxmin(), "Sector"]) if not df[ranking_choice].isnull().all() else "N/A",
        f"{min_val*100:.1f}%" if not np.isnan(min_val) else "N/A",
    )
else:
    cols[0].metric(f"Best Sector ({ranking_choice})", "N/A", "N/A")
    cols[1].metric(f"Worst Sector ({ranking_choice})", "N/A", "N/A")

cols[2].metric(f"Benchmark (SPY, {ranking_choice})", f"{spy_return:.1f}%" if not np.isnan(spy_return) else "N/A", "")


# =====================================================
# Tabs
# =====================================================
tab1, tab2, tab3 = st.tabs(["📊 Sector Ranking", "📈 Stock Signals", "📉 Charts"])

# -----------------------------------------------------
# TAB 1: Sector Ranking
# -----------------------------------------------------
with tab1:
    st.subheader(f"Sector Ranking ({ranking_choice})")

    if not df.empty:
        df_plot = df.copy()
        df_plot[ranking_choice] = df_plot[ranking_choice] * 100

        fig = px.bar(
            df_plot.sort_values(ranking_choice, ascending=False),
            x="Sector",
            y=ranking_choice,
            color=ranking_choice,
            color_continuous_scale="RdYlGn",
            labels={ranking_choice: f"{ranking_choice} (%)"},
            title=f"Sector Performance over {ranking_choice}",
        )

        # Removed fixed height=500 to avoid deprecation warning
        fig.update_layout(xaxis=dict(tickangle=45), template="plotly_white", margin=dict(l=40, r=40, t=60, b=120))

        st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

        num_cols = df.select_dtypes(include=[np.number]).columns
        st.markdown("---")
        st.subheader("Returns Table")
        st.dataframe(df.style.format({col: "{:.2%}" for col in num_cols}), use_container_width=True)
        st.write("RS is Relative Strength(How a sector ETF has performed relative to the benchmark SPY)")
        st.write(
            "Vol (20D, ann) is 20-day annualized volatility. It measures how much a sector’s or stock’s price fluctuates (its risk) based on the last 20 trading days (about one month)."
        )
    else:
        st.warning("No sector data available for ranking.")


# -----------------------------------------------------
# TAB 2: Stock Signals
# -----------------------------------------------------
with tab2:
    st.subheader("📈 Stock Buy/Sell Signals")

    if not df.empty:
        # Identify top sectors based on the control panel settings
        top_sectors_df = df.nlargest(top_n, ranking_choice)
        top_sector_tickers = top_sectors_df["Ticker"].tolist()

        st.info(f"Analyzing top **{top_n}** sectors ranked by **{ranking_choice}**: {', '.join(top_sector_tickers)}")

        # Collect unique stocks from the holdings of the top sectors
        stocks = []
        for t in top_sector_tickers:
            stocks.extend(SECTOR_HOLDINGS.get(t, []))
        stocks = sorted(list(set(stocks)))

        if not stocks:
            st.warning("No stocks found in the holdings of the selected top sectors.")
        else:
            st.write(
                f"Fetching 1-year data for **{len(stocks)}** stocks — taking the **top 10 stocks** from each of the **top performing sectors**."
            )

            try:
                # Download stock data (caching handled by Streamlit's cache decorator in load_prices,
                # but yfinance is used directly here for the specific stock list)
                stock_data_raw = yf.download(stocks, period="1y", interval="1d", auto_adjust=True, progress=False)

                if isinstance(stock_data_raw.columns, pd.MultiIndex):
                    stock_data = stock_data_raw["Close"]
                else:
                    # Case where only one stock is downloaded, resulting in a single Series/DataFrame
                    stock_data = (
                        stock_data_raw["Close"].to_frame()
                        if isinstance(stock_data_raw, pd.DataFrame)
                        else stock_data_raw
                    )

                stock_metrics = []
                for ticker in stock_data.columns:
                    s = stock_data[ticker].dropna()
                    if len(s) < 100:
                        # Skip if not enough data for 100 DMA
                        continue

                    # Calculate metrics
                    rets = s.pct_change().dropna()
                    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252) if len(rets) >= 20 else np.nan
                    ma100 = moving_average(s, 100).iloc[-1]

                    # Determine the sector(s) the stock belongs to (for display)
                    sector_list = [
                        name for name, t in SECTOR_ETFS.items() if t in SECTOR_HOLDINGS and ticker in SECTOR_HOLDINGS[t]
                    ]
                    sector = ", ".join(sector_list) if sector_list else "Unknown"

                    # Return calculations
                    m1 = pct_return(s, 21)
                    m3 = pct_return(s, 63)
                    ytd = (
                        s.iloc[-1] / s[s.index.year == s.index[-1].year].iloc[0] - 1
                        if s[s.index.year == s.index[-1].year].iloc[0] != 0
                        else np.nan
                    )
                    above = s.iloc[-1] > ma100

                    # Signal Logic: Simple Trend Following
                    # BUY: Above 100DMA AND strong 3M momentum (> 5%) AND positive 1M momentum
                    if above and m3 > 0.05 and m1 > 0:
                        signal = "BUY ✅"
                    # SELL: Below 100DMA AND significant 3M loss (< -5%)
                    elif not above and m3 < -0.05:
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

                if stock_df.empty:
                    st.warning(
                        "Could not generate signals. Check if yfinance data is available for the selected stocks."
                    )
                else:
                    # Styling function
                    def highlight_signal(val):
                        if "BUY" in val:
                            color = "rgb(200, 230, 201)"  # Light Green
                        elif "SELL" in val:
                            color = "rgb(255, 205, 210)"  # Light Red
                        else:
                            color = "rgb(240, 240, 240)"  # Light Gray
                        return f"background-color: {color}"

                    st.dataframe(
                        stock_df.style.format(
                            {"1M": "{:.2%}", "3M": "{:.2%}", "YTD": "{:.2%}", "Vol (20D, ann)": "{:.2f}"}
                        ).map(highlight_signal, subset=["Signal"]),
                        use_container_width=True,
                    )

                    st.download_button(
                        "Download Stock Signals (CSV)",
                        stock_df.to_csv(index=False).encode(),
                        file_name="stock_signals.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error fetching or processing stock data: {e}")
    else:
        st.warning("Cannot generate stock signals because sector data is unavailable.")


# -----------------------------------------------------
# TAB 3: Sector & Stock Charts
# -----------------------------------------------------
with tab3:
    st.subheader("📉 Sector & Stock Charts")

    if not df.empty:
        chosen_sector = st.selectbox("Select Sector", df["Sector"].unique())
        etf_ticker = SECTOR_ETFS[chosen_sector]
        sector_price = prices.get(etf_ticker)

        if sector_price is not None and not sector_price.empty:
            sector_price = sector_price.dropna()
            ma50 = sector_price.rolling(50).mean()
            ma100 = sector_price.rolling(100).mean()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sector_price.index, y=sector_price, name=f"{etf_ticker} Price", mode="lines"))
            fig.add_trace(
                go.Scatter(x=ma50.index, y=ma50, name="50 DMA", mode="lines", line=dict(dash="dash", color="orange"))
            )
            fig.add_trace(
                go.Scatter(x=ma100.index, y=ma100, name="100 DMA", mode="lines", line=dict(dash="dot", color="red"))
            )

            fig.update_layout(
                template="plotly_white", title=f"{chosen_sector} Sector ETF ({etf_ticker}) Price vs. Moving Averages"
            )
            st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

            st.markdown("### 🔍 Individual Stock Trend")

            stocks = SECTOR_HOLDINGS.get(etf_ticker, [])
            if not stocks:
                st.warning("No stock data found for this sector in the hardcoded list.")
            else:
                selected_stock = st.selectbox("Select Stock", stocks)

                # Fetching 1y data for selected stock
                try:
                    # Download raw data (returns a DataFrame)
                    data_raw = yf.download(selected_stock, period="1y", interval="1d", auto_adjust=True, progress=False)

                    if data_raw.empty:
                        st.warning(f"No price data could be retrieved from Yahoo Finance for {selected_stock}.")
                        # Exit early to prevent further errors
                        st.stop()

                    # Safely extract the price data (usually "Close")
                    if "Close" in data_raw.columns:
                        stock_data = data_raw["Close"].dropna()
                    elif not data_raw.columns.empty:
                        # Fallback to the first column if "Close" is missing (e.g. sometimes happens with adjusted data)
                        stock_data = data_raw.iloc[:, 0].dropna()
                    else:
                        st.warning(f"Could not find a price column in the retrieved data for {selected_stock}.")
                        st.stop()

                    # Ensure it is a Series (it should be after slicing, but safer to check)
                    stock_data = stock_data.squeeze()

                    MIN_DATA_POINTS = 100  # Minimum data points required for 100 DMA

                    if len(stock_data) < MIN_DATA_POINTS:
                        st.warning(
                            f"Could not retrieve sufficient price data for {selected_stock}. Found {len(stock_data)} data points, but need at least {MIN_DATA_POINTS} for the 100-day moving average to plot."
                        )
                    else:
                        ma50_stock = stock_data.rolling(50).mean()
                        ma100_stock = stock_data.rolling(100).mean()

                        fig2 = go.Figure()
                        # Ensure the data passed to plotly is a series or clean array
                        fig2.add_trace(
                            go.Scatter(
                                x=stock_data.index, y=stock_data.values, name=f"{selected_stock} Price", mode="lines"
                            )
                        )
                        fig2.add_trace(
                            go.Scatter(
                                x=ma50_stock.index,
                                y=ma50_stock.values,
                                name="50 DMA",
                                mode="lines",
                                line=dict(dash="dash", color="orange"),
                            )
                        )
                        fig2.add_trace(
                            go.Scatter(
                                x=ma100_stock.index,
                                y=ma100_stock.values,
                                name="100 DMA",
                                mode="lines",
                                line=dict(dash="dot", color="red"),
                            )
                        )

                        fig2.update_layout(
                            template="plotly_white",
                            title=f"{selected_stock} — {chosen_sector} Price vs. Moving Averages",
                        )
                        st.plotly_chart(fig2, config={"displayModeBar": False}, use_container_width=True)
                except Exception as e:
                    st.error(f"An unexpected error occurred while processing data for {selected_stock}: {e}")
        else:
            st.warning(f"Price data for sector ETF {etf_ticker} is currently unavailable.")
    else:
        st.warning("No sector data is available to generate charts.")


st.markdown("---")
st.caption("Data source: Yahoo Finance | Updated hourly | © 2025 TradeFit Scan")
