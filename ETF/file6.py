import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# Streamlit setup
# =====================================================
st.set_page_config(page_title="US Market Drilldown Dashboard", layout="wide")

st.sidebar.title("📊 Control Panel")
ranking_choice = st.sidebar.selectbox("Performance Period", ["1W", "1M", "3M", "6M", "YTD", "1Y"], index=2)
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
ETF_TO_NAME = {v: k for k, v in SECTOR_ETFS.items()}
BENCHMARK = "SPY"

# =====================================================
# Sector → Subsector → Stock Hierarchy
# =====================================================
SECTOR_STRUCTURE = {
    "XLB": {
        "Chemicals": ["LIN", "SHW", "APD", "ECL", "DD", "PPG", "ALB", "CE", "IFF", "AVNT"],
        "Metals & Mining": ["FCX", "NEM", "NUE", "MLM", "VMC", "STLD", "X", "CLF", "AA", "TECK"],
        "Fertilizers & Ag Chemicals": ["CF", "MOS", "NTR", "SMG"],
        "Construction Materials": ["EXP", "SUM", "CRH", "VMC", "MLM"],
        "Containers & Packaging": ["BALL", "PKG", "IP", "WRK", "SEE"],
        "Precious Metals": ["GOLD", "AEM", "FNV", "WPM", "PAAS"],
    },
    "XLE": {
        "Integrated Oil & Gas": ["XOM", "CVX", "SHEL", "BP", "TTE"],
        "Oil & Gas E&P": ["COP", "EOG", "DVN", "PXD", "OXY", "APA", "MRO"],
        "Oilfield Services & Equipment": ["SLB", "HAL", "BKR", "FTI", "NOV"],
        "Refining & Marketing": ["MPC", "PSX", "VLO", "DINO"],
        "Midstream": ["EPD", "ENB", "KMI", "WMB", "TRGP"],
    },
    "XLK": {
        "Semiconductors": ["NVDA", "AMD", "INTC", "AVGO", "QCOM", "MU", "TXN", "ADI", "NXPI", "ARM"],
        "Semiconductor Equipment": ["ASML", "AMAT", "LRCX", "KLAC", "TER"],
        "Software & Services": ["MSFT", "CRM", "ADBE", "ORCL", "NOW", "SNOW", "PANW", "CRWD", "DDOG"],
        "IT Hardware & Peripherals": ["AAPL", "DELL", "HPQ", "STX", "WDC", "NTAP"],
        "Electronic Equip/Components": ["APH", "TEL", "GLW", "PH", "KEYS"],
        "Payments & Data Processing": ["V", "MA", "PYPL", "GPN", "FI"],
    },
    "XLF": {
        "Banks": ["JPM", "BAC", "C", "WFC", "USB", "PNC", "TFC", "COF"],
        "Capital Markets": ["MS", "GS", "BLK", "SCHW", "ICE", "CME", "NTRS", "BK"],
        "Insurance": ["PGR", "TRV", "AIG", "ALL", "MET", "PRU", "AFL", "HIG"],
        "Diversified Financials": ["BRK.B", "SPGI", "MCO", "FICO"],
        "Consumer Finance": ["AXP", "DFS", "SYF"],
        "Insurance Brokers": ["MMC", "AON", "WTW"],
    },
    "XLC": {
        "Interactive Media & Services": ["GOOGL", "META", "SNAP", "PINS", "SPOT"],
        "Media & Entertainment": ["DIS", "NFLX", "PARA", "WBD", "ROKU", "LYV"],
        "Video Games & Esports": ["TTWO", "EA", "U"],
        "Advertising": ["OMC", "IPG", "WPP"],
        "Telecom Carriers": ["TMUS", "VZ", "T", "CMCSA", "CHTR"],
    },
    "XLY": {
        "Retail & E-Commerce": ["AMZN", "HD", "LOW", "TJX", "TGT", "ROST", "BBY", "DG"],
        "Automobiles & EVs": ["TSLA", "GM", "F", "RIVN", "LCID"],
        "Apparel/Footwear": ["NKE", "LULU", "UAA", "TPR"],
        "Hotels/Leisure/Travel": ["BKNG", "MAR", "HLT", "H", "CCL", "RCL", "NCLH"],
        "Homebuilding": ["DHI", "LEN", "PHM", "NVR", "TOL"],
        "Specialty Retail (Auto Parts)": ["AZO", "ORLY", "AAP"],
        "Restaurants": ["MCD", "SBUX", "CMG", "YUM"],
    },
    "XLI": {
        "Aerospace & Defense": ["LMT", "RTX", "NOC", "GD", "HII", "BA"],
        "Transportation (Railroads)": ["UNP", "CSX", "NSC", "CP", "CNI"],
        "Air Freight & Logistics": ["UPS", "FDX", "CHRW", "EXPD"],
        "Airlines": ["DAL", "AAL", "UAL", "LUV"],
        "Machinery": ["CAT", "DE", "EMR", "ETN", "PH", "ROK", "MMM"],
        "Building Products & HVAC": ["JCI", "CARR", "TT", "MAS"],
        "Electrical Equipment": ["GE", "ABB", "AME", "AOS"],
    },
    "XLV": {
        "Pharma": ["LLY", "PFE", "MRK", "BMY", "JNJ", "AZN", "GSK"],
        "Biotech": ["AMGN", "VRTX", "GILD", "REGN", "BIIB"],
        "Healthcare Equipment": ["TMO", "DHR", "ABT", "BSX", "BAX", "SYK"],
        "Managed Care": ["UNH", "HUM", "CI", "ELV", "CNC"],
        "Life Sciences Tools": ["IQV", "MTD", "WAT", "BIO", "ILMN", "PKI"],
        "Distributors": ["MCK", "CAH", "COR"],  # COR = Cencora (formerly ABC)
        "Providers & Services": ["HCA", "UHS", "DGX", "LH"],
    },
    "XLP": {
        "Food & Beverage": ["KO", "PEP", "MDLZ", "KDP", "KHC", "GIS", "K"],
        "Tobacco": ["PM", "MO", "BTI"],
        "Food Retail": ["WMT", "COST", "KR", "ACI"],
        "Household Products": ["PG", "CL", "KMB", "CHD"],
        "Personal Care/Beauty": ["EL", "CLX", "CPRI"],  # EL = Estée Lauder
        "Beverage Alcohol": ["STZ", "BF.B", "SAM"],
    },
    "XLRE": {
        "Industrial REITs": ["PLD", "REXR", "TRNO"],
        "Data Center REITs": ["EQIX", "DLR"],
        "Retail REITs": ["SPG", "O", "FRT"],
        "Residential REITs": ["AVB", "EQR", "UDR", "AMH"],
        "Office REITs": ["BXP", "VNO", "ARE"],
        "Self-Storage REITs": ["PSA", "EXR", "CUBE"],
        "Specialty REITs": ["IRM", "WY"],  # WY is technically Timber REIT (also seen in Materials lists)
        "Healthcare REITs": ["WELL", "VTR", "MPW"],
    },
    "XLU": {
        "Electric Utilities": ["NEE", "DUK", "SO", "AEP", "ED", "PCG", "FE"],
        "Gas & Water Utilities": ["SRE", "XEL", "D", "PEG", "AEE", "AWK", "WTRG"],
        "Multi-Utilities & IPPs": ["ES", "EIX", "PNW", "NEP", "AY"],
    },
}


# =====================================================
# Helpers
# =====================================================
@st.cache_data(ttl=3600)
def download_close(tickers, period="1y", interval="1d"):
    """Returns a DataFrame of Close prices with columns = tickers (even for a single ticker)."""
    if isinstance(tickers, str):
        tickers_list = [tickers]
    else:
        tickers_list = tickers

    df = yf.download(tickers_list, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Normalize to Close-only
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        else:
            lvl0 = df.columns.levels[0][0]
            df = df.xs(lvl0, axis=1, level=0).select_dtypes(include=[np.number])
    else:
        if "Close" in df.columns:
            df = df[["Close"]]
        else:
            df = df.select_dtypes(include=[np.number]).iloc[:, :1]
        if len(tickers_list) == 1:
            df.columns = [tickers_list[0]]

    df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna(how="all")


def pct_return(series: pd.Series, days: int):
    if series is None or series.empty or len(series) < days + 1:
        return np.nan
    try:
        return series.iloc[-1] / series.iloc[-(days + 1)] - 1
    except Exception:
        return np.nan


# ===== Subsector helpers =====
def equal_weight_index(close_df: pd.DataFrame) -> pd.Series:
    """
    Build an equal-weighted index from a Close-price DataFrame (columns=tickers).
    Handles missing data by averaging daily % changes across available tickers.
    Returns a series rebased to 100.
    """
    if close_df is None or close_df.empty:
        return pd.Series(dtype=float)
    # daily cross-sectional mean return (skipna)
    ew_rets = close_df.pct_change().mean(axis=1, skipna=True)
    idx = (1 + ew_rets.fillna(0)).cumprod()
    # rebase to 100 at first valid point
    if not idx.empty:
        idx = idx / idx.iloc[0] * 100.0
    return idx


@st.cache_data(ttl=3600)
def get_subsector_prices(selected_sectors, period="2y", interval="1d"):
    """
    Download Close prices for all stocks in the subsectors belonging to the selected sectors.
    Returns a dict: {(SectorName, SubsectorName): Close-DF for its tickers}, and a union Close DF.
    """
    # collect all unique tickers across chosen sectors
    wanted = []
    sec_sub_map = {}  # (Sector, Subsector) -> list[tickers]
    for sec in selected_sectors:
        etf = SECTOR_ETFS.get(sec)
        if etf and etf in SECTOR_STRUCTURE:
            for sub, tickers in SECTOR_STRUCTURE[etf].items():
                sec_sub_map[(sec, sub)] = tickers
                wanted.extend(tickers)
    wanted = sorted(set(wanted))

    all_close = download_close(wanted, period=period, interval=interval)
    # build per-subsector DF (filtering to columns that actually returned)
    per_sub_dfs = {}
    for (sec, sub), tickers in sec_sub_map.items():
        cols = [t for t in tickers if t in all_close.columns]
        if cols:
            per_sub_dfs[(sec, sub)] = all_close[cols]
        else:
            per_sub_dfs[(sec, sub)] = pd.DataFrame()
    return per_sub_dfs, all_close


# =====================================================
# Load Data
# =====================================================
tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
prices = download_close(tickers, period="2y", interval="1d")

# =====================================================
# Compute Metrics
# =====================================================
if prices.empty or BENCHMARK not in prices.columns:
    st.error("No price data returned (possibly rate-limited). Please try again in a bit.")
    st.stop()

spy = prices[BENCHMARK]
spy_ratio = prices.div(spy, axis=0)
metrics = []

for sector, ticker in SECTOR_ETFS.items():
    if ticker not in prices.columns:
        continue
    s = prices[ticker].dropna()
    if s.empty:
        continue

    rets = s.pct_change().dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252) if len(rets) >= 20 else np.nan

    # YTD robust: first available close in current year
    cur_year = s.index[-1].year
    cy = s[s.index.year == cur_year]
    ytd_return = (s.iloc[-1] / cy.iloc[0] - 1) if not cy.empty else np.nan

    row = {
        "Sector": sector,
        "Ticker": ticker,
        "1W": pct_return(s, 5),
        "1M": pct_return(s, 21),
        "3M": pct_return(s, 63),
        "6M": pct_return(s, 126),
        "YTD": ytd_return,
        "1Y": pct_return(s, 252),
        "Vol (20D→Ann)": vol,
        "RS 3M vs SPY": pct_return(spy_ratio[ticker].dropna(), 63) if ticker in spy_ratio.columns else np.nan,
    }
    metrics.append(row)

df = pd.DataFrame(metrics)

# =====================================================
# TABS: Overview → Subsector → Stocks → Charts
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sector Overview", "🏗️ Subsector Breakdown", "📈 Stock Signals", "📉 Charts"])

# -----------------------------------------------------
# TAB 1 — Sector Overview + Recommendations
# -----------------------------------------------------
with tab1:
    st.header("📊 Sector Performance Overview")

    if df.empty:
        st.warning("No sector metrics available. Check data source / try again later.")
    else:
        df_plot = df.copy()
        df_plot[ranking_choice] = df_plot[ranking_choice] * 100

        fig = px.bar(
            df_plot.sort_values(ranking_choice, ascending=False),
            x="Sector",
            y=ranking_choice,
            color=ranking_choice,
            color_continuous_scale="RdYlGn",
            title=f"S&P 500 Sector Performance — {ranking_choice}",
        )
        fig.update_layout(
            xaxis=dict(tickangle=45),
            template="plotly_white",
            coloraxis_colorbar=dict(title=f"{ranking_choice} (%)"),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 💡 Sector Recommendations")
        # New sliders in Tab 1 (dynamic max based on how many sectors we have)
        max_sectors = max(1, len(df))
        colA, colB = st.columns(2)
        with colA:
            top_n_buy = st.slider("Strong sectors to buy", 1, max_sectors, min(3, max_sectors), key="tab1_top_buy")
        with colB:
            top_n_short = st.slider(
                "Weakest sectors to short", 1, max_sectors, min(3, max_sectors), key="tab1_top_short"
            )

        # Use the new slider values
        top_df = df.nlargest(top_n_buy, ranking_choice)
        worst_df = df.nsmallest(top_n_short, ranking_choice)

        st.success("Top performing sectors ({}): {}".format(ranking_choice, ", ".join(top_df["Sector"].tolist())))
        st.error("Weakest sectors ({}): {}".format(ranking_choice, ", ".join(worst_df["Sector"].tolist())))

        # Keep using the top sector as the default selection downstream
        recommended_sector = top_df.iloc[0]["Sector"] if not top_df.empty else df["Sector"].iloc[0]

        st.markdown("### 🎯 Manual Selection")
        all_sectors = df["Sector"].tolist()
        default_idx = all_sectors.index(recommended_sector) if recommended_sector in all_sectors else 0

        selected_sector = st.selectbox(
            "Sector Chosen:", options=all_sectors, index=default_idx, key="tab1_single_sector"
        )

        # Save as a one-item list so downstream code (which expects a list) still works
        st.session_state["selected_sectors"] = [selected_sector]

        # Use the one sector chosen above (fallback to recommended if missing)
        chart_sector = st.session_state.get("selected_sectors", [recommended_sector])[0]

        # Controls (no sector picker needed anymore)
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            chart_period = st.radio(
                "Period", ["3M", "6M", "1Y", "2Y"], index=2, horizontal=True, key="tab1_sector_period"
            )
        with c2:
            show_ma = st.checkbox("Show 50/100-day MAs", value=True, key="tab1_show_ma")
        with c3:
            overlay_spy = st.checkbox("Overlay SPY (benchmark)", value=True, key="tab1_overlay_spy")

        # Map period -> approx trading days for slicing
        period_days = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
        days = period_days[chart_period]

        # Pull series
        etf_ticker = SECTOR_ETFS.get(chart_sector)
        sector_series = prices.get(etf_ticker) if etf_ticker in prices.columns else None
        spy_series = prices.get(BENCHMARK) if BENCHMARK in prices.columns else None

        if sector_series is None or sector_series.empty:
            st.warning(f"No price data for sector ETF {etf_ticker}.")
        else:
            # Slice by days if we have enough data
            s = sector_series.dropna()
            s = s.iloc[-days:] if len(s) > days else s

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{etf_ticker} Price", mode="lines"))

            if show_ma:
                if len(s) >= 50:
                    ma50 = s.rolling(50).mean()
                    fig_line.add_trace(
                        go.Scatter(x=ma50.index, y=ma50.values, name="50 DMA", mode="lines", line=dict(dash="dash"))
                    )
                if len(s) >= 100:
                    ma100 = s.rolling(100).mean()
                    fig_line.add_trace(
                        go.Scatter(x=ma100.index, y=ma100.values, name="100 DMA", mode="lines", line=dict(dash="dot"))
                    )

            if overlay_spy and spy_series is not None and not spy_series.empty:
                sp = spy_series.dropna()
                sp = sp.iloc[-days:] if len(sp) > days else sp
                # Normalize both to 100 at start for easier comparison
                try:
                    s_norm = s / s.iloc[0] * 100
                    sp_norm = sp / sp.iloc[0] * 100
                    fig_line.add_trace(
                        go.Scatter(x=sp_norm.index, y=sp_norm.values, name="SPY (rebased=100)", mode="lines")
                    )
                    # Also show the sector as rebased if overlay enabled, to match scale
                    fig_line.data[0].y = s_norm.values
                    fig_line.data[0].name = f"{etf_ticker} (rebased=100)"
                    y_title = "Rebased to 100"
                except Exception:
                    y_title = "Price"
            else:
                y_title = "Price"

            fig_line.update_layout(
                title=f"{chart_sector} — {etf_ticker} ({chart_period})",
                template="plotly_white",
                margin=dict(l=20, r=20, t=50, b=20),
                yaxis_title=y_title,
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # ---- NEW: Show the sector metrics DataFrame ----
            st.markdown("### 📄 Sector Metrics Table")
            st.dataframe(
                df.style.format(
                    {
                        "1W": "{:.2%}",
                        "1M": "{:.2%}",
                        "3M": "{:.2%}",
                        "6M": "{:.2%}",
                        "YTD": "{:.2%}",
                        "1Y": "{:.2%}",
                        "RS 3M vs SPY": "{:.2%}",
                        "Vol (20D→Ann)": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )
# -----------------------------------------------------
# TAB 2 — Subsector Breakdown
# -----------------------------------------------------

with tab2:
    st.header("🏗️ Subsector Performance")

    selected_sectors = st.session_state.get("selected_sectors", [])
    if not selected_sectors:
        st.warning("Please select at least one sector in Tab 1.")
        st.stop()

    # Pull prices for all subsector constituents just once
    per_sub_dfs, all_sub_close = get_subsector_prices(selected_sectors, period="2y", interval="1d")

    if all_sub_close is None or all_sub_close.empty:
        st.warning(
            "No stock data available for selected sectors (rate limit or symbols). Try different sectors/period."
        )
        st.stop()

    # Build SPY ref for RS calculations
    spy_series = prices[BENCHMARK].dropna() if BENCHMARK in prices.columns else pd.Series(dtype=float)

    # Compute metrics per (Sector, Subsector) using equal-weight index
    sub_metrics = []
    sub_index_store = {}  # keep the built index for optional detail chart

    for (sec, sub), df_close in per_sub_dfs.items():
        if df_close is None or df_close.empty:
            continue
        idx = equal_weight_index(df_close).dropna()
        if idx.empty:
            continue

        # Annualized vol from daily returns of the index
        idx_rets = idx.pct_change().dropna()
        vol = idx_rets.rolling(20).std().iloc[-1] * np.sqrt(252) if len(idx_rets) >= 20 else np.nan

        # YTD from first available in current year
        cur_year = idx.index[-1].year
        idx_y = idx[idx.index.year == cur_year]
        ytd = (idx.iloc[-1] / idx_y.iloc[0] - 1) if not idx_y.empty else np.nan

        row = {
            "Sector": sec,
            "Subsector": sub,
            "Constituents": len(df_close.columns),
            "1W": pct_return(idx, 5),
            "1M": pct_return(idx, 21),
            "3M": pct_return(idx, 63),
            "6M": pct_return(idx, 126),
            "YTD": ytd,
            "1Y": pct_return(idx, 252),
            "Vol (20D→Ann)": vol,
            "RS 3M vs SPY": (
                pct_return((idx / spy_series.reindex(idx.index)).dropna(), 63) if not spy_series.empty else np.nan
            ),
        }
        sub_metrics.append(row)
        sub_index_store[(sec, sub)] = idx

    df_subperf = pd.DataFrame(sub_metrics)

    if df_subperf.empty:
        st.info("No subsector performance could be computed for your selections.")
        st.stop()

    # ---- Chart like Tab1 (sorted by current ranking_choice) ----
    st.markdown("### 📊 Subsector Bar Chart (Horizontal)")
    plot_col = ranking_choice
    plot_df = df_subperf.copy()
    plot_df[plot_col] = plot_df[plot_col] * 100

    # Keep value-sorted order (largest at top)
    y_order = plot_df.sort_values(plot_col, ascending=False)["Subsector"].drop_duplicates().tolist()

    vmin = float(plot_df[plot_col].min())
    vmax = float(plot_df[plot_col].max())
    max_abs = max(abs(vmin), abs(vmax))

    # Choose color mode
    mixed = vmin < 0 < vmax
    pos_only = vmin >= 0
    neg_only = vmax <= 0

    # Prepare color column & scale
    color_col = plot_col
    color_scale = None
    range_color = None
    midpoint = None
    cbar_title = f"{plot_col} (%)"

    if mixed:
        color_scale = [(0, "red"), (0.5, "yellow"), (1, "green")]
        range_color = [-max_abs, max_abs]
        midpoint = 0
    elif pos_only:
        color_scale = "Greens"
        range_color = [0, vmax]
    else:  # neg_only
        # Color by magnitude so more negative => darker red
        color_col = "_loss_mag"
        plot_df[color_col] = -plot_df[plot_col]  # positive magnitudes
        color_scale = "Reds"
        range_color = [0, max_abs]
        cbar_title = "Loss magnitude (%)"

    fig_sub = px.bar(
        plot_df.sort_values(plot_col, ascending=False),
        y="Subsector",
        x=plot_df[plot_col],  # keep actual values on the axis
        color=plot_df[color_col],  # but color by chosen column
        color_continuous_scale=color_scale,
        range_color=range_color,
        color_continuous_midpoint=midpoint,
        facet_col="Sector",
        facet_col_wrap=3,
        title=f"Subsector Performance — {plot_col}",
        orientation="h",
        category_orders={"Subsector": y_order},
    )

    fig_sub.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title=cbar_title),
        margin=dict(l=100, r=20, t=60, b=40),
        bargap=0.45,
        bargroupgap=0.25,
    )

    fig_sub.update_yaxes(automargin=True)
    fig_sub.update_xaxes(title=f"{plot_col} (%)")

    st.plotly_chart(fig_sub, use_container_width=True)

    # ---- Top / Weakest callouts within selected sectors ----
    st.markdown("### 💡 Subsector Recommendations")
    top_n_sub = st.slider("Top Subsectors to highlight", 1, 10, min(5, len(df_subperf)), key="tab2_topn")
    top_sub = df_subperf.nlargest(top_n_sub, plot_col)[["Sector", "Subsector"]]
    worst_sub = df_subperf.nsmallest(min(3, len(df_subperf)), plot_col)[["Sector", "Subsector"]]
    st.success("Top performing subsectors: " + ", ".join(f"{r.Sector} — {r.Subsector}" for _, r in top_sub.iterrows()))
    st.error("Weakest subsectors: " + ", ".join(f"{r.Sector} — {r.Subsector}" for _, r in worst_sub.iterrows()))

    # ---- Optional: detail chart for a chosen subsector ----
    st.markdown("### 📈 Subsector Detail Chart")
    opt_sector = st.selectbox("Sector:", sorted(set(df_subperf["Sector"])), key="tab2_det_sector")
    opts = [k[1] for k in sub_index_store.keys() if k[0] == opt_sector]
    opt_sub = st.selectbox("Subsector:", sorted(opts), key="tab2_det_sub")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        period_choice = st.radio("Period", ["3M", "6M", "1Y", "2Y"], index=2, horizontal=True, key="tab2_period")
    with c2:
        show_ma = st.checkbox("Show 50/100-day MAs", value=True, key="tab2_ma")
    with c3:
        overlay_spy = st.checkbox("Overlay SPY (rebased=100)", value=True, key="tab2_spy")

    days_map = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504}
    days = days_map[period_choice]

    idx_series = sub_index_store.get((opt_sector, opt_sub), pd.Series(dtype=float)).dropna()
    if idx_series.empty:
        st.info("No index series available for this subsector.")
    else:
        s = idx_series.iloc[-days:] if len(idx_series) > days else idx_series
        fig_det = go.Figure()
        fig_det.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{opt_sector} — {opt_sub} (EW idx)", mode="lines"))

        if show_ma:
            if len(s) >= 50:
                ma50 = s.rolling(50).mean()
                fig_det.add_trace(
                    go.Scatter(x=ma50.index, y=ma50.values, name="50 DMA", mode="lines", line=dict(dash="dash"))
                )
            if len(s) >= 100:
                ma100 = s.rolling(100).mean()
                fig_det.add_trace(
                    go.Scatter(x=ma100.index, y=ma100.values, name="100 DMA", mode="lines", line=dict(dash="dot"))
                )

        if overlay_spy and not spy_series.empty:
            sp = spy_series.reindex(s.index).dropna()
            if not sp.empty:
                s_norm = s / s.iloc[0] * 100
                sp_norm = sp / sp.iloc[0] * 100
                # replace main line with rebased for consistent scale
                fig_det.data[0].y = s_norm.reindex(fig_det.data[0].x).values
                fig_det.data[0].name = f"{opt_sector} — {opt_sub} (rebased=100)"
                fig_det.add_trace(go.Scatter(x=sp_norm.index, y=sp_norm.values, name="SPY (rebased=100)", mode="lines"))
                y_title = "Rebased to 100"
            else:
                y_title = "Index Level"
        else:
            y_title = "Index Level"

        fig_det.update_layout(
            title=f"{opt_sector} — {opt_sub} ({period_choice})",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title=y_title,
        )
        st.plotly_chart(fig_det, use_container_width=True)

        # ---- Full table (formatted) ----
    st.markdown("### 📄 Subsector Metrics Table")
    st.dataframe(
        df_subperf.style.format(
            {
                "1W": "{:.2%}",
                "1M": "{:.2%}",
                "3M": "{:.2%}",
                "6M": "{:.2%}",
                "YTD": "{:.2%}",
                "1Y": "{:.2%}",
                "RS 3M vs SPY": "{:.2%}",
                "Vol (20D→Ann)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )


# -----------------------------------------------------
# TAB 3 — Stock Signals
# -----------------------------------------------------
with tab3:
    st.header("📈 Buy/Sell Signals by Sector Selection")

    selected_sectors = st.session_state.get("selected_sectors", [])
    if not selected_sectors:
        st.warning("Please select sectors in Tab 1 first.")
    else:
        selected_tickers = [SECTOR_ETFS[s] for s in selected_sectors if s in SECTOR_ETFS]
        stocks = []
        for t in selected_tickers:
            for _, lst in SECTOR_STRUCTURE.get(t, {}).items():
                stocks.extend(lst)
        all_stocks = sorted(set(stocks))

        st.info(f"Fetching data for {len(all_stocks)} stocks across {len(selected_sectors)} selected sectors...")
        stock_close = download_close(all_stocks, period="1y", interval="1d")
        if stock_close.empty:
            st.warning("No stock price data returned. Possibly rate-limited; try changing sectors/period.")
        else:

            def moving_average(series, window):
                return series.rolling(window).mean()

            stock_metrics = []
            for tk in stock_close.columns:
                s = stock_close[tk].dropna()
                if len(s) < 100:
                    continue

                vol = s.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) if len(s) >= 21 else np.nan

                # YTD robust
                cur_year = s.index[-1].year
                s_y = s[s.index.year == cur_year]
                ytd = (s.iloc[-1] / s_y.iloc[0] - 1) if not s_y.empty else np.nan

                ma100 = moving_average(s, 100)
                ma100_last = ma100.iloc[-1] if len(ma100) else np.nan
                above = (s.iloc[-1] > ma100_last) if pd.notna(ma100_last) else False

                m1 = pct_return(s, 21)
                m3 = pct_return(s, 63)
                y1 = pct_return(s, 252)  # ---- NEW: 1Y return ----

                if above and (pd.notna(m3) and m3 > 0.05) and (pd.notna(m1) and m1 > 0):
                    signal = "BUY ✅"
                elif (not above) and (pd.notna(m3) and m3 < -0.05):
                    signal = "SELL ❌"
                else:
                    signal = "HOLD ➖"

                stock_metrics.append(
                    {"Ticker": tk, "1M": m1, "3M": m3, "YTD": ytd, "1Y": y1, "Vol": vol, "Signal": signal}
                )

            stock_df = pd.DataFrame(stock_metrics)
            if stock_df.empty:
                st.info("No stocks met the minimum data criteria.")
            else:
                # Sort: BUY first, then HOLD, then SELL; within each, highest 3M first
                sig_rank = {"BUY ✅": 0, "HOLD ➖": 1, "SELL ❌": 2}
                stock_df["SignalScore"] = stock_df["Signal"].map(sig_rank)
                stock_df = stock_df.sort_values(
                    by=["SignalScore", "3M"], ascending=[True, False], na_position="last"
                ).drop(columns=["SignalScore"])

                # ---- Show the DF with 1Y included ----
                st.markdown("### 📄 Stock Signals Table")
                st.dataframe(
                    stock_df.style.format(
                        {"1M": "{:.2%}", "3M": "{:.2%}", "YTD": "{:.2%}", "1Y": "{:.2%}", "Vol": "{:.2f}"}
                    ),
                    use_container_width=True,
                )

                st.download_button(
                    "Download CSV", stock_df.to_csv(index=False).encode(), "stock_signals.csv", "text/csv"
                )

# -----------------------------------------------------
# TAB 4 — Charts (Sector → Subsector → Stock)
# -----------------------------------------------------
with tab4:
    st.header("📉 Interactive Charts — Sector → Subsector → Stock")

    # --- Optional: hierarchy overview treemap with human-readable sector names ---
    try:
        hierarchy = []
        for etf, subsectors in SECTOR_STRUCTURE.items():
            for sub, stocks_in_sub in subsectors.items():
                for stock in stocks_in_sub:
                    hierarchy.append({"Sector": ETF_TO_NAME.get(etf, etf), "Subsector": sub, "Stock": stock})
        if hierarchy:
            df_hierarchy = pd.DataFrame(hierarchy)
            fig_tree = px.treemap(
                df_hierarchy, path=["Sector", "Subsector", "Stock"], title="Sector → Subsector → Stock Overview"
            )
            fig_tree.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)
    except Exception as e:
        st.info(f"(Treemap skipped: {e})")

    # --- Sector & Subsector selections ---
    if df.empty:
        st.warning("No sector metrics available. Check price data upstream.")
        st.stop()

    chosen_sector = st.selectbox("Select Sector", df["Sector"].unique(), key="tab4_sector")
    etf_ticker = SECTOR_ETFS[chosen_sector]
    subsectors = SECTOR_STRUCTURE.get(etf_ticker, {})

    if not subsectors:
        st.warning(f"No subsector mapping found for {chosen_sector} ({etf_ticker}).")
        st.stop()

    selected_subsector = st.selectbox("Select Subsector", list(subsectors.keys()), key="tab4_subsector")
    stock_list = subsectors[selected_subsector]
    if not stock_list:
        st.warning(f"No stocks listed for subsector: {selected_subsector}")
        st.stop()

    selected_stock = st.selectbox("Select Stock", stock_list, key="tab4_stock")

    # --- 1) Plot Sector ETF (context) ---
    sector_price = prices.get(etf_ticker) if (prices is not None and not prices.empty) else None
    if sector_price is not None and not sector_price.empty:
        ma50_s = sector_price.rolling(50).mean() if len(sector_price) >= 50 else None
        ma100_s = sector_price.rolling(100).mean() if len(sector_price) >= 100 else None

        fig_sector = go.Figure()
        fig_sector.add_trace(
            go.Scatter(x=sector_price.index, y=sector_price.values, name=f"{etf_ticker} Price", mode="lines")
        )
        if ma50_s is not None and ma50_s.notna().any():
            fig_sector.add_trace(
                go.Scatter(x=ma50_s.index, y=ma50_s.values, name="50 DMA", mode="lines", line=dict(dash="dash"))
            )
        if ma100_s is not None and ma100_s.notna().any():
            fig_sector.add_trace(
                go.Scatter(x=ma100_s.index, y=ma100_s.values, name="100 DMA", mode="lines", line=dict(dash="dot"))
            )
        fig_sector.update_layout(
            title=f"{chosen_sector} — {etf_ticker}", template="plotly_white", margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_sector, use_container_width=True)
    else:
        st.warning(f"No price data for sector ETF {etf_ticker}.")

    # --- 2) Plot Selected Stock (robust fetch) ---
    try:
        data_raw = yf.download(selected_stock, period="1y", interval="1d", auto_adjust=True, progress=False)

        if data_raw is None or data_raw.empty:
            st.warning(f"No price data retrieved for {selected_stock}. Try another stock/period.")
            st.stop()

        # Normalize to a clean close-price Series (handles Series/DF/MultiIndex)
        if isinstance(data_raw, pd.DataFrame):
            if isinstance(data_raw.columns, pd.MultiIndex):
                if "Close" in data_raw.columns.get_level_values(0):
                    stock_series = data_raw["Close"].squeeze().dropna()
                else:
                    first_level = data_raw.columns.levels[0][0]
                    stock_series = (
                        data_raw.xs(first_level, axis=1, level=0).select_dtypes(include=[np.number]).iloc[:, 0].dropna()
                    )
            else:
                if "Close" in data_raw.columns:
                    stock_series = data_raw["Close"].dropna()
                else:
                    stock_series = data_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
        else:
            stock_series = data_raw.dropna()

        if stock_series is None or stock_series.empty:
            st.warning(f"Could not parse a valid price series for {selected_stock}.")
            st.stop()

        # Compute MAs if enough data; otherwise show price only
        ma50 = stock_series.rolling(50).mean() if len(stock_series) >= 50 else None
        ma100 = stock_series.rolling(100).mean() if len(stock_series) >= 100 else None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=stock_series.index, y=stock_series.values, name=f"{selected_stock} Price", mode="lines")
        )
        if ma50 is not None and ma50.notna().any():
            fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name="50 DMA", mode="lines", line=dict(dash="dash")))
        else:
            st.info("Not enough data for 50DMA; showing price only.")

        if ma100 is not None and ma100.notna().any():
            fig.add_trace(
                go.Scatter(x=ma100.index, y=ma100.values, name="100 DMA", mode="lines", line=dict(dash="dot"))
            )
        else:
            st.info("Not enough data for 100DMA; showing price only.")

        fig.update_layout(
            title=f"{selected_stock} — {selected_subsector}",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching/plotting {selected_stock}: {e}")

st.markdown("---")
st.caption("Data source: Yahoo Finance | Updated hourly | © 2025 TradeFit Scan")
