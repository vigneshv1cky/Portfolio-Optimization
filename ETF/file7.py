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

PLOT_CONFIG = {
    "displayModeBar": False,  # hide toolbar
    "scrollZoom": False,  # disable scroll wheel zoom
    "doubleClick": "reset",  # double-click to reset axes
    "responsive": True,  # keep charts responsive
}

# ---- Lightweight UI polish ----
st.markdown(
    """
<style>
/* wider page */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1400px; }
/* section headers */
h2, h3, h4 { margin-top: 0.25rem !important; }
/* subtle cards */
.card { border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 16px 18px; margin: 8px 0 16px; background: var(--background-color); }
.card h4 { margin: 0 0 8px; }
.small { color: var(--text-color-secondary); font-size: 0.9rem; }
hr { margin: 0.8rem 0 0.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

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
# Sector → Subsector → Stock Hierarchy (curated examples)
# =====================================================

SECTOR_STRUCTURE = {
    # ===============================
    # INFORMATION TECHNOLOGY (XLK)
    # ===============================
    "XLK": {
        "Semiconductors": ["NVDA", "AVGO", "TSM", "AMD", "INTC", "QCOM", "TXN", "MU", "ADI", "NXPI"],
        "Software—Infrastructure": ["MSFT", "ORCL", "PLTR", "DDOG", "MDB", "NET", "SNOW", "ZS", "PANW", "CRWD"],
        "Software—Application": ["ADBE", "CRM", "NOW", "INTU", "SHOP", "TEAM", "UBER", "WDAY", "SQ", "DOCU"],
        "Consumer Electronics": [
            "AAPL",
            "HPQ",
            "DELL",
            "LOGI",
            "GRMN",
            "ROKU",
            "SONO",
            "UEIC",
            "GPRO",
            "PLUG",
        ],  # last is a device/energy play but liquid
        "Information Technology Services": ["ACN", "IBM", "FI", "FIS", "DXC", "EPAM", "GIB", "CDW", "IT", "LDOS"],
        "Computer Hardware": ["DELL", "HPQ", "HPE", "SMCI", "NTAP", "STX", "WDC", "ANET", "CSCO", "NTNX"],
        "Communication Equipment": ["CSCO", "ANET", "JNPR", "CIEN", "ERIC", "NOK", "IDCC", "UI", "FFIV", "VSAT"],
        "Semiconductor Equipment & Materials": [
            "ASML",
            "AMAT",
            "LRCX",
            "KLAC",
            "TER",
            "ACLS",
            "ONTO",
            "AEIS",
            "COHU",
            "UCTT",
        ],
        "Electronic Components": ["APH", "GLW", "TEL", "LFUS", "MXL", "TTMI", "MTSI", "NVT", "VSH", "SANM"],
        "Solar": [
            "ENPH",
            "SEDG",
            "FSLR",
            "RUN",
            "SHLS",
            "FTC.I",
        ],  # FTC.I=First Solar 2029 bond proxy? If you prefer pure equities, drop it.
        "Electronics & Computer Distribution": ["ARW", "AVT", "SNX", "NSIT", "CNXN", "SCSC", "WCC", "PLUS"],
        "Scientific & Technical Instruments": [
            "KEYS",
            "FTV",
            "ZBRA",
            "A",
            "BRKR",
            "MTD",
            "WAT",
            "COHR",
            "TRMB",
            "RVTY",
        ],
    },
    # =========================================
    # COMMUNICATION SERVICES (XLC)
    # =========================================
    "XLC": {
        "Internet Content & Information": ["GOOG", "META", "SPOT", "PINS", "RDDT", "IAC", "BIDU", "SE", "BILI", "YELP"],
        "Entertainment": ["NFLX", "DIS", "WBD", "LYV", "CMCSA", "ROKU", "EA", "TTWO", "PARA", "CHTR"],
        "Broadcasting": ["NXST", "TGNA", "GTN", "SSP", "SIRI", "IHRT", "SBGI", "FOXA", "CMCSA", "DIS"],
        "Publishing": ["NWSA", "RELX", "NYT", "SCHL", "WLY", "LEE", "GCI", "IDT", "TRIP", "Z"],
        "Advertising Agencies": ["OMC", "IPG", "WPP", "PUBGY", "TTD", "MGNI", "PUBM", "STGW", "CTRA", "HHS"],
        "Telecom Services": ["TMUS", "VZ", "T", "BCE", "LUMN", "ATUS", "USM", "CHTR", "CABO", "WOW"],
        "Wireless Telecommunications Services": ["TMUS", "VZ", "T", "USM", "AMX", "VOD", "TU", "SJR", "TDS", "UONEK"],
        "Electronic Gaming & Multimedia": ["TTWO", "EA", "U", "RBLX", "SONY", "PLTK", "MAT", "HAS", "CRSR", "NTDOY"],
    },
    # ===============================
    # ENERGY (XLE)
    # ===============================
    "XLE": {
        "Oil & Gas Integrated": ["XOM", "CVX", "SHEL", "TTE", "BP", "EQNR", "PBR", "EC", "IMO", "OXY"],
        "Oil & Gas E&P": [
            "COP",
            "EOG",
            "FANG",
            "MRO",
            "APA",
            "DVN",
            "HES",
            "AR",
            "PR",
            "PXD",
        ],  # PXD trades until full close; drop if needed
        "Oil & Gas Midstream": ["KMI", "ENB", "ET", "TRP", "WMB", "OKE", "EPD", "MPLX", "PAA", "AM"],
        "Oil & Gas Equipment & Services": ["SLB", "HAL", "BKR", "FTI", "VAL", "TDW", "NBR", "OIS", "RES", "NOV"],
        "Oil & Gas Refining & Marketing": ["PSX", "VLO", "MPC", "PBF", "DINO", "DK", "CVI", "SUN"],
        "Thermal Coal": ["BTU", "ARLP", "HCC", "SXC"],  # keep small set with reliable quotes
        "Uranium": ["CCJ", "UEC", "DNN", "NXE", "UUUU", "LEU", "URG"],
    },
    # ===============================
    # FINANCIALS (XLF)
    # ===============================
    "XLF": {
        "Banks—Diversified": ["JPM", "BAC", "WFC", "C", "HSBC", "UBS", "RY", "TD", "HDB", "SMFG"],
        "Banks—Regional": ["PNC", "USB", "FITB", "TFC", "RF", "KEY", "HBAN", "CFG", "ZION", "FHN"],
        "Capital Markets": ["GS", "MS", "SCHW", "IBKR", "RJF", "SF", "LAZ", "EVR", "PJT", "MKTX"],
        "Asset Management": ["BLK", "TROW", "BEN", "IVZ", "APO", "KKR", "BX", "CG", "AB", "OWL"],
        "Financial Data & Stock Exchanges": ["SPGI", "MCO", "CME", "ICE", "NDAQ", "CBOE", "MSCI", "TW", "COIN", "SEIC"],
        "Credit Services": ["V", "MA", "AXP", "COF", "ALLY", "SYF", "AFRM", "NAVI", "ENVA", "FOUR"],
        "Mortgage Finance": ["RITM", "PFSI", "COOP", "UWMC", "RDN", "MTG", "NMIH", "PMT", "LADR", "ABR"],
        "Insurance—Diversified": ["BRK-B", "PRU", "MET", "AIG", "AFL", "CNA", "LNC", "UNM", "GL", "FG"],
        "Insurance—Life": ["MET", "PRU", "SLF", "MFC", "VOYA", "EQH", "BHF"],
        "Insurance—Property & Casualty": ["PGR", "TRV", "ALL", "CB", "CINF", "WRB", "HIG", "ACGL", "SIGI", "MKL"],
        "Specialty Finance": ["ALLY", "SYF", "LC", "UPST", "OMF", "SOFI", "ESNT", "AGO", "AX", "TREE"],
    },
    # =========================
    # HEALTH CARE (XLV)
    # =========================
    "XLV": {
        "Biotechnology": [
            "VRTX",
            "REGN",
            "GILD",
            "MRNA",
            "BIIB",
            "ALNY",
            "AMGN",
            "NBIX",
            "EXEL",
            "SGEN",
        ],  # SGEN trades until close
        "Drug Manufacturers—General": ["LLY", "JNJ", "MRK", "PFE", "BMY", "AZN", "NVO", "SNY", "GSK"],
        "Drug Manufacturers—Specialty & Generic": [
            "ZTS",
            "UTHR",
            "TEVA",
            "VTRS",
            "RDY",
            "JAZZ",
            "HCM",
            "IRMN",
            "KNSA",
            "ACAD",
        ],
        "Healthcare Plans": ["UNH", "ELV", "CI", "HUM", "CVS", "CNC", "MOH", "OSCR", "ALHC"],
        "Medical Care Facilities": ["HCA", "UHS", "THC", "ACHC", "ENSG", "AMED", "ADUS", "SGRY", "SEM", "DVA"],
        "Medical Devices": ["ISRG", "SYK", "MDT", "BSX", "EW", "ZBH", "ABT", "BAX", "TFX", "ALGN"],
        "Medical Instruments & Supplies": ["TMO", "DHR", "A", "MTD", "WAT", "BRKR", "RVTY", "BIO", "IQV", "RGEN"],
        "Diagnostics & Research": ["EXAS", "GH", "NTRA", "QGEN", "DGX", "LH", "MYGN", "NVCR", "FTRE", "NEOG"],
    },
    # =========================
    # INDUSTRIALS (XLI)
    # =========================
    "XLI": {
        "Aerospace & Defense": ["GE", "RTX", "BA", "LMT", "NOC", "GD", "HII", "TDG", "HEI", "TXT"],
        "Airlines": ["DAL", "AAL", "UAL", "LUV", "ALK", "JBLU", "RYAAY", "CPA", "AZUL", "SAVE"],
        "Railroads": ["UNP", "CSX", "NSC", "CNI", "CP", "WAB", "GATX", "TRN", "GBX"],
        "Marine Shipping": ["MATX", "SBLK", "EGLE", "GOGL", "DAC", "CMRE", "TK", "GLNG", "STNG", "INSW"],
        "Integrated Freight & Logistics": ["UPS", "FDX", "XPO", "GXO", "EXPD", "CHRW", "ATSG", "FWRD", "TFII", "ODFL"],
        "Trucking": ["ODFL", "SAIA", "KNX", "WERN", "SNDR", "HTLD", "ARCB", "LSTR", "MRTN", "CVLG"],
        "Industrial Distribution": ["GWW", "FAST", "MSM", "WCC", "MRC", "POOL", "SITE", "BECN", "DXPE", "AIT"],
        "Specialty Industrial Machinery": ["ROK", "ROP", "EMR", "AME", "IEX", "ITW", "DOV", "IR", "XYL", "CARR"],
        "Farm & Heavy Construction Machinery": [
            "CAT",
            "DE",
            "CNHI",
            "AGCO",
            "OSK",
            "TEX",
            "MTW",
            "PCAR",
            "TTM",
            "NAVB",
        ],
        "Building Products & Equipment": ["OC", "MAS", "TREX", "AZEK", "JHX", "BLD", "AOS", "ALLE", "DOOR", "MHK"],
        "Electrical Equipment & Parts": ["ETN", "PH", "ABB", "HUBB", "LECO", "NVT", "ALSN", "ENS", "WCC"],
        "Engineering & Construction": ["J", "ACM", "KBR", "FLR", "TTEK", "PWR", "EME", "DY", "GVA", "MTZ"],
        "Waste Management": ["WM", "RSG", "WCN", "GFL", "CWST", "CLH", "HCCI", "SRCL", "ECVT"],
        "Conglomerates": ["HON", "MMM", "GE", "ITW", "DHR", "ROP", "SWK", "DOV", "EMR", "CSL"],
    },
    # =========================
    # MATERIALS (XLB)
    # =========================
    "XLB": {
        "Chemicals": ["DOW", "DD", "LYB", "WLK", "EMN", "APD", "LIN", "ECL", "CF", "CTVA"],
        "Specialty Chemicals": ["SHW", "PPG", "ECL", "ALB", "CE", "IFF", "FUL", "SXT", "AXTA", "AVNT"],
        "Agricultural Inputs": ["NTR", "MOS", "CF", "FMC", "IPI", "CTVA", "SMG", "BG", "ADM", "AGFY"],
        "Aluminum": ["AA", "CENX", "KALU"],
        "Steel": ["NUE", "STLD", "CLF", "RS", "CMC", "HLMN", "MT", "TX", "GGB"],
        "Building Materials": ["MLM", "VMC", "EXP", "SUM", "USLM", "BECN", "JHX", "AWI", "APG"],
        "Lumber & Wood Production": ["WY", "PCH", "BCC", "LPX", "UFPI"],
        "Paper & Paper Products": ["IP", "PKG", "WRK", "MATV", "SUZ", "MERC"],
        "Packaging & Containers": ["BALL", "CCK", "BERY", "SEE", "GEF", "AMCR", "OI", "SLGN", "ATR"],
        "Gold": ["NEM", "GOLD", "AEM", "FNV", "WPM", "KGC", "AU", "SSRM", "AGI", "BTG"],
        "Silver": ["PAAS", "HL", "CDE", "AG", "MAG", "SVM", "SILV", "EXK", "FSM"],
        "Other Industrial Metals & Mining": ["FCX", "SCCO", "TECK", "BHP", "RIO", "VALE", "MP", "S32", "HBM"],
    },
    # =========================
    # REAL ESTATE (XLRE)
    # =========================
    "XLRE": {
        "REIT—Industrial": ["PLD", "REXR", "TRNO", "EGP", "STAG", "LXP", "PLYM"],
        "REIT—Office": ["BXP", "VNO", "KRC", "HIW", "CUZ", "PDM", "SLG", "DEI", "ESRT"],
        "REIT—Retail": ["SPG", "KIM", "FRT", "REG", "BRX", "ROIC", "SKT", "MAC"],
        "REIT—Residential": ["AVB", "EQR", "ESS", "UDR", "CPT", "MAA", "INVH", "AMH"],
        "REIT—Specialty": ["AMT", "CCI", "SBAC", "DLR", "EQIX", "IRM", "IIPR", "WY", "GLPI", "UNIT"],
        "REIT—Mortgage": ["NLY", "AGNC", "STWD", "BXMT", "RITM", "MFA", "TWO", "CIM", "PMT", "DX"],
        "Real Estate Services": ["CBRE", "JLL", "CIGI", "ZG", "COMP", "EXPI", "OPAD", "OPEN"],
        "Real Estate—Development": ["HHH", "JOE", "FPH", "ALEX", "STRS", "FOR"],
        "Real Estate—Diversified": ["WPC", "O", "NNN", "ADC", "ARE", "VICI", "KIM", "REG", "FRT", "SPG"],
    },
    # =========================
    # CONSUMER STAPLES (XLP)
    # =========================
    "XLP": {
        "Household & Personal Products": ["PG", "CL", "KMB", "CHD", "CLX", "EL", "UL", "EPC", "NWL"],
        "Beverages—Non-Alcoholic": ["KO", "PEP", "KDP", "MNST", "CELH", "COKE", "FIZZ"],
        "Beverages—Wineries & Distilleries": ["STZ", "BF-B", "DEO", "MGPI", "CCU", "TAP", "SAM"],
        "Tobacco": ["PM", "MO", "BTI", "VGR", "UVV", "RLX"],
        "Food Distribution": ["SYY", "USFD", "PFGC", "UNFI", "CHEF", "SPTN"],
        "Packaged Foods & Meats": ["GIS", "K", "KHC", "MDLZ", "HSY", "CPB", "CAG", "TSN", "HRL", "POST"],
        "Grocery Stores": ["KR", "ACI", "SFM", "IMKTA", "WMK", "GO", "BJ", "TGT"],
        "Discount Stores": ["WMT", "COST", "DG", "DLTR", "FIVE", "OLLI", "BURL", "TJX", "ROST"],
    },
    # =========================
    # CONSUMER DISCRETIONARY (XLY)
    # =========================
    "XLY": {
        "Auto Manufacturers": ["TSLA", "TM", "HMC", "GM", "F", "RIVN", "LCID", "NIO", "STLA"],
        "Auto & Truck Dealerships": ["KMX", "LAD", "AN", "PAG", "SAH", "ABG", "GPI", "CVNA", "ACVA", "VRM"],
        "Auto Parts": ["GPC", "AZO", "ORLY", "AAP", "LKQ", "MGA", "BWA", "AXL", "DORM", "VC"],
        "Specialty Retail": ["ULTA", "BBY", "RH", "WSM", "TSCO", "ASO", "BBWI", "DDS", "BKE", "DKS"],
        "Apparel Retail": ["LULU", "ANF", "URBN", "GPS", "AEO", "TPR", "CPRI", "TJX", "ROST", "BURL"],
        "Footwear & Accessories": ["NKE", "DECK", "CROX", "SKX", "UAA", "VFC", "ONON", "SHOO", "WWW", "BOOT"],
        "Internet Retail": ["AMZN", "MELI", "ETSY", "PDD", "JD", "BABA", "EBAY", "W", "RVLV", "CHWY"],
        "Restaurants": ["MCD", "SBUX", "CMG", "YUM", "QSR", "DPZ", "WEN", "DRI", "TXRH", "SHAK"],
        "Home Improvement Retail": ["HD", "LOW", "FND", "TTSH"],
        "Leisure": ["CCL", "RCL", "NCLH", "PLNT", "EDR", "LYV", "PTON", "MCW"],
        "Resorts & Casinos": ["LVS", "MGM", "WYNN", "CZR", "MLCO", "PENN", "BYD", "CHDN", "RRR"],
        "Lodging": ["MAR", "HLT", "H", "IHG", "CHH", "WH", "HST", "PK", "DRH"],
        "Recreational Vehicles": ["THO", "WGO", "CWH", "PATK", "LCII", "PII", "MBUU", "MCFT", "BC"],
        "Furnishings, Fixtures & Appliances": ["WHR", "LEG", "SNBR", "TILE", "MHK", "SNA", "SWK", "NWL", "AOS", "KTB"],
        "Travel Services": ["BKNG", "EXPE", "TRIP", "ABNB", "TCOM", "SABR", "DESP", "CWT", "HGV", "MMYT"],
    },
    # =========================
    # UTILITIES (XLU)
    # =========================
    "XLU": {
        "Utilities—Regulated Electric": ["NEE", "DUK", "SO", "AEP", "EXC", "D", "PCG", "EIX", "PEG", "PPL"],
        "Utilities—Regulated Gas": ["ATO", "OGS", "NWN", "NI", "NJR", "SWX", "UGI", "SR", "SRE", "CPK"],
        "Utilities—Regulated Water": ["AWK", "WTRG", "AWR", "MSEX", "CWCO", "ARTNA", "YORW", "GWRS"],
        "Utilities—Independent Power Producers": ["NRG", "AES", "CEG", "VST", "BEP", "CWEN", "ORA"],
        "Utilities—Renewable": ["BEP", "CWEN", "ORA", "AQN", "NEE", "RUN", "SHLS"],
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


def equal_weight_index(close_df: pd.DataFrame) -> pd.Series:
    """
    Build an equal-weighted index from a Close-price DataFrame (columns=tickers).
    Handles missing data by averaging daily % changes across available tickers.
    Returns a series rebased to 100.
    """
    if close_df is None or close_df.empty:
        return pd.Series(dtype=float)
    ew_rets = close_df.sort_index().pct_change(fill_method=None).mean(axis=1, skipna=True)
    idx = (1 + ew_rets.fillna(0)).cumprod()
    if not idx.empty:
        idx = idx / idx.iloc[0] * 100.0
    return idx


@st.cache_data(ttl=3600)
def get_subsector_prices(selected_sectors, period="2y", interval="1d"):
    """
    Download Close prices for all stocks in the subsectors belonging to the selected sectors.
    Returns a dict: {(SectorName, SubsectorName): Close-DF for its tickers}, and a union Close DF.
    """
    wanted = []
    sec_sub_map = {}
    for sec in selected_sectors:
        etf = SECTOR_ETFS.get(sec)
        if etf and etf in SECTOR_STRUCTURE:
            for sub, tickers in SECTOR_STRUCTURE[etf].items():
                sec_sub_map[(sec, sub)] = tickers
                wanted.extend(tickers)
    wanted = sorted(set(wanted))

    all_close = download_close(wanted, period=period, interval=interval)
    per_sub_dfs = {}
    for (sec, sub), tickers in sec_sub_map.items():
        cols = [t for t in tickers if t in all_close.columns]
        per_sub_dfs[(sec, sub)] = all_close[cols] if cols else pd.DataFrame()
    return per_sub_dfs, all_close


# =====================================================
# Load Data
# =====================================================
tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
prices = download_close(tickers, period="2y", interval="1d")

# =====================================================
# Compute Metrics (independent of UI)
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

    rets = s.sort_index().pct_change(fill_method=None).dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252) if len(rets) >= 20 else np.nan

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
# TOP TOOLBAR (replaces sidebar) + ROW LAYOUT
# =====================================================
st.markdown("## 📊 Sector Performance Overview")

with st.container():
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        ranking_choice = st.radio(
            "Performance Period", ["1W", "1M", "3M", "6M", "YTD", "1Y"], index=2, horizontal=True, key="p_top_period"
        )
    with c2:
        st.markdown("")
        st.markdown("")
        st.caption("Source: Yahoo Finance (auto-adjusted closes)")
    with c3:
        st.markdown("")
        st.markdown("")
        st.caption("Tip: Picks flow into subsectors and stocks below.")

if df.empty:
    st.error("No sector metrics available. Check data source / try again later.")
    st.stop()
else:
    # --- Prepare data once ---
    df_plot = df.copy()
    df_plot[ranking_choice] = df_plot[ranking_choice] * 100
    top_row = df.nlargest(1, ranking_choice).iloc[0]
    worst_row = df.nsmallest(1, ranking_choice).iloc[0]

    # === ROW: Sector bar (left) + Picks/controls (right) ===
    top_row_cols = st.columns([3, 2], gap="large")

    with top_row_cols[0]:
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
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    with top_row_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Sector Picks (1 each)")
        st.success(f"**BUY:** {top_row['Sector']}  — {top_row[ranking_choice]:.2%}")
        st.error(f"**SELL:** {worst_row['Sector']} — {worst_row[ranking_choice]:.2%}")

        all_sectors = df["Sector"].tolist()
        pick_mode = st.radio(
            "Carry one sector forward:",
            ["Top Buy", "Top Sell", "Manual"],
            index=0,
            horizontal=True,
            key="p_sector_pick_mode",
        )
        if pick_mode == "Top Buy":
            selected_sector = top_row["Sector"]
        elif pick_mode == "Top Sell":
            selected_sector = worst_row["Sector"]
        else:
            default_idx = all_sectors.index(top_row["Sector"])
            selected_sector = st.selectbox("Sector:", options=all_sectors, index=default_idx, key="p_sector_manual")
        st.markdown("</div>", unsafe_allow_html=True)

    # Save carried sector
    st.session_state["selected_sector"] = selected_sector
    st.session_state["selected_sectors"] = [selected_sector]

    # === Sector context line chart (full width under row) ===
    c1, c2 = st.columns([3, 1])
    with c1:
        chart_period = st.radio("Period", ["3M", "6M", "1Y"], index=2, horizontal=True, key="p_sector_period")
    with c2:
        overlay_spy = st.checkbox("Overlay SPY (benchmark)", value=False, key="p_sector_overlay_spy")

    period_days = {"3M": 63, "6M": 126, "1Y": 252}
    days = period_days[chart_period]

    etf_ticker = SECTOR_ETFS.get(selected_sector)
    sector_series = prices.get(etf_ticker) if etf_ticker in prices.columns else None
    spy_series = prices.get(BENCHMARK) if BENCHMARK in prices.columns else None

    if sector_series is None or sector_series.empty:
        st.warning(f"No price data for sector ETF {etf_ticker}.")
    else:
        s = sector_series.dropna()
        s = s.iloc[-days:] if len(s) > days else s

        fig_line = go.Figure()
        if overlay_spy and spy_series is not None and not spy_series.empty:
            sp = spy_series.dropna()
            sp = sp.iloc[-days:] if len(sp) > days else sp
            s_plot = s / float(s.iloc[0]) * 100.0
            sp_plot = sp / float(sp.iloc[0]) * 100.0
            fig_line.add_trace(
                go.Scatter(x=s_plot.index, y=s_plot.values, name=f"{etf_ticker} (rebased=100)", mode="lines")
            )
            fig_line.add_trace(go.Scatter(x=sp_plot.index, y=sp_plot.values, name="SPY (rebased=100)", mode="lines"))
            y_title = "Rebased to 100"
        else:
            fig_line.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{etf_ticker} Price", mode="lines"))
            y_title = "Price"

        fig_line.update_layout(
            title=f"{selected_sector} — {etf_ticker} ({chart_period})",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title=y_title,
        )
        st.plotly_chart(fig_line, use_container_width=True, config=PLOT_CONFIG)

    # Big table → expander
    with st.expander("📄 Sector Metrics Table", expanded=False):
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

st.divider()
st.markdown("## 🏗️ Subsector Performance — Carried Sector")

carried_sector = st.session_state.get("selected_sector")
if not carried_sector:
    st.warning("No sector carried. Please choose a sector above.")
else:
    st.info(f"**Carried Sector:** {carried_sector}")

    # Pull prices for subsectors in the carried sector only
    per_sub_dfs, all_sub_close = get_subsector_prices([carried_sector], period="2y", interval="1d")

    if all_sub_close is None or all_sub_close.empty:
        st.warning("No stock data available (rate limit or symbols). Try different period.")
    else:
        spy_series = prices[BENCHMARK].dropna() if BENCHMARK in prices.columns else pd.Series(dtype=float)

        # Compute per-subsector metrics
        sub_metrics, sub_index_store = [], {}
        for (sec, sub), df_close in per_sub_dfs.items():
            if df_close is None or df_close.empty:
                continue
            idx = equal_weight_index(df_close).dropna()
            if idx.empty:
                continue

            idx_rets = idx.sort_index().pct_change(fill_method=None).dropna()
            vol = idx_rets.rolling(20).std().iloc[-1] * np.sqrt(252) if len(idx_rets) >= 20 else np.nan

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
            st.info("No subsector performance could be computed.")
        else:
            # Build the bar first
            # ---- Build a single, filtered ranking view for THIS period
            rank_df = df_subperf[["Subsector", plot_col := ranking_choice]].dropna().astype({plot_col: float})

            # % only for plotting
            plot_df = rank_df.copy()
            plot_df[plot_col] = plot_df[plot_col] * 100

            # ordering & color logic
            y_order = plot_df.sort_values(plot_col, ascending=False)["Subsector"].tolist()
            vmin, vmax = float(plot_df[plot_col].min()), float(plot_df[plot_col].max())
            max_abs = max(abs(vmin), abs(vmax))
            mixed, pos_only = (vmin < 0 < vmax), (vmin >= 0)

            color_col, color_scale, range_color, midpoint = plot_col, None, None, None
            cbar_title = f"{plot_col} (%)"
            if mixed:
                color_scale = [(0, "red"), (0.5, "yellow"), (1, "green")]
                range_color, midpoint = [-max_abs, max_abs], 0
            elif pos_only:
                color_scale, range_color = "Greens", [0, vmax]
            else:
                color_col = "_loss_mag"
                plot_df[color_col] = -plot_df[plot_col]
                color_scale, range_color, cbar_title = "Reds", [0, max_abs], "Loss magnitude (%)"

            # IMPORTANT: pass COLUMN NAMES, not Series
            dfp = plot_df.sort_values(plot_col, ascending=False)
            fig_sub = px.bar(
                dfp,
                y="Subsector",
                x=plot_col,
                color=color_col,
                color_continuous_scale=color_scale,
                range_color=range_color,
                color_continuous_midpoint=midpoint,
                title=f"{carried_sector} — Subsector Performance — {plot_col}",
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
            fig_sub.update_xaxes(title=f"{plot_col} (%)", ticksuffix="%")

            # === ROW: Subsector bar (left) + Picks/controls (right) ===
            sub_cols = st.columns([3, 2], gap="large")
            with sub_cols[0]:
                st.plotly_chart(fig_sub, use_container_width=True, config=PLOT_CONFIG)  # ← replace use_container_width

            # Picks computed from the SAME ranking view (rank_df)
            top_row = rank_df.sort_values(plot_col, ascending=False).iloc[0]
            worst_row = rank_df.sort_values(plot_col, ascending=True).iloc[0]
            available_subs = rank_df["Subsector"].tolist()

            with sub_cols[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 💡 Subsector Picks (1 each)")
                st.success(f"**BUY:** {top_row['Subsector']} — {top_row[plot_col]:.2%}")
                st.error(f"**SELL:** {worst_row['Subsector']} — {worst_row[plot_col]:.2%}")
                sub_pick_mode = st.radio(
                    "Carry one subsector forward:",
                    ["Top Buy", "Top Sell", "Manual"],
                    index=0,
                    horizontal=True,
                    key="p_sub_pick_mode",
                )
                if sub_pick_mode == "Top Buy":
                    selected_subsector = top_row["Subsector"]
                elif sub_pick_mode == "Top Sell":
                    selected_subsector = worst_row["Subsector"]
                else:
                    default_idx = available_subs.index(top_row["Subsector"])
                    selected_subsector = st.selectbox(
                        "Subsector:", available_subs, index=default_idx, key="p_sub_manual"
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            # Save carried subsector
            st.session_state["selected_subsector"] = selected_subsector

            # === Subsector detail (full width) ===
            st.markdown("#### 📈 Subsector Detail (carried)")
            idx_series = sub_index_store.get((carried_sector, selected_subsector), pd.Series(dtype=float)).dropna()
            if idx_series.empty:
                st.info("No index series available for this subsector.")
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    period_choice = st.radio("Period", ["3M", "6M", "1Y"], index=1, horizontal=True, key="p_sub_period")
                with c2:
                    overlay_spy = st.checkbox("Overlay SPY (rebased=100)", value=False, key="p_sub_spy")

                days_map = {"3M": 63, "6M": 126, "1Y": 252}
                s = (
                    idx_series.iloc[-days_map[period_choice] :]
                    if len(idx_series) > days_map[period_choice]
                    else idx_series
                )

                fig_det = go.Figure()
                if overlay_spy and not spy_series.empty:
                    sp = spy_series.reindex(s.index).dropna()
                    if not sp.empty:
                        s_plot = s / float(s.iloc[0]) * 100.0
                        sp_plot = sp / float(sp.iloc[0]) * 100.0
                        fig_det.add_trace(
                            go.Scatter(
                                x=s_plot.index,
                                y=s_plot.values,
                                name=f"{carried_sector} — {selected_subsector} (rebased=100)",
                                mode="lines",
                            )
                        )
                        fig_det.add_trace(
                            go.Scatter(x=sp_plot.index, y=sp_plot.values, name="SPY (rebased=100)", mode="lines")
                        )
                        y_title = "Rebased to 100"
                    else:
                        fig_det.add_trace(
                            go.Scatter(
                                x=s.index,
                                y=s.values,
                                name=f"{carried_sector} — {selected_subsector} (EW idx)",
                                mode="lines",
                            )
                        )
                        y_title = "Index Level"
                else:
                    fig_det.add_trace(
                        go.Scatter(
                            x=s.index,
                            y=s.values,
                            name=f"{carried_sector} — {selected_subsector} (EW idx)",
                            mode="lines",
                        )
                    )
                    y_title = "Index Level"

                fig_det.update_layout(
                    title=f"{carried_sector} — {selected_subsector} ({period_choice})",
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                    yaxis_title=y_title,
                )
                st.plotly_chart(fig_det, use_container_width=True, config=PLOT_CONFIG)

            # Big table → expander
            with st.expander("📄 Subsector Metrics Table", expanded=False):
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

st.divider()
st.markdown("## 📈 Buy/Sell Signals — Carried Subsector")

carried_sector = st.session_state.get("selected_sector")
carried_sub = st.session_state.get("selected_subsector")

# We'll compute signals table once here and reuse summary in the next section.
stock_df = pd.DataFrame()

if not carried_sector or not carried_sub:
    st.warning("Please pick a sector above and a subsector in the Subsector section.")
else:
    etf = SECTOR_ETFS[carried_sector]
    subsectors = SECTOR_STRUCTURE.get(etf, {})
    stock_list = subsectors.get(carried_sub, [])
    if not stock_list:
        st.warning(f"No stocks found for {carried_sector} → {carried_sub}.")
    else:
        st.info(f"**Carried:** {carried_sector} → {carried_sub}  |  Constituents: {len(stock_list)}")
        stock_close = download_close(stock_list, period="2y", interval="1d")
        if stock_close.empty:
            st.warning("No stock price data returned. Possibly rate-limited; try changing period.")
        else:
            stock_metrics = []
            for tk in stock_close.columns:
                s = stock_close[tk].dropna()
                if len(s) < 100:
                    continue

                vol = (
                    s.sort_index().pct_change(fill_method=None).rolling(20).std().iloc[-1] * np.sqrt(252)
                    if len(s) >= 21
                    else np.nan
                )

                cur_year = s.index[-1].year
                s_y = s[s.index.year == cur_year]
                ytd = (s.iloc[-1] / s_y.iloc[0] - 1) if not s_y.empty else np.nan

                m1 = pct_return(s, 21)
                m3 = pct_return(s, 63)
                y1 = pct_return(s, 252)

                if (pd.notna(m3) and m3 > 0.05) and (pd.notna(m1) and m1 > 0):
                    signal = "BUY ✅"
                elif pd.notna(m3) and m3 < -0.05:
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
                sig_rank = {"BUY ✅": 0, "HOLD ➖": 1, "SELL ❌": 2}
                stock_df["SignalScore"] = stock_df["Signal"].map(sig_rank)
                stock_df = stock_df.sort_values(
                    by=["SignalScore", "3M"], ascending=[True, False], na_position="last"
                ).drop(columns=["SignalScore"])

                with st.expander("📄 Stock Signals Table", expanded=True):
                    st.dataframe(
                        stock_df.style.format(
                            {"1M": "{:.2%}", "3M": "{:.2%}", "YTD": "{:.2%}", "1Y": "{:.2%}", "Vol": "{:.2f}"}
                        ),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download CSV", stock_df.to_csv(index=False).encode(), "stock_signals.csv", "text/csv"
                    )

st.divider()
st.markdown("## 📉 Stocks")

try:
    # --- Map ranking choice to window ---
    def _days_from_choice(choice: str) -> tuple[int, bool]:
        mapping = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
        if choice == "YTD":
            return (0, True)
        return (mapping.get(choice, 63), False)  # default 3M

    days_window, is_ytd = _days_from_choice(ranking_choice)

    # --- Return over window / YTD ---
    def _window_return(close: pd.Series, days: int, is_ytd: bool) -> float:
        s = close.dropna()
        if s.empty:
            return np.nan
        if is_ytd:
            tz = getattr(s.index, "tz", None)
            start = pd.Timestamp(pd.Timestamp.today().year, 1, 1, tz=tz)
            s = s.loc[s.index >= start]
            if len(s) < 2:
                return np.nan
            return s.iloc[-1] / s.iloc[0] - 1
        if len(s) < 2:
            return np.nan
        s_win = s.iloc[-days:] if len(s) >= 2 else s
        if len(s_win) < 2:
            return np.nan
        return s_win.iloc[-1] / s_win.iloc[0] - 1

    # --- Fetch all stocks in this subsector (2y so 1Y/YTD covered) ---
    carried_sector = st.session_state.get("selected_sector")
    carried_subsector = st.session_state.get("selected_subsector")
    if carried_sector and carried_subsector:
        etf_ticker = SECTOR_ETFS[carried_sector]
        subsectors = SECTOR_STRUCTURE.get(etf_ticker, {})
        stock_list = subsectors.get(carried_subsector, [])
    else:
        stock_list = []

    if stock_list:
        data_multi = yf.download(
            stock_list, period="2y", interval="1d", auto_adjust=True, progress=False, group_by="ticker", threads=True
        )

        rows = []
        for t in stock_list:
            try:
                if isinstance(data_multi, pd.DataFrame) and isinstance(data_multi.columns, pd.MultiIndex):
                    if t in data_multi.columns.get_level_values(0):
                        df_t = data_multi[t]
                        close = (
                            df_t["Close"]
                            if "Close" in df_t.columns
                            else df_t.select_dtypes(include=[np.number]).iloc[:, 0]
                        )
                    else:
                        close = pd.Series(dtype=float)
                else:
                    close = (
                        data_multi["Close"].dropna()
                        if isinstance(data_multi, pd.DataFrame) and "Close" in data_multi
                        else pd.Series(dtype=float)
                    )

                r = _window_return(close, days_window, is_ytd)
                rows.append({"Stock": t, "ReturnPct": (r * 100) if pd.notna(r) else np.nan})
            except Exception:
                rows.append({"Stock": t, "ReturnPct": np.nan})

        df_stockperf = pd.DataFrame(rows).dropna(subset=["ReturnPct"])
        if df_stockperf.empty:
            st.warning("No sufficient data to compute stock returns for this subsector.")
        else:
            plot_col = "ReturnPct"
            plot_df = df_stockperf.copy()

            # --- Ordering & color scale logic (same as above) ---
            y_order = plot_df.sort_values(plot_col, ascending=False)["Stock"].tolist()

            vmin, vmax = float(plot_df[plot_col].min()), float(plot_df[plot_col].max())
            max_abs = max(abs(vmin), abs(vmax))
            mixed, pos_only = (vmin < 0 < vmax), (vmin >= 0)

            color_col, color_scale, range_color, midpoint = plot_col, None, None, None
            cbar_title = f"{ranking_choice} (%)"
            if mixed:
                color_scale = [(0, "red"), (0.5, "yellow"), (1, "green")]
                range_color, midpoint = [-max_abs, max_abs], 0
            elif pos_only:
                color_scale, range_color = "Greens", [0, vmax]
            else:
                color_col = "_loss_mag"
                plot_df[color_col] = -plot_df[plot_col]
                color_scale, range_color, cbar_title = "Reds", [0, max_abs], "Loss magnitude (%)"

            fig_stocks = px.bar(
                plot_df.sort_values(plot_col, ascending=False),
                y="Stock",
                x=plot_df[plot_col],
                color=plot_df[color_col],
                color_continuous_scale=color_scale,
                range_color=range_color,
                color_continuous_midpoint=midpoint,
                title=f"{carried_sector} → {carried_subsector} — Stock Performance — {ranking_choice}",
                orientation="h",
                category_orders={"Stock": y_order},
            )
            fig_stocks.update_layout(
                template="plotly_white",
                coloraxis_colorbar=dict(title=cbar_title),
                margin=dict(l=100, r=20, t=60, b=40),
                bargap=0.45,
                bargroupgap=0.25,
            )
            fig_stocks.update_yaxes(automargin=True)
            fig_stocks.update_xaxes(title=f"{ranking_choice} (%)", ticksuffix="%")

            # === ROW: Stock bar (left) + Context/summary card (right) ===
            row1 = st.columns([3, 2], gap="large")
            with row1[0]:
                st.plotly_chart(fig_stocks, use_container_width=True, config=PLOT_CONFIG)
            with row1[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 🧭 Context")
                st.markdown(
                    f"**Sector:** `{carried_sector}`  \n**ETF:** `{etf_ticker}`  \n**Subsector:** `{carried_subsector}`"
                )
                # quick counts by signal (reuse if available)
                if "stock_df" in locals() and not stock_df.empty and "Signal" in stock_df.columns:
                    counts = stock_df["Signal"].value_counts().to_dict()
                    st.caption("Signal counts")
                    st.write(counts)
                st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error building stocks horizontal barplot: {e}")

# =======================
# Minimal Stock Chart area (kept, but now visually lighter)
# =======================
if df.empty:
    st.warning("No sector metrics available. Check price data upstream.")
else:
    # ---------- Safe defaults (no UI for sector/subsector) ----------
    sector_options = df["Sector"].dropna().unique().tolist()
    if not sector_options:
        st.warning("No sectors available.")
        st.stop()

    carried_sector = st.session_state.get("selected_sector")
    if not carried_sector or carried_sector not in sector_options:
        carried_sector = sector_options[0]
        st.session_state["selected_sector"] = carried_sector

    etf_ticker = SECTOR_ETFS[carried_sector]
    subsectors = SECTOR_STRUCTURE.get(etf_ticker, {})
    subs_options = list(subsectors.keys())

    if not subs_options:
        st.warning(f"No subsector mapping found for {carried_sector} ({etf_ticker}).")
        st.stop()

    carried_subsector = st.session_state.get("selected_subsector")
    if not carried_subsector or carried_subsector not in subs_options:
        carried_subsector = subs_options[0]
        st.session_state["selected_subsector"] = carried_subsector

    stock_list = subsectors.get(carried_subsector, [])
    if not stock_list:
        st.warning(f"No stocks listed for subsector: {carried_subsector}")
        st.stop()

    st.session_state["_prev_sector"] = st.session_state.get("_prev_sector", carried_sector)
    st.session_state["_prev_subsector"] = st.session_state.get("_prev_subsector", carried_subsector)

    st.markdown("#### 🧭 Context: Sector ETF")
    sector_price = prices.get(etf_ticker) if (prices is not None and not prices.empty) else None
    if sector_price is not None and not sector_price.empty:
        fig_sector = go.Figure()
        fig_sector.add_trace(
            go.Scatter(x=sector_price.index, y=sector_price.values, name=f"{etf_ticker} Price", mode="lines")
        )
        fig_sector.update_layout(
            title=f"{carried_sector} — {etf_ticker}", template="plotly_white", margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_sector, use_container_width=True, config=PLOT_CONFIG)
    else:
        st.warning(f"No price data for sector ETF {etf_ticker}.")

    st.markdown("#### 📈 Selected Stock")
    try:
        # keep dropdown, but it's now below the main charts
        if "selected_stock" not in st.session_state or st.session_state["selected_stock"] not in stock_list:
            st.session_state["selected_stock"] = stock_list[0]
        selected_stock = st.selectbox(
            "Select Stock", stock_list, index=stock_list.index(st.session_state["selected_stock"]), key="p_chart_stock"
        )
        st.session_state["selected_stock"] = selected_stock

        data_raw = yf.download(selected_stock, period="1y", interval="1d", auto_adjust=True, progress=False)
        if data_raw is None or data_raw.empty:
            st.warning(f"No price data retrieved for {selected_stock}. Try another stock/period.")
        else:
            if isinstance(data_raw, pd.DataFrame):
                if isinstance(data_raw.columns, pd.MultiIndex):
                    if "Close" in data_raw.columns.get_level_values(0):
                        stock_series = data_raw["Close"].squeeze().dropna()
                    else:
                        first_level = data_raw.columns.levels[0][0]
                        stock_series = (
                            data_raw.xs(first_level, axis=1, level=0)
                            .select_dtypes(include=[np.number])
                            .iloc[:, 0]
                            .dropna()
                        )
                else:
                    stock_series = (
                        data_raw["Close"].dropna()
                        if "Close" in data_raw.columns
                        else data_raw.select_dtypes(include=[np.number]).iloc[:, 0].dropna()
                    )
            else:
                stock_series = data_raw.dropna()

            if stock_series is None or stock_series.empty:
                st.warning(f"Could not parse a valid price series for {selected_stock}.")
            else:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=stock_series.index, y=stock_series.values, name=f"{selected_stock} Price", mode="lines"
                    )
                )
                fig.update_layout(
                    title=f"{selected_stock} — {carried_subsector}",
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    except Exception as e:
        st.error(f"Error fetching/plotting {selected_stock}: {e}")

st.divider()
st.caption("Source: Yahoo Finance | Updated hourly | © 2025 TradeFit Scan")
