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
# Stock universe — flattened to sector level (no subsectors)
# (Curated tickers aggregated from your previous mapping)
# =====================================================
SECTOR_STOCKS = {
    # ===============================
    # INFORMATION TECHNOLOGY (XLK)
    # ===============================
    "Information Technology": sorted(
        set(
            [
                # Semiconductors
                "NVDA",
                "AVGO",
                "TSM",
                "AMD",
                "INTC",
                "QCOM",
                "TXN",
                "MU",
                "ADI",
                "NXPI",
                # Software—Infrastructure
                "MSFT",
                "ORCL",
                "PLTR",
                "DDOG",
                "MDB",
                "NET",
                "SNOW",
                "ZS",
                "PANW",
                "CRWD",
                # Software—Application
                "ADBE",
                "CRM",
                "NOW",
                "INTU",
                "SHOP",
                "TEAM",
                "UBER",
                "WDAY",
                "SQ",
                "DOCU",
                # Consumer Electronics
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
                # IT Services
                "ACN",
                "IBM",
                "FI",
                "FIS",
                "DXC",
                "EPAM",
                "GIB",
                "CDW",
                "IT",
                "LDOS",
                # Computer Hardware
                "HPE",
                "SMCI",
                "NTAP",
                "STX",
                "WDC",
                "ANET",
                "CSCO",
                "NTNX",
                # Communication Equipment
                "JNPR",
                "CIEN",
                "ERIC",
                "NOK",
                "IDCC",
                "UI",
                "FFIV",
                "VSAT",
                # Semi Equip & Materials
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
                # Electronic Components
                "APH",
                "GLW",
                "TEL",
                "LFUS",
                "MXL",
                "TTMI",
                "MTSI",
                "NVT",
                "VSH",
                "SANM",
                # Solar
                "ENPH",
                "SEDG",
                "FSLR",
                "RUN",
                "SHLS",
                # Distribution
                "ARW",
                "AVT",
                "SNX",
                "NSIT",
                "CNXN",
                "SCSC",
                "WCC",
                "PLUS",
                # Scientific & Technical Instruments
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
            ]
        )
    ),
    # =========================================
    # COMMUNICATION SERVICES (XLC)
    # =========================================
    "Communication Services": sorted(
        set(
            [
                # Internet Content & Info
                "GOOG",
                "META",
                "SPOT",
                "PINS",
                "RDDT",
                "IAC",
                "BIDU",
                "SE",
                "BILI",
                "YELP",
                # Entertainment
                "NFLX",
                "DIS",
                "WBD",
                "LYV",
                "CMCSA",
                "ROKU",
                "EA",
                "TTWO",
                "PARA",
                "CHTR",
                # Broadcasting
                "NXST",
                "TGNA",
                "GTN",
                "SSP",
                "SIRI",
                "IHRT",
                "SBGI",
                "FOXA",
                # Publishing
                "NWSA",
                "RELX",
                "NYT",
                "SCHL",
                "WLY",
                "LEE",
                "GCI",
                "IDT",
                "TRIP",
                "Z",
                # Advertising
                "OMC",
                "IPG",
                "WPP",
                "PUBGY",
                "TTD",
                "MGNI",
                "PUBM",
                "STGW",
                # Telecom & Wireless
                "TMUS",
                "VZ",
                "T",
                "BCE",
                "LUMN",
                "ATUS",
                "USM",
                "CABO",
                "WOW",
                "TU",
                "SJR",
                "TDS",
                # Gaming & Multimedia
                "U",
                "RBLX",
                "SONY",
                "PLTK",
                "MAT",
                "HAS",
                "CRSR",
                "NTDOY",
            ]
        )
    ),
    # ===============================
    # ENERGY (XLE)
    # ===============================
    "Energy": sorted(
        set(
            [
                # Integrated
                "XOM",
                "CVX",
                "SHEL",
                "TTE",
                "BP",
                "EQNR",
                "PBR",
                "EC",
                "IMO",
                "OXY",
                # E&P
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
                # Midstream
                "KMI",
                "ENB",
                "ET",
                "TRP",
                "WMB",
                "OKE",
                "EPD",
                "MPLX",
                "PAA",
                "AM",
                # Oilfield Services
                "SLB",
                "HAL",
                "BKR",
                "FTI",
                "VAL",
                "TDW",
                "NBR",
                "OIS",
                "RES",
                "NOV",
                # Refining & Marketing
                "PSX",
                "VLO",
                "MPC",
                "PBF",
                "DINO",
                "DK",
                "CVI",
                "SUN",
                # Thermal Coal & Uranium
                "BTU",
                "ARLP",
                "HCC",
                "SXC",
                "CCJ",
                "UEC",
                "DNN",
                "NXE",
                "UUUU",
                "LEU",
                "URG",
            ]
        )
    ),
    # ===============================
    # FINANCIALS (XLF)
    # ===============================
    "Financials": sorted(
        set(
            [
                # Banks
                "JPM",
                "BAC",
                "WFC",
                "C",
                "HSBC",
                "UBS",
                "RY",
                "TD",
                "HDB",
                "SMFG",
                "PNC",
                "USB",
                "FITB",
                "TFC",
                "RF",
                "KEY",
                "HBAN",
                "CFG",
                "ZION",
                "FHN",
                # Capital Markets & AM
                "GS",
                "MS",
                "SCHW",
                "IBKR",
                "RJF",
                "SF",
                "LAZ",
                "EVR",
                "PJT",
                "MKTX",
                "BLK",
                "TROW",
                "BEN",
                "IVZ",
                "APO",
                "KKR",
                "BX",
                "CG",
                "AB",
                "OWL",
                # Data/Exchanges
                "SPGI",
                "MCO",
                "CME",
                "ICE",
                "NDAQ",
                "CBOE",
                "MSCI",
                "TW",
                "COIN",
                "SEIC",
                # Credit/Payments & Specialty
                "V",
                "MA",
                "AXP",
                "COF",
                "ALLY",
                "SYF",
                "AFRM",
                "NAVI",
                "ENVA",
                "FOUR",
                "LC",
                "UPST",
                "OMF",
                "SOFI",
                "ESNT",
                "AGO",
                "AX",
                "TREE",
                # Insurance
                "BRK-B",
                "PRU",
                "MET",
                "AIG",
                "AFL",
                "CNA",
                "LNC",
                "UNM",
                "GL",
                "FG",
                "SLF",
                "MFC",
                "VOYA",
                "EQH",
                "BHF",
                "PGR",
                "TRV",
                "ALL",
                "CB",
                "CINF",
                "WRB",
                "HIG",
                "ACGL",
                "SIGI",
                "MKL",
                # Mortgage Finance
                "RITM",
                "PFSI",
                "COOP",
                "UWMC",
                "RDN",
                "MTG",
                "NMIH",
                "PMT",
                "LADR",
                "ABR",
            ]
        )
    ),
    # =========================
    # HEALTH CARE (XLV)
    # =========================
    "Health Care": sorted(
        set(
            [
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
                "LLY",
                "JNJ",
                "MRK",
                "PFE",
                "BMY",
                "AZN",
                "NVO",
                "SNY",
                "GSK",
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
                "UNH",
                "ELV",
                "CI",
                "HUM",
                "CVS",
                "CNC",
                "MOH",
                "OSCR",
                "ALHC",
                "HCA",
                "UHS",
                "THC",
                "ACHC",
                "ENSG",
                "AMED",
                "ADUS",
                "SGRY",
                "SEM",
                "DVA",
                "ISRG",
                "SYK",
                "MDT",
                "BSX",
                "EW",
                "ZBH",
                "ABT",
                "BAX",
                "TFX",
                "ALGN",
                "TMO",
                "DHR",
                "A",
                "MTD",
                "WAT",
                "BRKR",
                "RVTY",
                "BIO",
                "IQV",
                "RGEN",
                "EXAS",
                "GH",
                "NTRA",
                "QGEN",
                "DGX",
                "LH",
                "MYGN",
                "NVCR",
                "FTRE",
                "NEOG",
            ]
        )
    ),
    # =========================
    # INDUSTRIALS (XLI)
    # =========================
    "Industrials": sorted(
        set(
            [
                "GE",
                "RTX",
                "BA",
                "LMT",
                "NOC",
                "GD",
                "HII",
                "TDG",
                "HEI",
                "TXT",
                "DAL",
                "AAL",
                "UAL",
                "LUV",
                "ALK",
                "JBLU",
                "RYAAY",
                "CPA",
                "AZUL",
                "SAVE",
                "UNP",
                "CSX",
                "NSC",
                "CNI",
                "CP",
                "WAB",
                "GATX",
                "TRN",
                "GBX",
                "MATX",
                "SBLK",
                "EGLE",
                "GOGL",
                "DAC",
                "CMRE",
                "TK",
                "GLNG",
                "STNG",
                "INSW",
                "UPS",
                "FDX",
                "XPO",
                "GXO",
                "EXPD",
                "CHRW",
                "ATSG",
                "FWRD",
                "TFII",
                "ODFL",
                "ODFL",
                "SAIA",
                "KNX",
                "WERN",
                "SNDR",
                "HTLD",
                "ARCB",
                "LSTR",
                "MRTN",
                "CVLG",
                "GWW",
                "FAST",
                "MSM",
                "WCC",
                "MRC",
                "POOL",
                "SITE",
                "BECN",
                "DXPE",
                "AIT",
                "ROK",
                "ROP",
                "EMR",
                "AME",
                "IEX",
                "ITW",
                "DOV",
                "IR",
                "XYL",
                "CARR",
                "CAT",
                "DE",
                "CNHI",
                "AGCO",
                "OSK",
                "TEX",
                "MTW",
                "PCAR",
                "TTM",
                "OC",
                "MAS",
                "TREX",
                "AZEK",
                "JHX",
                "BLD",
                "AOS",
                "ALLE",
                "DOOR",
                "MHK",
                "ETN",
                "PH",
                "ABB",
                "HUBB",
                "LECO",
                "NVT",
                "ALSN",
                "ENS",
                "WCC",
                "J",
                "ACM",
                "KBR",
                "FLR",
                "TTEK",
                "PWR",
                "EME",
                "DY",
                "GVA",
                "MTZ",
                "WM",
                "RSG",
                "WCN",
                "GFL",
                "CWST",
                "CLH",
                "HCCI",
                "SRCL",
                "ECVT",
                "HON",
                "MMM",
                "DHR",
                "SWK",
                "CSL",
            ]
        )
    ),
    # =========================
    # MATERIALS (XLB)
    # =========================
    "Materials": sorted(
        set(
            [
                "DOW",
                "DD",
                "LYB",
                "WLK",
                "EMN",
                "APD",
                "LIN",
                "ECL",
                "CF",
                "CTVA",
                "SHW",
                "PPG",
                "ALB",
                "CE",
                "IFF",
                "FUL",
                "SXT",
                "AXTA",
                "AVNT",
                "NTR",
                "MOS",
                "FMC",
                "IPI",
                "SMG",
                "BG",
                "ADM",
                "AGFY",
                "AA",
                "CENX",
                "KALU",
                "NUE",
                "STLD",
                "CLF",
                "RS",
                "CMC",
                "HLMN",
                "MT",
                "TX",
                "GGB",
                "MLM",
                "VMC",
                "EXP",
                "SUM",
                "USLM",
                "BECN",
                "JHX",
                "AWI",
                "APG",
                "WY",
                "PCH",
                "BCC",
                "LPX",
                "UFPI",
                "IP",
                "PKG",
                "WRK",
                "MATV",
                "SUZ",
                "MERC",
                "BALL",
                "CCK",
                "BERY",
                "SEE",
                "GEF",
                "AMCR",
                "OI",
                "SLGN",
                "ATR",
                "NEM",
                "GOLD",
                "AEM",
                "FNV",
                "WPM",
                "KGC",
                "AU",
                "SSRM",
                "AGI",
                "BTG",
                "PAAS",
                "HL",
                "CDE",
                "AG",
                "MAG",
                "SVM",
                "SILV",
                "EXK",
                "FSM",
                "FCX",
                "SCCO",
                "TECK",
                "BHP",
                "RIO",
                "VALE",
                "MP",
                "S32",
                "HBM",
            ]
        )
    ),
    # =========================
    # REAL ESTATE (XLRE)
    # =========================
    "Real Estate": sorted(
        set(
            [
                "PLD",
                "REXR",
                "TRNO",
                "EGP",
                "STAG",
                "LXP",
                "PLYM",
                "BXP",
                "VNO",
                "KRC",
                "HIW",
                "CUZ",
                "PDM",
                "SLG",
                "DEI",
                "ESRT",
                "SPG",
                "KIM",
                "FRT",
                "REG",
                "BRX",
                "ROIC",
                "SKT",
                "MAC",
                "AVB",
                "EQR",
                "ESS",
                "UDR",
                "CPT",
                "MAA",
                "INVH",
                "AMH",
                "AMT",
                "CCI",
                "SBAC",
                "DLR",
                "EQIX",
                "IRM",
                "IIPR",
                "WY",
                "GLPI",
                "UNIT",
                "NLY",
                "AGNC",
                "STWD",
                "BXMT",
                "RITM",
                "MFA",
                "TWO",
                "CIM",
                "PMT",
                "DX",
                "CBRE",
                "JLL",
                "CIGI",
                "ZG",
                "COMP",
                "EXPI",
                "OPAD",
                "OPEN",
                "HHH",
                "JOE",
                "FPH",
                "ALEX",
                "STRS",
                "FOR",
                "WPC",
                "O",
                "NNN",
                "ADC",
                "ARE",
                "VICI",
                "KIM",
                "REG",
                "FRT",
                "SPG",
            ]
        )
    ),
    # =========================
    # CONSUMER STAPLES (XLP)
    # =========================
    "Consumer Staples": sorted(
        set(
            [
                "PG",
                "CL",
                "KMB",
                "CHD",
                "CLX",
                "EL",
                "UL",
                "EPC",
                "NWL",
                "KO",
                "PEP",
                "KDP",
                "MNST",
                "CELH",
                "COKE",
                "FIZZ",
                "STZ",
                "BF-B",
                "DEO",
                "MGPI",
                "CCU",
                "TAP",
                "SAM",
                "PM",
                "MO",
                "BTI",
                "VGR",
                "UVV",
                "RLX",
                "SYY",
                "USFD",
                "PFGC",
                "UNFI",
                "CHEF",
                "SPTN",
                "GIS",
                "K",
                "KHC",
                "MDLZ",
                "HSY",
                "CPB",
                "CAG",
                "TSN",
                "HRL",
                "POST",
                "KR",
                "ACI",
                "SFM",
                "IMKTA",
                "WMK",
                "GO",
                "BJ",
                "TGT",
                "WMT",
                "COST",
                "DG",
                "DLTR",
                "FIVE",
                "OLLI",
                "BURL",
                "TJX",
                "ROST",
            ]
        )
    ),
    # =========================
    # CONSUMER DISCRETIONARY (XLY)
    # =========================
    "Consumer Discretionary": sorted(
        set(
            [
                "TSLA",
                "TM",
                "HMC",
                "GM",
                "F",
                "RIVN",
                "LCID",
                "NIO",
                "STLA",
                "KMX",
                "LAD",
                "AN",
                "PAG",
                "SAH",
                "ABG",
                "GPI",
                "CVNA",
                "ACVA",
                "VRM",
                "GPC",
                "AZO",
                "ORLY",
                "AAP",
                "LKQ",
                "MGA",
                "BWA",
                "AXL",
                "DORM",
                "VC",
                "ULTA",
                "BBY",
                "RH",
                "WSM",
                "TSCO",
                "ASO",
                "BBWI",
                "DDS",
                "BKE",
                "DKS",
                "LULU",
                "ANF",
                "URBN",
                "GPS",
                "AEO",
                "TPR",
                "CPRI",
                "TJX",
                "ROST",
                "BURL",
                "NKE",
                "DECK",
                "CROX",
                "SKX",
                "UAA",
                "VFC",
                "ONON",
                "SHOO",
                "WWW",
                "BOOT",
                "AMZN",
                "MELI",
                "ETSY",
                "PDD",
                "JD",
                "BABA",
                "EBAY",
                "W",
                "RVLV",
                "CHWY",
                "MCD",
                "SBUX",
                "CMG",
                "YUM",
                "QSR",
                "DPZ",
                "WEN",
                "DRI",
                "TXRH",
                "SHAK",
                "HD",
                "LOW",
                "FND",
                "TTSH",
                "CCL",
                "RCL",
                "NCLH",
                "PLNT",
                "EDR",
                "LYV",
                "PTON",
                "MCW",
                "LVS",
                "MGM",
                "WYNN",
                "CZR",
                "MLCO",
                "PENN",
                "BYD",
                "CHDN",
                "RRR",
                "MAR",
                "HLT",
                "H",
                "IHG",
                "CHH",
                "WH",
                "HST",
                "PK",
                "DRH",
                "THO",
                "WGO",
                "CWH",
                "PATK",
                "LCII",
                "PII",
                "MBUU",
                "MCFT",
                "BC",
                "WHR",
                "LEG",
                "SNBR",
                "TILE",
                "MHK",
                "SNA",
                "SWK",
                "NWL",
                "AOS",
                "KTB",
                "BKNG",
                "EXPE",
                "TRIP",
                "ABNB",
                "TCOM",
                "SABR",
                "DESP",
                "CWT",
                "HGV",
                "MMYT",
            ]
        )
    ),
    # =========================
    # UTILITIES (XLU)
    # =========================
    "Utilities": sorted(
        set(
            [
                "NEE",
                "DUK",
                "SO",
                "AEP",
                "EXC",
                "D",
                "PCG",
                "EIX",
                "PEG",
                "PPL",
                "ATO",
                "OGS",
                "NWN",
                "NI",
                "NJR",
                "SWX",
                "UGI",
                "SR",
                "SRE",
                "CPK",
                "AWK",
                "WTRG",
                "AWR",
                "MSEX",
                "CWCO",
                "ARTNA",
                "YORW",
                "GWRS",
                "NRG",
                "AES",
                "CEG",
                "VST",
                "BEP",
                "CWEN",
                "ORA",
                "AQN",
                "RUN",
                "SHLS",
            ]
        )
    ),
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

    if not tickers_list:
        return pd.DataFrame()

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


# =====================================================
# Load Data (Sector ETFs + SPY)
# =====================================================
tickers = list(SECTOR_ETFS.values()) + [BENCHMARK]
prices = download_close(tickers, period="2y", interval="1d")

# =====================================================
# Compute Sector-Level Metrics
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
        st.caption("Tip: Sector pick drives the stocks below.")

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
st.markdown("## 📈 Buy/Sell Signals — Carried Sector")

carried_sector = st.session_state.get("selected_sector")
stock_df = pd.DataFrame()

if not carried_sector:
    st.warning("Please pick a sector above.")
else:
    etf = SECTOR_ETFS[carried_sector]
    stock_list = SECTOR_STOCKS.get(carried_sector, [])
    if not stock_list:
        st.warning(f"No stocks found for sector: {carried_sector}.")
    else:
        st.info(f"**Carried Sector:** {carried_sector} (ETF: {etf})  |  Constituents: {len(stock_list)}")
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
st.markdown("## 📉 Stocks (Carried Sector)")

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

    carried_sector = st.session_state.get("selected_sector")
    stock_list = SECTOR_STOCKS.get(carried_sector, []) if carried_sector else []

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
            st.warning("No sufficient data to compute stock returns for this sector.")
        else:
            plot_col = "ReturnPct"
            plot_df = df_stockperf.copy()

            # --- Build figure (go.Bar to lock y labels) ---
            dfp = plot_df.dropna(subset=["ReturnPct"]).copy()
            dfp["Stock"] = dfp["Stock"].astype(str)
            dfp = dfp.sort_values("ReturnPct", ascending=False)

            y_vals = dfp["Stock"].tolist()
            x_vals = dfp["ReturnPct"].tolist()

            vmin, vmax = float(dfp["ReturnPct"].min()), float(dfp["ReturnPct"].max())
            max_abs = max(abs(vmin), abs(vmax))

            if vmin < 0 < vmax:
                color_arr = dfp["ReturnPct"].values
                cmin, cmax = -max_abs, max_abs
                colorscale = "RdYlGn"
                cbar_title = f"{ranking_choice} (%)"
            elif vmin >= 0:
                color_arr = dfp["ReturnPct"].values
                cmin, cmax = 0, vmax
                colorscale = "Greens"
                cbar_title = f"{ranking_choice} (%)"
            else:
                color_arr = (-dfp["ReturnPct"]).values  # loss magnitude
                cmin, cmax = 0, max_abs
                colorscale = "Reds"
                cbar_title = "Loss magnitude (%)"

            fig_stocks = go.Figure(
                data=go.Bar(
                    x=x_vals,
                    y=y_vals,
                    orientation="h",
                    marker=dict(
                        color=color_arr, colorscale=colorscale, cmin=cmin, cmax=cmax, colorbar=dict(title=cbar_title)
                    ),
                    hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
                )
            )

            # Pin y-axis labels to the bar order
            fig_stocks.update_yaxes(categoryorder="array", categoryarray=y_vals, automargin=True)
            fig_stocks.update_xaxes(title=f"{ranking_choice} (%)", ticksuffix="%")
            fig_stocks.update_layout(
                title=f"{carried_sector} — Stock Performance — {ranking_choice}",
                template="plotly_white",
                margin=dict(l=100, r=20, t=60, b=40),
                bargap=0.45,
                bargroupgap=0.25,
            )

            # === ROW: Stock bar (left) + Context/summary card (right) ===
            row1 = st.columns([3, 2], gap="large")
            with row1[0]:
                st.plotly_chart(fig_stocks, use_container_width=True, config=PLOT_CONFIG)
            with row1[1]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### 🧭 Context")
                st.markdown(f"**Sector:** `{carried_sector}`  ")
                st.markdown(f"**ETF:** `{SECTOR_ETFS[carried_sector]}`  ")
                # quick counts by signal (if available)
                if not stock_df.empty and "Signal" in stock_df.columns:
                    counts = stock_df["Signal"].value_counts().to_dict()
                    st.caption("Signal counts")
                    st.write(counts)
                st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error building stocks horizontal barplot: {e}")

# =======================
# Minimal Stock Chart area
# =======================
if df.empty:
    st.warning("No sector metrics available. Check price data upstream.")
else:
    sector_options = df["Sector"].dropna().unique().tolist()
    if not sector_options:
        st.warning("No sectors available.")
        st.stop()

    carried_sector = st.session_state.get("selected_sector")
    if not carried_sector or carried_sector not in sector_options:
        carried_sector = sector_options[0]
        st.session_state["selected_sector"] = carried_sector

    stock_list = SECTOR_STOCKS.get(carried_sector, [])
    if not stock_list:
        st.warning(f"No stocks listed for sector: {carried_sector}")
        st.stop()

    st.markdown("#### 🧭 Context: Sector ETF")
    etf_ticker = SECTOR_ETFS[carried_sector]
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
                    title=f"{selected_stock} — {carried_sector}",
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)

    except Exception as e:
        st.error(f"Error fetching/plotting {selected_stock}: {e}")

st.divider()
st.caption("Source: Yahoo Finance | Updated hourly | © 2025 TradeFit Scan")
