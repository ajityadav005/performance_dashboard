"""
app_final_updated.py
─────────────────────────────────────────────────────────────────────
Portfolio Analytics Dashboard — Streamlit application.

Logging strategy
────────────────
• Every logical boundary (data load, metric computation, chart render,
  tab entry, error path) emits a structured log record via get_logger().
• DEBUG  : fine-grained internal state (slice sizes, loop counters …)
• INFO   : normal lifecycle events (file loaded, tab rendered …)
• WARNING: recoverable issues (empty series, bad date range …)
• ERROR  : caught exceptions that prevent a section from rendering
• All records go to stdout AND logs/portfolio_analytics.log (rotating,
  5 MB cap, 3 backups) via logger.py.
─────────────────────────────────────────────────────────────────────
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

# ── Internal ──────────────────────────────────────────────────────────
from logger import get_logger

# Acquire a module-level logger; all records from this file will be
# tagged with the module name for easy filtering in log files.
log = get_logger(__name__)
log.info("app_final_updated.py loaded — Streamlit session starting")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)
log.debug("Streamlit page config applied")

# ─────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #04080F;
    color: #E8EFF8;
}

/* Force all text to be visible */
p, span, div, label, li, td, th {
    color: #E8EFF8;
}

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0D1626 0%, #111E35 100%);
    border: 1px solid #243855;
    border-radius: 14px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 0%, #00CFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.dash-sub {
    color: #8BAAC8;
    font-size: 13px;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* Streamlit native metric labels & values */
[data-testid="stMetricLabel"] {
    color: #8BAAC8 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 28px !important;
    font-weight: 700 !important;
    font-family: 'Space Mono', monospace !important;
}

/* Section titles */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
}
.section-sub {
    color: #8BAAC8;
    font-size: 13px;
    margin-bottom: 16px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0A1220;
    border-bottom: 1px solid #1A2B45;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #8BAAC8 !important;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    padding: 10px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #00CFFF !important;
    border-bottom: 2px solid #00CFFF !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
    background: transparent;
}

/* Slider labels */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p {
    color: #C8D8EA !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
.stSlider [data-testid="stWidgetLabel"] p {
    color: #C8D8EA !important;
    font-size: 14px !important;
}

/* Date input labels */
[data-testid="stDateInput"] label,
[data-testid="stDateInput"] p {
    color: #C8D8EA !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* File uploader text */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span {
    color: #C8D8EA !important;
    font-size: 14px !important;
}

/* All widget labels globally */
[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stDateInput label,
.stNumberInput label,
.stTextInput label {
    color: #C8D8EA !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}

/* Dataframe text */
.stDataFrame {
    background: #0A1220 !important;
}
.stDataFrame td, .stDataFrame th {
    color: #E8EFF8 !important;
}

/* Markdown text */
.stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #E8EFF8 !important;
    font-size: 14px;
}
.stMarkdown strong {
    color: #FFFFFF !important;
}

/* Divider */
hr {
    border-color: #243855;
    margin: 20px 0;
}

/* Info box */
.info-box {
    background: rgba(0,207,255,0.08);
    border: 1px solid rgba(0,207,255,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #B8D8EA;
    margin-bottom: 16px;
    font-family: 'Space Mono', monospace;
}
.warning-box {
    background: rgba(255,184,77,0.08);
    border: 1px solid rgba(255,184,77,0.25);
    border-radius: 10px;
    padding: 14px 18px;
    font-size: 13px;
    color: #FFB84D;
    margin-bottom: 16px;
}

/* Hide streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {
    padding-top: 20px;
    padding-bottom: 40px;
    max-width: 1400px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PALETTE (no red/green for portfolios)
# ─────────────────────────────────────────────
PALETTE = [
    '#00CFFF', '#A78BFA', '#FFB84D', '#F472B6', '#38BDF8',
    '#E879F9', '#FACC15', '#60A5FA', '#FB923C', '#818CF8',
    '#FCD34D', '#67E8F9', '#C084FC', '#FCA5A5', '#93C5FD',
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(10,18,32,0.6)',
    font=dict(family='DM Sans', color='#C8D8EA', size=12),
    xaxis=dict(
        gridcolor='rgba(26,43,69,0.5)',
        linecolor='rgba(36,56,85,0.8)',
        tickcolor='rgba(26,43,69,0)',
        tickfont=dict(family='Space Mono', size=11, color='#C8D8EA'),
        title_font=dict(family='DM Sans', size=13, color='#E8EFF8'),
        showgrid=True,
    ),
    yaxis=dict(
        gridcolor='rgba(26,43,69,0.5)',
        linecolor='rgba(36,56,85,0.8)',
        tickcolor='rgba(26,43,69,0)',
        tickfont=dict(family='Space Mono', size=11, color='#C8D8EA'),
        title_font=dict(family='DM Sans', size=13, color='#E8EFF8'),
        showgrid=True,
    ),
    legend=dict(
        bgcolor='rgba(10,18,32,0.8)',
        bordercolor='#243855',
        borderwidth=1,
        font=dict(family='DM Sans', size=12, color='#E8EFF8'),
    ),
    margin=dict(l=60, r=30, t=40, b=60),
    hovermode='x unified',
)

# ─────────────────────────────────────────────
# CALCULATION ENGINE
# ─────────────────────────────────────────────

def detect_frequency(dates: pd.DatetimeIndex) -> str:
    """
    Infer data frequency from the average gap between consecutive dates.
    Returns 'daily' when the average gap is ≤ 6 days, else 'monthly'.
    Logs the detected frequency for traceability.
    """
    if len(dates) < 2:
        # Cannot determine frequency from a single point; default to monthly
        log.warning("detect_frequency: fewer than 2 dates — defaulting to 'monthly'")
        return 'monthly'

    avg_days = (dates[-1] - dates[0]).days / len(dates)
    freq = 'daily' if avg_days <= 6 else 'monthly'
    log.info("detect_frequency: avg_days=%.1f → frequency='%s'", avg_days, freq)
    return freq


def compute_returns(nav: pd.Series) -> pd.Series:
    """
    Compute simple period-on-period returns from a NAV series.
    The first value is NaN (no prior period) and is dropped.
    """
    returns = nav.pct_change().dropna()
    log.debug("compute_returns: input_len=%d → returns_len=%d", len(nav), len(returns))
    return returns


def to_monthly_nav(nav: pd.Series) -> pd.Series:
    """
    Resample a daily NAV series to month-end by taking the last value in
    each calendar month.  For monthly input this is a no-op (harmless).
    """
    monthly = nav.resample('ME').last().dropna()
    log.debug("to_monthly_nav: daily_len=%d → monthly_len=%d", len(nav), len(monthly))
    return monthly


def compute_metrics(nav: pd.Series, ann_factor: int) -> dict:
    """
    Compute a full suite of annualised risk/return metrics for one NAV series.

    Parameters
    ----------
    nav        : NAV price series (DatetimeIndex).
    ann_factor : 252 for daily data, 12 for monthly data.

    Returns
    -------
    dict of metrics, or empty dict when there is insufficient data.

    Metrics computed
    ────────────────
    • CAGR            — Compound Annual Growth Rate
    • ann_std         — Annualised standard deviation of period returns
    • cv              — Coefficient of Variation (ann_std / |CAGR|)
    • sharpe          — Sharpe Ratio assuming RF = 0 %
    • max_dd          — Maximum peak-to-trough drawdown (as a fraction)
    • max_dd_date     — Date the maximum drawdown was reached
    • sortino         — Sortino Ratio (downside deviation denominator)
    • calmar          — Calmar Ratio (CAGR / |max drawdown|)
    """
    log.debug("compute_metrics: series='%s' len=%d ann_factor=%d",
              nav.name, len(nav), ann_factor)

    nav = nav.dropna()
    if len(nav) < 2:
        log.warning("compute_metrics: series too short (len=%d) — returning empty dict", len(nav))
        return {}

    ret = nav.pct_change().dropna()
    if len(ret) < 2:
        log.warning("compute_metrics: fewer than 2 return observations — returning empty dict")
        return {}

    n = len(ret)
    years = n / ann_factor

    # ── CAGR ─────────────────────────────────────────────────────────
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    log.debug("compute_metrics: CAGR=%.4f years=%.2f", cagr, years)

    # ── Volatility ───────────────────────────────────────────────────
    ann_std = ret.std() * np.sqrt(ann_factor)

    # ── Coefficient of Variation ──────────────────────────────────────
    cv = ann_std / abs(cagr) if cagr != 0 else np.nan

    # ── Sharpe Ratio (RF = 0 %) ───────────────────────────────────────
    sharpe = cagr / ann_std if ann_std != 0 else np.nan

    # ── Maximum Drawdown ──────────────────────────────────────────────
    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    log.debug("compute_metrics: max_dd=%.4f at %s", max_dd, max_dd_date)

    # ── Sortino Ratio ─────────────────────────────────────────────────
    neg_ret = ret[ret < 0]
    if len(neg_ret) > 1:
        downside_std = np.sqrt((neg_ret ** 2).mean()) * np.sqrt(ann_factor)
        sortino = cagr / downside_std if downside_std != 0 else np.nan
    else:
        # Fewer than 2 negative return periods — Sortino is undefined
        log.warning("compute_metrics: insufficient negative returns for Sortino (n_neg=%d)", len(neg_ret))
        sortino = np.nan

    # ── Calmar Ratio ──────────────────────────────────────────────────
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    metrics = {
        'start_date': nav.index[0],
        'end_date': nav.index[-1],
        'years': years,
        'cagr': cagr,
        'ann_std': ann_std,
        'cv': cv,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'max_dd_date': max_dd_date,
        'sortino': sortino,
        'calmar': calmar,
        'n_obs': n,
    }
    log.info("compute_metrics: completed for series '%s' — CAGR=%.2f%% Sharpe=%.2f MaxDD=%.2f%%",
             nav.name, cagr * 100, sharpe if sharpe is not np.nan else float('nan'), max_dd * 100)
    return metrics


def rolling_metric(nav: pd.Series, window_months: int, metric: str, ann_factor: int) -> pd.Series:
    """
    Compute a rolling metric over a sliding window on monthly NAV data.

    Parameters
    ----------
    nav           : Raw NAV series (daily or monthly).
    window_months : Width of the sliding window in calendar months.
    metric        : One of 'std', 'sharpe', 'cagr'.
    ann_factor    : 252 for daily input, 12 for monthly input.

    Returns
    -------
    pd.Series indexed by the window end-date, or an empty Series when
    there is not enough data.

    Implementation notes
    ────────────────────
    • Daily NAVs are first resampled to month-end so the window is always
      measured in months regardless of the raw frequency.
    • Each window is a pure slice of monthly observations — no overlapping
      daily windows — which keeps observations independent.
    • std   : annualised std dev of monthly returns (× √12) in percent.
    • sharpe: CAGR / annualised std dev (RF = 0 %).
    • cagr  : compound annual growth rate of the window in percent.
    """
    log.debug("rolling_metric: metric='%s' window=%d months series='%s'",
              metric, window_months, nav.name)

    # Resample to monthly if the source is daily
    monthly = to_monthly_nav(nav) if ann_factor == 252 else nav.copy()
    monthly = monthly.dropna()
    n = len(monthly)

    if n < window_months:
        # Not enough monthly observations to fill even one window
        log.warning("rolling_metric: only %d monthly obs for window=%d — returning empty",
                    n, window_months)
        return pd.Series(dtype=float)

    nav_arr = monthly.values
    idx = monthly.index
    results = {}

    for i in range(window_months - 1, n):
        slice_nav = nav_arr[i - window_months + 1: i + 1]

        # Skip any window that contains NaN values
        if np.any(np.isnan(slice_nav)):
            log.debug("rolling_metric: NaN in window ending %s — skipped", idx[i])
            continue

        start_val, end_val = slice_nav[0], slice_nav[-1]
        if start_val <= 0:
            # Guard against zero or negative starting NAV (would give nonsensical CAGR)
            log.warning("rolling_metric: non-positive start_val=%.4f at %s — skipped",
                        start_val, idx[i])
            continue

        ret = np.diff(slice_nav) / slice_nav[:-1]
        if len(ret) < 2:
            continue

        if metric == 'std':
            # Annualised standard deviation of monthly returns (returned as %)
            val = float(np.std(ret, ddof=1) * np.sqrt(12) * 100)
        elif metric == 'sharpe':
            # CAGR over the window divided by annualised std dev (RF = 0 %)
            ann_ret = (end_val / start_val) ** (12 / window_months) - 1
            ann_std = float(np.std(ret, ddof=1) * np.sqrt(12))
            val = ann_ret / ann_std if ann_std > 0 else np.nan
        elif metric == 'cagr':
            # Annualised compound return over the window (returned as %)
            val = float(((end_val / start_val) ** (12 / window_months) - 1) * 100)
        else:
            log.error("rolling_metric: unknown metric '%s'", metric)
            val = np.nan

        results[idx[i]] = val

    log.info("rolling_metric: metric='%s' window=%d → %d data-points computed",
             metric, window_months, len(results))
    return pd.Series(results)


# ─────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────

def apply_layout(fig, title='', yaxis_title='', xaxis_title=''):
    """
    Apply the shared dark-theme Plotly layout to a figure.
    Logs the chart title for breadcrumb tracing in the log file.
    """
    fig.update_layout(**PLOTLY_LAYOUT)
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(family='Syne', size=15, color='#FFFFFF'), x=0)
        )
    if yaxis_title:
        fig.update_yaxes(
            title_text=yaxis_title,
            title_font=dict(size=13, color='#E8EFF8', family='DM Sans'),
            tickfont=dict(size=11, color='#C8D8EA', family='Space Mono'),
        )
    if xaxis_title:
        fig.update_xaxes(
            title_text=xaxis_title,
            title_font=dict(size=13, color='#E8EFF8', family='DM Sans'),
            tickfont=dict(size=11, color='#C8D8EA', family='Space Mono'),
        )
    # Ensure axis tick fonts are always visible even without explicit axis titles
    fig.update_xaxes(tickfont=dict(size=11, color='#C8D8EA', family='Space Mono'))
    fig.update_yaxes(tickfont=dict(size=11, color='#C8D8EA', family='Space Mono'))
    log.debug("apply_layout: chart='%s' styled", title or '(untitled)')
    return fig


def fmt_pct(v):
    """Format a decimal fraction as a signed percentage string (e.g. +12.34 %)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v*100:+.2f}%"


def fmt_val(v, decimals=2):
    """Format a numeric value to a fixed number of decimal places."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f"{v:.{decimals}f}"


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
log.debug("Rendering dashboard header")
st.markdown("""
<div class="dash-header">
  <div style="width:40px;height:40px;background:linear-gradient(135deg,#00CFFF,#A78BFA);border-radius:10px;
              display:flex;align-items:center;justify-content:center;font-family:'Space Mono';font-weight:700;
              font-size:16px;color:#000;flex-shrink:0">P∑</div>
  <div>
    <div class="dash-title">Portfolio Analytics Dashboard</div>
    <div class="dash-sub">Upload NAV data · Explore performance · Risk-Free Rate = 0%</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
log.debug("Rendering file uploader widget")
upload_col, info_col = st.columns([2, 1])

with upload_col:
    uploaded = st.file_uploader(
        "Upload Excel / CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="First column = Date | Remaining columns = Portfolio NAV series",
        label_visibility='collapsed',
    )

with info_col:
    st.markdown("""
    <div class="info-box">
    📋 <b>Format:</b> First column = Date<br>
    &nbsp;&nbsp;&nbsp;&nbsp;Remaining columns = Portfolio NAV<br>
    📅 Daily or Monthly data supported<br>
    ⚠️ RF Rate = 0% assumed throughout
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────

@st.cache_data
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a cleaned DataFrame.

    Decorated with @st.cache_data so the file is parsed only once per
    unique (file_bytes, filename) pair — subsequent reruns reuse the
    cached result, avoiding redundant I/O.

    Steps performed
    ───────────────
    1. Parse file bytes into a DataFrame (CSV or Excel auto-detected).
    2. Parse the index as datetimes with dayfirst=True (DD/MM/YYYY support).
    3. Sort chronologically.
    4. Coerce all value columns to numeric, silencing non-numeric tokens.
    5. Drop rows where every column is NaN (completely empty rows).
    """
    log.info("load_data: reading file='%s'", filename)

    if filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=[0], index_col=0)
        log.debug("load_data: CSV parsed, raw shape=%s", df.shape)
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), parse_dates=[0], index_col=0)
        log.debug("load_data: Excel parsed, raw shape=%s", df.shape)

    # Normalise the DatetimeIndex (handles mixed day/month ordering)
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index()

    # Coerce: any cell that cannot be interpreted as a number becomes NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(how='all', inplace=True)

    log.info("load_data: clean DataFrame ready — shape=%s date_range=[%s → %s]",
             df.shape, df.index[0].date(), df.index[-1].date())
    return df


# ── Guard: nothing to show without a file ────────────────────────────
if uploaded is None:
    log.info("No file uploaded yet — showing placeholder screen")
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;color:#3D5470;">
        <div style="font-size:48px;margin-bottom:16px;opacity:0.4">📊</div>
        <div style="font-family:'Syne',sans-serif;font-size:18px;color:#6B85A8;">Upload a file above to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Read the uploaded bytes and parse the DataFrame ──────────────────
file_bytes = uploaded.read()
log.info("File received: name='%s' size=%d bytes", uploaded.name, len(file_bytes))

try:
    df = load_data(file_bytes, uploaded.name)
except Exception as e:
    # Unrecoverable parse error — surface to user and halt execution
    log.error("load_data failed for '%s': %s", uploaded.name, e, exc_info=True)
    st.error(f"Error reading file: {e}")
    st.stop()

portfolios = df.columns.tolist()
freq = detect_frequency(df.index)
ann_factor = 252 if freq == 'daily' else 12
log.info("Dataset ready: portfolios=%s freq='%s' ann_factor=%d obs=%d",
         portfolios, freq, ann_factor, len(df))


# ─────────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────────
log.debug("Rendering status bar metrics")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Portfolios", len(portfolios))
with c2:
    st.metric("Observations", len(df))
with c3:
    st.metric("Start Date", df.index[0].strftime('%d %b %Y'))
with c4:
    st.metric("End Date", df.index[-1].strftime('%d %b %Y'))

st.markdown("---")


# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
log.debug("Rendering main tab layout")
tab_summary, tab_perf, tab_nav, tab_dd, tab_rstd, tab_rsharpe, tab_rcagr, tab_yearly, tab_dist = st.tabs([
    "📋 Summary",
    "📈 Performance",
    "📉 NAV Curve",
    "🔻 Drawdown",
    "〰 Rolling Std",
    "⚡ Rolling Sharpe",
    "🚀 Rolling CAGR",
    "📅 Yearly Returns",
    "🔔 Distribution",
])


# ══════════════════════════════════════════════
# TAB 1 — SUMMARY
# ══════════════════════════════════════════════
with tab_summary:
    log.info("TAB: Summary — rendering portfolio summary cards")
    st.markdown('<div class="section-title">Data Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">{len(portfolios)} portfolios · {len(df)} observations · {freq} frequency</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(portfolios), 4))
    for i, p in enumerate(portfolios):
        nav = df[p].dropna()
        total_ret = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
        log.debug("Summary card '%s': start_nav=%.2f end_nav=%.2f total_ret=%.2f%%",
                  p, nav.iloc[0], nav.iloc[-1], total_ret)
        color = PALETTE[i % len(PALETTE)]
        with cols[i % len(cols)]:
            sign = '+' if total_ret >= 0 else ''
            st.markdown(f"""
            <div style="background:#0F1A2E;border:1px solid #1A2B45;border-top:3px solid {color};
                        border-radius:12px;padding:16px;margin-bottom:12px">
              <div style="font-family:'Syne',sans-serif;font-weight:700;color:{color};margin-bottom:10px">{p}</div>
              <div style="display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid #1A2B45">
                <span style="color:#6B85A8">Start NAV</span>
                <span style="font-family:'Space Mono';color:#E2EAF4">{nav.iloc[0]:.2f}</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid #1A2B45">
                <span style="color:#6B85A8">End NAV</span>
                <span style="font-family:'Space Mono';color:#E2EAF4">{nav.iloc[-1]:.2f}</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid #1A2B45">
                <span style="color:#6B85A8">Total Return</span>
                <span style="font-family:'Space Mono';color:{'#00E5A0' if total_ret >= 0 else '#FF4E6A'}">{sign}{total_ret:.2f}%</span>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:12px;padding:3px 0">
                <span style="color:#6B85A8">Valid Obs</span>
                <span style="font-family:'Space Mono';color:#E2EAF4">{len(nav)}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("**Data Preview (first 10 rows)**")
    preview_df = df.head(10).round(2)
    st.dataframe(
        preview_df.style.format("{:.2f}").set_properties(**{'text-align': 'center'}),
        use_container_width=True,
    )
    log.debug("TAB Summary: complete")


# ══════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════
with tab_perf:
    log.info("TAB: Performance — computing full-period metrics for %d portfolios", len(portfolios))
    st.markdown('<div class="section-title">Performance Measures</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Risk-Free Rate = 0% · Annualisation factor auto-detected from data frequency</div>',
        unsafe_allow_html=True,
    )

    # ── Full-period metrics table ─────────────────────────────────────
    all_metrics = {}
    for p in portfolios:
        log.debug("Performance tab: computing metrics for '%s'", p)
        all_metrics[p] = compute_metrics(df[p].dropna(), ann_factor)

    # Row definitions: label → formatting lambda applied to the metrics dict
    metric_rows = {
        'Research Period': lambda m: f"{m['start_date'].strftime('%b %Y')} – {m['end_date'].strftime('%b %Y')} ({m['years']:.1f} yrs)",
        'CAGR': lambda m: fmt_pct(m['cagr']),
        'Annual Std Dev': lambda m: fmt_pct(m['ann_std']),
        'Coefficient of Variation': lambda m: fmt_val(m['cv']),
        'Sharpe Ratio': lambda m: fmt_val(m['sharpe']),
        'Max Drawdown': lambda m: fmt_pct(m['max_dd']),
        'Sortino Ratio': lambda m: fmt_val(m['sortino']),
        'Calmar Ratio': lambda m: fmt_val(m['calmar']),
    }

    perf_data = {
        label: [fn(all_metrics[p]) if all_metrics.get(p) else '—' for p in portfolios]
        for label, fn in metric_rows.items()
    }
    perf_df = pd.DataFrame(perf_data, index=portfolios).T
    log.debug("Performance tab: metrics table shape=%s", perf_df.shape)
    st.dataframe(perf_df, use_container_width=True)

    st.markdown("---")

    # ── Quick reference cards ─────────────────────────────────────────
    log.debug("Performance tab: rendering quick-reference CAGR cards")
    st.markdown('<div class="section-title" style="font-size:15px">Quick Reference</div>', unsafe_allow_html=True)
    qcols = st.columns(min(len(portfolios), 5))
    for i, p in enumerate(portfolios):
        m = all_metrics.get(p, {})
        if not m:
            log.warning("Performance tab quick-reference: no metrics for '%s' — skipped", p)
            continue
        color = PALETTE[i % len(PALETTE)]
        cagr_val = m['cagr'] * 100
        sign = '+' if cagr_val >= 0 else ''
        with qcols[i % len(qcols)]:
            st.markdown(f"""
            <div style="background:#0F1A2E;border:1px solid #1A2B45;border-top:2px solid {color};
                        border-radius:10px;padding:14px;text-align:center">
              <div style="font-size:11px;color:{color};text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px">{p}</div>
              <div style="font-family:'Space Mono';font-size:18px;font-weight:700;
                          color:{'#00E5A0' if cagr_val >= 0 else '#FF4E6A'}">{sign}{cagr_val:.2f}%</div>
              <div style="font-size:11px;color:#3D5470;font-family:'Space Mono'">Sharpe: {fmt_val(m.get('sharpe'))}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Custom Period Analysis ────────────────────────────────────────
    log.debug("Performance tab: rendering custom period date inputs")
    st.markdown('<div class="section-title" style="font-size:15px">📅 Custom Period Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Select a sub-period to compute performance metrics for that window only</div>',
        unsafe_allow_html=True,
    )

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        perf_start = st.date_input(
            "Start Date",
            value=df.index[0].date(),
            min_value=df.index[0].date(),
            max_value=df.index[-1].date(),
            key='perf_start',
        )
    with p_col2:
        perf_end = st.date_input(
            "End Date",
            value=df.index[-1].date(),
            min_value=df.index[0].date(),
            max_value=df.index[-1].date(),
            key='perf_end',
        )

    if perf_start >= perf_end:
        # User has given an invalid date range — warn but do not crash
        log.warning("Performance tab custom period: invalid range perf_start=%s >= perf_end=%s",
                    perf_start, perf_end)
        st.markdown('<div class="warning-box">⚠️ Start date must be before end date</div>', unsafe_allow_html=True)
    else:
        df_period = df.loc[str(perf_start):str(perf_end)]
        log.info("Performance tab custom period: [%s → %s] obs=%d", perf_start, perf_end, len(df_period))

        if len(df_period) < 2:
            log.warning("Performance tab custom period: only %d row(s) in selected window — too short",
                        len(df_period))
            st.markdown('<div class="warning-box">⚠️ Not enough data in selected period</div>', unsafe_allow_html=True)
        else:
            # Recompute all metrics restricted to the chosen sub-period
            period_metrics = {p: compute_metrics(df_period[p].dropna(), ann_factor) for p in portfolios}
            period_data = {
                label: [fn(period_metrics[p]) if period_metrics.get(p) else '—' for p in portfolios]
                for label, fn in metric_rows.items()
            }
            period_df = pd.DataFrame(period_data, index=portfolios).T
            st.markdown(f"**Period: {perf_start.strftime('%d %b %Y')} → {perf_end.strftime('%d %b %Y')}**")
            st.dataframe(period_df, use_container_width=True)
            log.debug("Performance tab custom period: table rendered, shape=%s", period_df.shape)


# ══════════════════════════════════════════════
# TAB 3 — NAV CURVE
# ══════════════════════════════════════════════
with tab_nav:
    log.info("TAB: NAV Curve — rendering NAV chart")
    st.markdown('<div class="section-title">NAV Curves</div>', unsafe_allow_html=True)

    n_col1, n_col2 = st.columns(2)
    with n_col1:
        nav_start = st.date_input(
            "Start Date", value=df.index[0].date(),
            min_value=df.index[0].date(), max_value=df.index[-1].date(),
            key='nav_start',
        )
    with n_col2:
        nav_end = st.date_input(
            "End Date", value=df.index[-1].date(),
            min_value=df.index[0].date(), max_value=df.index[-1].date(),
            key='nav_end',
        )

    if nav_start >= nav_end:
        log.warning("NAV Curve tab: invalid date range nav_start=%s >= nav_end=%s", nav_start, nav_end)
        st.warning("Start date must be before end date")
    else:
        df_nav = df.loc[str(nav_start):str(nav_end)]
        log.debug("NAV Curve tab: plotting %d portfolios over %d observations", len(portfolios), len(df_nav))

        fig = go.Figure()
        for i, p in enumerate(portfolios):
            nav = df_nav[p].dropna()
            log.debug("NAV Curve tab: adding trace '%s' len=%d", p, len(nav))
            fig.add_trace(go.Scatter(
                x=nav.index, y=nav.values,
                name=p,
                line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
                mode='lines',
                hovertemplate=f'<b>{p}</b><br>%{{x|%b %Y}}<br>NAV: %{{y:.2f}}<extra></extra>',
            ))

        apply_layout(fig, yaxis_title='NAV Value')
        fig.update_xaxes(tickformat='%b %Y')
        st.plotly_chart(fig, use_container_width=True)
        log.info("NAV Curve tab: chart rendered")


# ══════════════════════════════════════════════
# TAB 4 — DRAWDOWN
# ══════════════════════════════════════════════
with tab_dd:
    log.info("TAB: Drawdown — computing and rendering drawdown chart")
    st.markdown('<div class="section-title">Drawdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Peak-to-trough decline · Max drawdown point marked with ✱</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    for i, p in enumerate(portfolios):
        nav = df[p].dropna()
        nav = nav[nav > 0]          # guard against zero/negative NAV values
        rolling_max = nav.cummax()
        # Express drawdown in percentage; clip at 0 to avoid rounding artefacts above the peak
        dd = ((nav - rolling_max) / rolling_max * 100).clip(upper=0)
        color = PALETTE[i % len(PALETTE)]

        max_dd_idx = dd.idxmin()
        max_dd_val = dd.min()
        log.info("Drawdown tab '%s': max_dd=%.2f%% at %s", p, max_dd_val, max_dd_idx)

        # Main drawdown line
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=p,
            legendgroup=p,
            line=dict(color=color, width=1.8),
            mode='lines',
            hovertemplate=f'<b>{p}</b><br>%{{x|%b %Y}}<br>Drawdown: %{{y:.2f}}%<extra></extra>',
        ))

        # Asterisk marker at the maximum drawdown point
        # (shares legendgroup so it hides/shows with the main line)
        fig.add_trace(go.Scatter(
            x=[max_dd_idx], y=[max_dd_val],
            name=p,
            legendgroup=p,
            showlegend=False,
            mode='markers+text',
            marker=dict(color='#FF4E6A', size=14, symbol='asterisk', line=dict(color='#FF4E6A', width=2.5)),
            text=[f"{max_dd_val:.1f}%"],
            textposition='bottom center',
            textfont=dict(color='#FF4E6A', size=9, family='Space Mono'),
            hovertemplate=f'<b>{p} Max DD</b><br>%{{x|%b %Y}}<br>%{{y:.2f}}%<extra></extra>',
        ))

    apply_layout(fig, yaxis_title='Drawdown (%)')
    fig.update_xaxes(tickformat='%b %Y')
    # Thin dotted zero-line for reference (not the same as the series zero)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.1)', line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Max Drawdown Summary Table ────────────────────────────────────
    st.markdown("**Max Drawdown Summary**")
    dd_rows = []
    for p in portfolios:
        nav = df[p].dropna()
        nav = nav[nav > 0]
        rolling_max = nav.cummax()
        dd = ((nav - rolling_max) / rolling_max).clip(upper=0)
        max_dd = dd.min()
        max_dd_date = dd.idxmin()

        # Recovery: first date after the max-drawdown trough where NAV
        # returns to (or exceeds) the pre-trough rolling peak
        peak_before = rolling_max.loc[max_dd_date]
        after = nav.loc[max_dd_date:]
        recovered = after[after >= peak_before]
        recovery = recovered.index[0].strftime('%b %Y') if len(recovered) > 0 else 'Not recovered'

        log.debug("Drawdown summary '%s': max_dd=%.2f%% recovery='%s'", p, max_dd * 100, recovery)
        dd_rows.append({
            'Portfolio': p,
            'Max Drawdown': f"{max_dd*100:.2f}%",
            'Date of Max DD': max_dd_date.strftime('%d %b %Y'),
            'Recovery Date': recovery,
        })

    st.dataframe(pd.DataFrame(dd_rows).set_index('Portfolio'), use_container_width=True)
    log.info("Drawdown tab: complete")


# ══════════════════════════════════════════════
# TAB 5 — ROLLING STD
# ══════════════════════════════════════════════
with tab_rstd:
    log.info("TAB: Rolling Std — rendering rolling volatility chart")
    st.markdown('<div class="section-title">Rolling Volatility (Annualised Std Dev)</div>', unsafe_allow_html=True)

    std_window = st.slider(
        "Rolling Window (months)", min_value=6, max_value=60, value=39, step=1, key='std_window'
    )
    log.debug("Rolling Std tab: window=%d months", std_window)

    fig = go.Figure()
    for i, p in enumerate(portfolios):
        rs = rolling_metric(df[p], std_window, 'std', ann_factor)
        if rs.empty:
            log.warning("Rolling Std tab: no data for '%s' with window=%d — trace skipped", p, std_window)
            continue
        log.debug("Rolling Std tab: '%s' → %d data-points", p, len(rs))
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            name=p,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            mode='lines',
            hovertemplate=f'<b>{p}</b><br>%{{x|%b %Y}}<br>Ann. Std: %{{y:.2f}}%<extra></extra>',
        ))

    apply_layout(fig, yaxis_title='Ann. Std Dev (%)')
    fig.update_xaxes(tickformat='%b %Y')
    st.plotly_chart(fig, use_container_width=True)
    log.info("Rolling Std tab: chart rendered")


# ══════════════════════════════════════════════
# TAB 6 — ROLLING SHARPE
# ══════════════════════════════════════════════
with tab_rsharpe:
    log.info("TAB: Rolling Sharpe — rendering rolling Sharpe chart")
    st.markdown('<div class="section-title">Rolling Sharpe Ratio</div>', unsafe_allow_html=True)

    sharpe_window = st.slider(
        "Rolling Window (months)", min_value=6, max_value=60, value=39, step=1, key='sharpe_window'
    )
    log.debug("Rolling Sharpe tab: window=%d months", sharpe_window)

    fig = go.Figure()
    for i, p in enumerate(portfolios):
        rs = rolling_metric(df[p], sharpe_window, 'sharpe', ann_factor)
        if rs.empty:
            log.warning("Rolling Sharpe tab: no data for '%s' with window=%d — trace skipped", p, sharpe_window)
            continue
        log.debug("Rolling Sharpe tab: '%s' → %d data-points", p, len(rs))
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            name=p,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            mode='lines',
            hovertemplate=f'<b>{p}</b><br>%{{x|%b %Y}}<br>Sharpe: %{{y:.3f}}<extra></extra>',
        ))

    apply_layout(fig, yaxis_title='Sharpe Ratio')
    fig.update_xaxes(tickformat='%b %Y')
    # Dotted zero-line: Sharpe > 0 means the portfolio beat the risk-free rate (0 %)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.1)', line_width=1)
    st.plotly_chart(fig, use_container_width=True)
    log.info("Rolling Sharpe tab: chart rendered")


# ══════════════════════════════════════════════
# TAB 7 — ROLLING CAGR
# ══════════════════════════════════════════════
with tab_rcagr:
    log.info("TAB: Rolling CAGR — detecting optimal window and rendering chart")
    st.markdown('<div class="section-title">Rolling CAGR</div>', unsafe_allow_html=True)

    @st.cache_data
    def find_optimal_cagr_window(file_bytes: bytes, filename: str) -> int:
        """
        Search windows 12–60 months and return the one that maximises the
        average median rolling CAGR across all portfolios.

        Cached on (file_bytes, filename) so it only runs once per uploaded
        file, even as the user adjusts the slider below.

        Algorithm
        ─────────
        For each candidate window width w:
          For each portfolio:
            1. Build the monthly NAV array once.
            2. Vectorise: slice all (start, end) pairs simultaneously with
               NumPy array slicing — avoids a Python loop over windows.
            3. Compute CAGR = (end/start)^(12/w) − 1 for valid pairs.
            4. Take the median CAGR for this portfolio/window combination.
          Average the per-portfolio medians → scalar score for window w.
        Return the window w with the highest score.
        """
        log.info("find_optimal_cagr_window: starting search 12–60 months for file='%s'", filename)
        df_c = load_data(file_bytes, filename)
        af = 252 if detect_frequency(df_c.index) == 'daily' else 12
        portfolios_c = df_c.columns.tolist()

        # Pre-compute monthly NAV arrays once per portfolio to avoid repeated resampling
        monthly_navs = []
        for p in portfolios_c:
            nav = df_c[p].dropna()
            mn = to_monthly_nav(nav) if af == 252 else nav.copy()
            monthly_navs.append(mn.dropna().values)
            log.debug("find_optimal_cagr_window: '%s' monthly_len=%d", p, len(mn.dropna()))

        best_window, best_score = 39, -np.inf
        for w in range(12, 61):
            scores = []
            for nav_arr in monthly_navs:
                n = len(nav_arr)
                if n < w:
                    continue
                # Vectorised window computation: all starting and ending values at once
                starts = nav_arr[:n - w + 1]
                ends = nav_arr[w - 1:]
                valid = starts > 0
                cagrs = np.where(valid, (ends / starts) ** (12 / w) - 1, np.nan) * 100
                if len(cagrs) > 0:
                    scores.append(float(np.nanmedian(cagrs)))
            if scores:
                avg = float(np.mean(scores))
                if avg > best_score:
                    best_score = avg
                    best_window = w

        log.info("find_optimal_cagr_window: optimal=%d months (score=%.2f%%)", best_window, best_score)
        return best_window

    optimal_window = find_optimal_cagr_window(file_bytes, uploaded.name)
    log.info("Rolling CAGR tab: optimal window=%d months", optimal_window)

    st.markdown(f"""
    <div class="info-box">
    🔍 <b>Optimal window detected:</b> <b style="color:#00CFFF">{optimal_window} months</b> — 
    this window produces the highest median rolling CAGR across all portfolios (tested 12–60 months).
    </div>
    """, unsafe_allow_html=True)

    cagr_window = st.slider(
        "Rolling Window (months)",
        min_value=12, max_value=60,
        value=optimal_window,
        step=1, key='cagr_window',
    )
    log.debug("Rolling CAGR tab: user-selected window=%d months", cagr_window)

    fig = go.Figure()
    for i, p in enumerate(portfolios):
        rs = rolling_metric(df[p], cagr_window, 'cagr', ann_factor)
        if rs.empty:
            log.warning("Rolling CAGR tab: no data for '%s' with window=%d — trace skipped", p, cagr_window)
            continue
        log.debug("Rolling CAGR tab: '%s' → %d data-points", p, len(rs))
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            name=p,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.8),
            mode='lines',
            hovertemplate=f'<b>{p}</b><br>%{{x|%b %Y}}<br>CAGR: %{{y:.2f}}%<extra></extra>',
        ))

    apply_layout(fig, yaxis_title='Rolling CAGR (%)')
    fig.update_xaxes(tickformat='%b %Y')
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.1)', line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Median Rolling CAGR Heatmap Table ────────────────────────────
    st.markdown("**Median Rolling CAGR by Window (12–60 months)**")
    log.debug("Rolling CAGR tab: building median CAGR heatmap table")
    heat_data = {}
    for w in range(12, 61):
        row = {}
        for p in portfolios:
            # Use the already-resampled monthly series to stay consistent
            nav_arr = to_monthly_nav(df[p].dropna()).dropna().values if ann_factor == 252 else df[p].dropna().values
            n = len(nav_arr)
            if n >= w:
                starts = nav_arr[:n - w + 1]
                ends = nav_arr[w - 1:]
                valid = starts > 0
                cagrs = np.where(valid, (ends / starts) ** (12 / w) - 1, np.nan) * 100
                row[p] = round(float(np.nanmedian(cagrs)), 2) if np.any(~np.isnan(cagrs)) else np.nan
            else:
                row[p] = np.nan
        heat_data[w] = row

    heat_df = pd.DataFrame(heat_data).T
    heat_df.index.name = 'Window (months)'
    log.info("Rolling CAGR tab: heatmap table shape=%s", heat_df.shape)

    def color_column(col):
        """
        Per-column light-green gradient colouring for the heatmap.
        Higher median CAGR in a column gets a slightly deeper green;
        the lowest value gets the palest shade.  Font is always black
        so it remains legible against the green background.
        """
        mn, mx = col.min(), col.max()
        rng = mx - mn if mx != mn else 1
        styles = []
        for v in col:
            if pd.isna(v):
                styles.append('color: #6B85A8; text-align: center;')
            else:
                intensity = (v - mn) / rng   # 0 = lowest value, 1 = highest value
                g = int(225 + intensity * 15)
                r = int(225 - intensity * 60)
                b = int(220 - intensity * 60)
                styles.append(
                    f'background-color: rgb({r},{g},{b}); '
                    f'color: #000000; text-align: center; font-family: Space Mono; font-weight: 500;'
                )
        return styles

    styled_heat = (
        heat_df
        .style
        .apply(color_column, axis=0)
        .format(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "—")
    )
    st.dataframe(styled_heat, use_container_width=True, height=420)
    log.info("Rolling CAGR tab: complete")


# ══════════════════════════════════════════════
# TAB 8 — YEARLY RETURNS
# ══════════════════════════════════════════════
with tab_yearly:
    log.info("TAB: Yearly Returns — computing year-by-year returns")
    st.markdown('<div class="section-title">Year-wise Returns</div>', unsafe_allow_html=True)

    # ── Year-by-year return computation ──────────────────────────────
    # The first year is typically partial (start date → Dec of that year);
    # the raw start NAV is used as the denominator so the partial-year
    # return is measured from the actual first observation.
    yearly = {}
    for p in portfolios:
        nav = df[p].dropna()
        annual = nav.resample('YE').last()   # month-end last value per calendar year
        year_rets = {}

        for yi, dt in enumerate(annual.index):
            yr = dt.year
            end_nav = annual.iloc[yi]

            if yi == 0:
                # First year: denominator is the very first NAV in the series
                start_nav = nav.iloc[0]
            else:
                # Subsequent years: denominator is the end-of-previous-year NAV
                start_nav = annual.iloc[yi - 1]

            if start_nav > 0:
                year_rets[yr] = (end_nav / start_nav - 1) * 100
                log.debug("Yearly '%s' %d: start=%.2f end=%.2f ret=%.2f%%",
                          p, yr, start_nav, end_nav, year_rets[yr])
            else:
                log.warning("Yearly '%s' %d: non-positive start_nav=%.4f — year skipped", p, yr, start_nav)

        yearly[p] = year_rets

    all_years = sorted(set().union(*[set(r.keys()) for r in yearly.values()]))
    log.info("Yearly tab: years_covered=%s portfolios=%d", all_years, len(portfolios))

    # ── Bar chart ────────────────────────────────────────────────────
    fig = go.Figure()
    for i, p in enumerate(portfolios):
        ret = yearly[p]
        y_vals = [ret.get(yr, None) for yr in all_years]
        fig.add_trace(go.Bar(
            x=[str(y) for y in all_years],
            y=y_vals,
            name=p,
            marker_color=PALETTE[i % len(PALETTE)],
            marker_opacity=0.8,
            offsetgroup=str(i),
            hovertemplate=f'<b>{p}</b><br>Year: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>',
        ))

    apply_layout(fig, yaxis_title='Return (%)')
    fig.update_layout(barmode='group', bargap=0.2)
    # Reference line at zero: positive bars above, negative bars below
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(255,255,255,0.15)', line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Yearly returns styled table ───────────────────────────────────
    table_data = {p: {yr: yearly[p].get(yr, np.nan) for yr in all_years} for p in portfolios}
    yearly_df = pd.DataFrame(table_data, index=all_years)
    yearly_df.index.name = 'Year'

    def color_row(row):
        """
        Per-row light-green gradient colouring for the yearly returns table.
        Within each year (row), the best-returning portfolio gets the deepest
        green; the worst gets the palest shade.  Black font for legibility.
        """
        valid = row.dropna()
        if valid.empty:
            return [''] * len(row)
        mn, mx = valid.min(), valid.max()
        rng = mx - mn if mx != mn else 1
        styles = []
        for v in row:
            if pd.isna(v):
                styles.append('color: #6B85A8; text-align: center;')
            else:
                intensity = (v - mn) / rng
                g = int(225 + intensity * 15)
                r = int(225 - intensity * 60)
                b = int(220 - intensity * 60)
                styles.append(
                    f'background-color: rgb({r},{g},{b}); '
                    f'color: #000000; text-align: center; font-family: Space Mono; font-weight: 500;'
                )
        return styles

    styled_yearly = (
        yearly_df
        .style
        .apply(color_row, axis=1)
        .format(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "—")
    )
    st.dataframe(styled_yearly, use_container_width=True)
    log.info("Yearly Returns tab: complete")


# ══════════════════════════════════════════════
# TAB 9 — DISTRIBUTION (Monthly Returns KDE)
# ══════════════════════════════════════════════
with tab_dist:
    log.info("TAB: Distribution — computing monthly return KDE plots for %d portfolios", len(portfolios))
    st.markdown('<div class="section-title">Monthly Returns Distribution</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">'
        'KDE on independent monthly returns · Dashed = fitted Normal · '
        'Each point = one calendar month · No overlapping windows'
        '</div>',
        unsafe_allow_html=True,
    )

    from scipy.stats import norm as sp_norm

    def get_monthly_returns(nav_series: pd.Series) -> pd.Series:
        """
        Return independent month-end percentage returns.

        For daily input: resample to month-end then pct_change.
        For monthly input: pct_change directly.
        Using month-end sampling ensures each observation covers a distinct
        calendar month, so there is no overlap between windows.
        Returns values as percentages (multiplied by 100).
        """
        monthly = nav_series.resample('ME').last().dropna() if ann_factor == 252 else nav_series.dropna()
        returns = monthly.pct_change().dropna() * 100
        log.debug("get_monthly_returns: '%s' → %d monthly return observations", nav_series.name, len(returns))
        return returns

    ncols = 2
    nrows = (len(portfolios) + ncols - 1) // ncols

    for row in range(nrows):
        cols_dist = st.columns(ncols)
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= len(portfolios):
                break

            p = portfolios[idx]
            color = PALETTE[idx % len(PALETTE)]
            ret = get_monthly_returns(df[p].dropna())

            if len(ret) < 6:
                log.warning("Distribution tab '%s': only %d monthly obs — skipping KDE", p, len(ret))
                with cols_dist[col]:
                    st.warning(f"{p}: not enough monthly observations for distribution")
                continue

            ret_arr = ret.values

            # ── KDE with Silverman's bandwidth rule ──────────────────
            # Silverman's rule is a good default for unimodal, roughly
            # normal data; it may under-smooth for heavy-tailed distributions
            kde = gaussian_kde(ret_arr, bw_method='silverman')
            x_min = ret_arr.mean() - 4 * ret_arr.std()
            x_max = ret_arr.mean() + 4 * ret_arr.std()
            xs = np.linspace(x_min, x_max, 300)
            ys = kde(xs)

            # Fitted normal density: same mean and std as the empirical returns
            # Overlay lets the user visually assess skewness and heavy tails
            normal_ys = sp_norm.pdf(xs, ret_arr.mean(), ret_arr.std())

            # ── Descriptive statistics ────────────────────────────────
            ret_s = pd.Series(ret_arr)
            monthly_mean = float(ret_s.mean())
            monthly_std  = float(ret_s.std())
            skew         = float(ret_s.skew())
            kurt         = float(ret_s.kurtosis())   # excess kurtosis (normal = 0)
            hit_rate     = float((ret_arr > 0).mean() * 100)   # % of positive months
            best_month   = float(ret_arr.max())
            worst_month  = float(ret_arr.min())
            n_months     = len(ret_arr)

            log.info(
                "Distribution tab '%s': n=%d mean=%.2f%% std=%.2f%% skew=%.2f kurt=%.2f hit=%.0f%%",
                p, n_months, monthly_mean, monthly_std, skew, kurt, hit_rate
            )

            # Full-period CAGR for context (not computed from returns here)
            m_full = all_metrics.get(p, {})
            cagr_ref = m_full.get('cagr', np.nan) * 100 if m_full else np.nan
            cagr_str = f"{cagr_ref:+.2f}%" if not np.isnan(cagr_ref) else "—"

            # ── Plotly figure ─────────────────────────────────────────
            fig = go.Figure()

            # Negative-return fill region (red tint)
            mask_neg = xs <= 0
            if mask_neg.any():
                fig.add_trace(go.Scatter(
                    x=xs[mask_neg], y=ys[mask_neg],
                    fill='tozeroy', fillcolor='rgba(255,78,106,0.15)',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False, hoverinfo='skip',
                ))

            # Positive-return fill region (portfolio colour tint)
            mask_pos = xs >= 0
            if mask_pos.any():
                rc, gc, bc = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
                fig.add_trace(go.Scatter(
                    x=xs[mask_pos], y=ys[mask_pos],
                    fill='tozeroy', fillcolor=f'rgba({rc},{gc},{bc},0.12)',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False, hoverinfo='skip',
                ))

            # KDE curve
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                name='KDE',
                line=dict(color=color, width=2.2),
                hovertemplate='Monthly Return: %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>',
            ))

            # Fitted normal overlay — dashed, muted colour
            fig.add_trace(go.Scatter(
                x=xs, y=normal_ys,
                name='Normal',
                line=dict(color='rgba(107,133,168,0.5)', width=1.2, dash='dash'),
                hoverinfo='skip',
            ))

            # Dashed mean line with annotation
            fig.add_vline(
                x=monthly_mean, line_dash='dash', line_color=color, line_width=1.2,
                annotation_text=f"Mean {monthly_mean:+.2f}%",
                annotation_font=dict(color=color, size=9, family='Space Mono'),
                annotation_position='top right',
            )

            # Dotted zero-return reference line
            fig.add_vline(x=0, line_dash='dot', line_color='rgba(255,255,255,0.15)', line_width=1)

            apply_layout(fig, yaxis_title='Density')
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=50, b=40),
                legend=dict(orientation='h', y=1.14, x=1, xanchor='right', font=dict(size=10)),
                title=dict(
                    text=(
                        f'<b style="color:{color}">{p}</b>  '
                        f'<span style="font-size:11px;color:#8BAAC8">'
                        f'n={n_months}mo · Mean {monthly_mean:+.2f}% · Std {monthly_std:.2f}% · '
                        f'Skew {skew:.2f} · Kurt {kurt:.2f} · Hit {hit_rate:.0f}%</span>'
                    ),
                    font=dict(family='Syne', size=13, color='#FFFFFF'), x=0,
                ),
            )
            fig.update_xaxes(ticksuffix='%', tickformat='.1f', title_text='Monthly Return (%)')

            with cols_dist[col]:
                st.plotly_chart(fig, use_container_width=True)
                log.debug("Distribution tab '%s': KDE chart rendered", p)

    log.info("Distribution tab: all charts rendered — session render complete")