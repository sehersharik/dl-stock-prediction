import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import sys

# ── Project Root Setup ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Streamlit only reloads modules under this script's folder, so after a redeploy the
# server can keep stale copies of src.* from before the update (e.g. an old
# src.features.generator without a newly added function). Re-import them on every run.
for _name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
    del sys.modules[_name]

from src.utils.config import load_config
from src.evaluation.metrics_registry import backtest_headline, chance_statement, confidence_statement
from src.inference.engine import InferenceEngine
from src.data.live_provider import YFinanceLiveProvider

# ── Streamlit Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense — AI Stock Direction & Risk Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom Professional Fintech CSS ──────────────────────────────────────────
st.markdown("""
<style>
  /* Base typography & page width */
  .block-container {
    padding: 1.8rem 2.5rem 3rem 2.5rem;
    max-width: 1200px;
  }
  body, p, div, label {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #0F172A;
    line-height: 1.5;
  }

  /* Header & Navigation styling */
  .navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1rem;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.5rem;
  }
  .brand-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: #0F172A;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .brand-subtitle {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 400;
    margin-top: 0.1rem;
  }

  /* Stock header card */
  .stock-header-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
  }
  .stock-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
  }
  .stock-ticker-badge {
    display: inline-block;
    background-color: #F1F5F9;
    color: #334155;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-family: monospace;
    margin-left: 0.5rem;
  }
  .data-status-badge {
    font-size: 0.78rem;
    color: #64748B;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }

  /* Main Prediction Cards */
  .prediction-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
  }
  .card-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 0.5rem;
  }

  /* Direction Badges */
  .badge-up {
    display: inline-flex;
    align-items: center;
    background-color: #F0FDF4;
    color: #16A34A;
    border: 1px solid #BBF7D0;
    font-size: 2.2rem;
    font-weight: 800;
    padding: 0.2rem 1.2rem;
    border-radius: 8px;
    letter-spacing: 0.02em;
  }
  .badge-down {
    display: inline-flex;
    align-items: center;
    background-color: #FEF2F2;
    color: #DC2626;
    border: 1px solid #FECACA;
    font-size: 2.2rem;
    font-weight: 800;
    padding: 0.2rem 1.2rem;
    border-radius: 8px;
    letter-spacing: 0.02em;
  }

  /* Stat numbers */
  .prob-value-large {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0F172A;
  }
  .prob-subtext {
    font-size: 0.85rem;
    color: #64748B;
  }

  /* Explanation Banner */
  .explanation-banner {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #2563EB;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin: 1.25rem 0 1.5rem 0;
    font-size: 0.95rem;
    color: #334155;
    line-height: 1.6;
  }

  /* Feature importance list item */
  .feature-item {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .feature-name {
    font-weight: 600;
    font-size: 0.92rem;
    color: #0F172A;
  }
  .feature-desc {
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 0.15rem;
  }
  .feature-impact {
    font-size: 0.82rem;
    font-weight: 600;
    color: #2563EB;
    background-color: #EFF6FF;
    padding: 0.2rem 0.55rem;
    border-radius: 4px;
    white-space: nowrap;
  }

  /* Simple stat tile */
  .stat-tile {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    height: 100%;
  }
  .stat-tile-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 0.2rem;
  }
  .stat-tile-sub {
    font-size: 0.8rem;
    color: #64748B;
    margin-top: 0.2rem;
  }

  /* Section Headings */
  .section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 1.75rem;
    margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
  }
  .section-subheading {
    font-size: 0.88rem;
    color: #64748B;
    margin-bottom: 1.1rem;
  }

  /* Disclaimer Footnote */
  .disclaimer-box {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: #64748B;
    margin-top: 2.5rem;
    line-height: 1.6;
  }

  /* Tab override styling */
  .stTabs [data-baseweb="tab-list"] {
    gap: 1.5rem;
    border-bottom: 1px solid #E2E8F0;
  }
  .stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
    color: #64748B;
    padding: 0.6rem 0.2rem;
  }
  .stTabs [aria-selected="true"] {
    color: #2563EB !important;
    border-bottom-color: #2563EB !important;
  }
</style>
""", unsafe_allow_html=True)

TICKER = "RELIANCE.NS"
MODEL = "mlp"

# ── Session State Initializer ─────────────────────────────────────────────────
if 'live_result' not in st.session_state:
    st.session_state.live_result = None

# ── Backend Data Loading Function (No modifications to metrics/pipeline) ─────
@st.cache_data
def load_backend_data():
    data = {}
    
    # 1. Processed Features & Inputs
    fp = PROJECT_ROOT / f"data/processed/{TICKER}_features.csv"
    if fp.exists():
        data['features'] = pd.read_csv(fp, index_col=0, parse_dates=True)

    # 2. Raw Price Files
    raw_rel = PROJECT_ROOT / f"data/raw/{TICKER}_10y.csv"
    if raw_rel.exists():
        data['raw_prices'] = pd.read_csv(raw_rel, index_col=0, parse_dates=True).sort_index()

    raw_nifty = PROJECT_ROOT / "data/raw/NSEI_10y.csv"
    if raw_nifty.exists():
        data['raw_nifty'] = pd.read_csv(raw_nifty, index_col=0, parse_dates=True).sort_index()

    raw_vix = PROJECT_ROOT / "data/raw/INDIAVIX_10y.csv"
    if raw_vix.exists():
        data['raw_vix'] = pd.read_csv(raw_vix, index_col=0, parse_dates=True).sort_index()

    # 3. Reported results: the active run in reports/metrics.json is the single source
    #    for every metric on this dashboard (built by `python main.py metrics`).
    with open(PROJECT_ROOT / "reports/metrics.json") as f:
        metrics_doc = json.load(f)
    run = {"run_id": metrics_doc["active_run"], **metrics_doc["runs"][metrics_doc["active_run"]]}
    data['run'] = run
    data['model_metrics'] = run['metrics']['mlp']['test']
    data['backtest'] = run.get('backtest')

    artifacts = run['artifacts']
    def _artifact(key, reader):
        path = PROJECT_ROOT / artifacts[key] if key in artifacts else None
        return reader(path) if path is not None and path.exists() else None

    data['test_preds'] = _artifact('test_preds', lambda p: pd.read_csv(p, index_col=0, parse_dates=True))
    data['explainability'] = _artifact('perm_importance', pd.read_csv)

    return data

backend_data = load_backend_data()

# ── TOP NAVIGATION HEADER ────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
  <div>
    <div class="brand-name">📈 StockSense</div>
    <div class="brand-subtitle">AI-Powered Stock Direction & Risk Analysis</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Define Primary Product Navigation Tabs
tab_overview, tab_analysis, tab_methodology, tab_about = st.tabs([
    "Overview", 
    "Analysis", 
    "Methodology", 
    "About"
])

# Helper for plain-English feature translation
FEATURE_TRANSLATIONS = {
    "Dist_SMA200": ("Distance from 200-day average", "How far the stock price is from its 200-day trend average"),
    "NIFTY_Return_1d": ("NIFTY 1-day market return", "How the overall Indian stock market (NIFTY 50) moved today"),
    "VIX_Change_1d": ("India VIX volatility change", "Change in overall market volatility and investor fear index"),
    "RSI_14": ("14-day RSI momentum", "Momentum oscillator measuring buying vs selling pressure"),
    "ROC_12": ("12-day price velocity", "Rate of price momentum over the past two trading weeks"),
    "NIFTY_Return_5d": ("NIFTY 5-day return", "Cumulative 5-day movement of the broader market index"),
    "Vol_Change": ("Volume change", "Daily percentage change in trading volume"),
    "Return_1d": ("1-day stock return", "Asset return over the previous trading session"),
    "MACD_Ratio": ("MACD trend momentum", "Normalized moving average convergence/divergence"),
    "CO_Spread": ("Daily Close-Open spread", "Intraday price movement direction spread"),
    "BB_Width": ("Bollinger bandwidth", "Bandwidth ratio showing price volatility squeeze"),
    "Rolling_Vol_20": ("20-day rolling volatility", "Short-term annualized price variance"),
    "NIFTY_Vol_20": ("20-day NIFTY volatility", "Broader market volatility trend"),
    "ATR_Ratio": ("Normalized True Range", "Daily trading range relative to price level"),
    "Dist_SMA50": ("Distance from 50-day average", "Medium-term trend position relative to 50-day average"),
    "Return_10d": ("10-day stock return", "Medium-term 2-week price return"),
    "Return_20d": ("20-day stock return", "Monthly price return"),
    "SMA20_50_Ratio": ("Short/Medium trend ratio", "20-day SMA relative to 50-day SMA"),
    "HL_Spread": ("Daily High-Low spread", "Intraday trading range relative to Close"),
    "VIX_Level": ("India VIX level", "Absolute market fear/volatility index reading"),
    "Dist_SMA20": ("Distance from 20-day average", "Short-term trend position relative to 20-day average"),
    "Return_5d": ("5-day stock return", "Weekly price momentum return"),
    "NIFTY_Dist_SMA50": ("NIFTY distance from SMA50", "Broader market medium-term trend position"),
    "SMA50_200_Ratio": ("Golden/Death cross ratio", "50-day SMA relative to 200-day SMA"),
    "Vol_Ratio_20": ("Volume relative to 20-day avg", "Current volume compared to 20-day average volume"),
    "BB_PctB": ("Bollinger %B position", "Price position within Bollinger Bands (0=Lower, 1=Upper)"),
    "MACD_Hist_Ratio": ("MACD histogram ratio", "Momentum acceleration indicator")
}


# ==============================================================================
# TAB 1: OVERVIEW PAGE (Primary Customer Experience)
# ==============================================================================
with tab_overview:
    
    # ── DISCLAIMER & BACKTEST HEADLINE (plain statement, generated from metrics.json)
    st.markdown("""
    <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 0.6rem 1.1rem; margin-bottom: 0.6rem; font-size: 0.9rem; color: #78350F;">
      <strong>Research prototype, not investment advice.</strong>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-left: 4px solid #DC2626; border-radius: 6px; padding: 0.8rem 1.1rem; margin-bottom: 1rem; font-size: 0.9rem; color: #7F1D1D;">
      <strong>Backtest result:</strong> {backtest_headline(backend_data['run'])}
      Model probabilities below are research outputs, not trading signals.
    </div>
    """, unsafe_allow_html=True)

    # ── STOCK SELECTOR & HEADER ────────────────────────────────────────────────
    col_sel, col_info = st.columns([1, 3])
    
    with col_sel:
        selected_stock = st.selectbox(
            "Stock",
            ["RELIANCE.NS"],
            index=0,
            help="Supported Indian stock tickers on NSE"
        )
    
    # Extract price & market data fallback
    raw_df = backend_data.get('raw_prices', pd.DataFrame())
    if len(raw_df) < 2:
        st.error("Historical price data (data/raw) is missing; the dashboard cannot be displayed.")
        st.stop()
    latest_raw_date = raw_df.index[-1]
    latest_raw_close = raw_df['Close'].iloc[-1]
    prev_raw_close = raw_df['Close'].iloc[-2]
    prev_raw_date = raw_df.index[-2]
    daily_change_pct = ((latest_raw_close / prev_raw_close) - 1) * 100 if prev_raw_close else 0.0
    
    # Check if user triggered live refresh
    live_res = st.session_state.live_result
    
    with col_info:
        st.markdown(f"""
        <div class="stock-header-card">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
            <div>
              <div class="stock-title">
                RELIANCE INDUSTRIES LTD.
                <span class="stock-ticker-badge">NSE: RELIANCE</span>
              </div>
              <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">
                Energy, Telecom & Retail Conglomerate · NIFTY 50 Component
              </div>
            </div>
            <div class="data-status-badge">
              <span>📅 Historical data through: <strong>{latest_raw_date.strftime('%d %b %Y')}</strong></span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Refresh Market Data Button
    col_ref_btn, col_ref_msg = st.columns([1, 4])
    with col_ref_btn:
        if st.button("🔄 Update Live Market Tick", width='stretch'):
            with st.spinner("Polling live market data and executing model inference..."):
                try:
                    config = load_config(str(PROJECT_ROOT / "config" / "settings.yaml"))
                    provider = YFinanceLiveProvider()
                    engine = InferenceEngine(config, provider, PROJECT_ROOT)
                    st.session_state.live_result = engine.run_live_inference(TICKER, MODEL)
                    live_res = st.session_state.live_result
                except Exception as e:
                    st.session_state.live_result = {"status": "Error", "message": str(e)}
                    live_res = st.session_state.live_result

    with col_ref_msg:
        if live_res and live_res.get("status") == "Error":
            st.warning(f"Live market provider status: {live_res.get('message')}. Displaying verified model results.")

    # Determine Active Prediction Data (Live or Out-of-Sample Test Prediction Fallback)
    test_preds = backend_data.get('test_preds')
    if test_preds is None or test_preds.empty:
        test_preds = pd.DataFrame()
    
    if live_res and live_res.get("status") == "Success":
        pred_direction = live_res.get('predicted_direction', 'UP').split('/')[0]
        prob_up = live_res['direction_prob']
        prob_down = 1.0 - prob_up
        display_price = live_res.get('price', latest_raw_close)
        display_timestamp = str(live_res.get('feature_date', live_res.get('timestamp')))[:10]
        data_source_label = "Live model run on completed session"
    else:
        # Out-of-sample latest test prediction
        if test_preds.empty:
            st.error("Saved test predictions for the active run are missing; no prediction can be shown.")
            st.stop()
        latest_pred_row = test_preds.iloc[-1]
        prob_up = float(latest_pred_row['Prob'])
        prob_down = 1.0 - prob_up
        pred_direction = "UP" if prob_up >= 0.50 else "DOWN"
        display_price = latest_raw_close
        display_timestamp = test_preds.index[-1].strftime('%Y-%m-%d')
        data_source_label = "Verified historical model snapshot"

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # ── MAIN PREDICTION & PRICE SECTION ────────────────────────────────────────
    col_pred, col_price = st.columns([7, 5])
    
    with col_pred:
        badge_html = f'<div class="badge-up">▲ UP</div>' if pred_direction == "UP" else f'<div class="badge-down">▼ DOWN</div>'
        
        st.markdown(f"""
        <div class="prediction-card">
          <div class="card-label">NEXT-DAY OUTLOOK</div>
          <div style="display: flex; align-items: center; gap: 1.5rem; margin: 0.75rem 0 1.25rem 0;">
            {badge_html}
            <div>
              <div style="font-size: 0.85rem; color: #64748B; font-weight: 500;">Model Estimate</div>
              <div style="font-size: 0.88rem; color: #334155;">Next Trading Session Target</div>
            </div>
          </div>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; background-color: #F8FAFC; padding: 1rem; border-radius: 8px; border: 1px solid #E2E8F0;">
            <div>
              <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Probability of Upward Movement</div>
              <div class="prob-value-large" style="color: #16A34A;">{prob_up:.1%}</div>
            </div>
            <div>
              <div style="font-size: 0.78rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Probability of Downward Movement</div>
              <div class="prob-value-large" style="color: #DC2626;">{prob_down:.1%}</div>
            </div>
          </div>
          
          <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.75rem; font-style: italic;">
            Based on recent price trends, technical indicators and broader market conditions ({data_source_label} - {display_timestamp}).
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_price:
        # Price, change and date must come from the same source: the live tick (vs the last
        # completed session before it) when a live run succeeded, otherwise the stored history.
        if live_res and live_res.get("status") == "Success" and live_res.get("price_change") is not None:
            change_pct = live_res["price_change"] * 100
            change_label = f"vs close of {pd.Timestamp(live_res['reference_date']).strftime('%d %b %Y')}"
            price_time_label = f"Live quote: <strong>{pd.Timestamp(live_res['timestamp']).strftime('%d %b %Y %H:%M')} IST</strong> (delayed)"
        else:
            change_pct = daily_change_pct
            change_label = f"vs close of {prev_raw_date.strftime('%d %b %Y')}"
            price_time_label = f"Close of NSE session: <strong>{latest_raw_date.strftime('%d %b %Y')}</strong> (stored history)"
        change_color = "#16A34A" if change_pct >= 0 else "#DC2626"
        change_icon = "+" if change_pct >= 0 else ""
        
        st.markdown(f"""
        <div class="prediction-card">
          <div class="card-label">LATEST STOCK PRICE</div>
          <div style="font-size: 2.2rem; font-weight: 800; color: #0F172A; margin: 0.2rem 0;">
            ₹{display_price:,.2f}
          </div>
          <div style="font-size: 0.92rem; font-weight: 600; color: {change_color}; display: flex; align-items: center; gap: 0.4rem; margin-bottom: 1rem;">
            <span>{change_icon}{change_pct:.2f}%</span>
            <span style="font-size: 0.8rem; color: #64748B; font-weight: 400;">({change_label})</span>
          </div>
          
          <div style="padding-top: 0.75rem; border-top: 1px solid #F1F5F9; font-size: 0.82rem; color: #64748B; line-height: 1.6;">
            <div>{price_time_label}</div>
            <div>Historical data: <strong>{raw_df.index[0].year} – {latest_raw_date.year}</strong></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SIMPLE "WHAT DOES THIS MEAN?" SECTION ─────────────────────────────────
    explanation_text = (
        f"The model currently estimates a higher probability (<strong>{prob_up:.1%}</strong>) of an <strong>upward move</strong> "
        f"on the next trading day. This is a statistical model output based on 27 stationary market signals, not a guarantee of future performance."
        if pred_direction == "UP" else
        f"The model currently estimates a higher probability (<strong>{prob_down:.1%}</strong>) of a <strong>downward or flat move</strong> "
        f"on the next trading day. This indicates higher statistical downside risk based on 27 stationary market signals, not a guarantee of future performance."
    )
    
    st.markdown(f"""
    <div class="explanation-banner">
      <div style="font-weight: 700; color: #1E40AF; margin-bottom: 0.25rem;">What does this mean?</div>
      <div>{explanation_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── WHY THE MODEL THINKS THIS ("Why this prediction?") ────────────────────
    st.markdown('<div class="section-heading">Why this prediction?</div>', unsafe_allow_html=True)
    perm_split = backend_data['run']['artifacts'].get('perm_importance_split', 'out-of-sample')
    st.markdown(f'<div class="section-subheading">Key indicators the model relies on, ranked by permutation importance (drop in ROC-AUC when a feature is shuffled) on the {perm_split} period.</div>', unsafe_allow_html=True)
    
    exp_data = backend_data.get('explainability')
    
    if exp_data is not None and not exp_data.empty:
        top_features = exp_data.head(5)
        col_f1, col_f2 = st.columns(2)
        
        for idx, (_, row) in enumerate(top_features.iterrows()):
            f_code = row['Feature']
            f_imp = row['Importance_Mean']
            title, desc = FEATURE_TRANSLATIONS.get(f_code, (f_code, "Technical indicator input"))
            
            card_html = f"""
            <div class="feature-item">
              <div>
                <div class="feature-name">{idx+1}. {title}</div>
                <div class="feature-desc">{desc}</div>
              </div>
              <div class="feature-impact">+{f_imp:.4f} impact</div>
            </div>
            """
            
            if idx % 2 == 0:
                with col_f1:
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                with col_f2:
                    st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info(f"Feature importance has not been computed for run {backend_data['run']['run_id']} yet.")

    st.markdown("""
    <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.4rem; font-style: italic;">
      Note: Feature importance shows which inputs influenced the model's decision boundaries most during empirical testing. It does not imply direct market causation.
    </div>
    """, unsafe_allow_html=True)

    # ── PRICE CHART ("Price & Market Trend") ──────────────────────────────────
    st.markdown('<div class="section-heading">Price & Market Trend</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Historical price action overlaid with key trend moving averages.</div>', unsafe_allow_html=True)

    if not raw_df.empty:
        # Time range selector
        range_col, _ = st.columns([2, 4])
        with range_col:
            chart_range = st.radio(
                "Time Horizon",
                ["3M", "6M", "1Y", "5Y", "ALL"],
                index=2,
                horizontal=True,
                key="overview_chart_range"
            )
        
        # Filter raw price data by range
        if chart_range == "3M":
            chart_df = raw_df.tail(63)
        elif chart_range == "6M":
            chart_df = raw_df.tail(126)
        elif chart_range == "1Y":
            chart_df = raw_df.tail(252)
        elif chart_range == "5Y":
            chart_df = raw_df.tail(1260)
        else:
            chart_df = raw_df

        # Compute moving averages for clean display
        chart_df = chart_df.copy()
        chart_df['SMA_50'] = chart_df['Close'].rolling(50).mean()
        chart_df['SMA_200'] = chart_df['Close'].rolling(200).mean()

        fig_chart = go.Figure()
        
        # Candlestick or clean line
        fig_chart.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['Close'],
            name='RELIANCE Close Price',
            line=dict(color='#2563EB', width=2)
        ))
        
        fig_chart.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['SMA_50'],
            name='50-Day Moving Average',
            line=dict(color='#F59E0B', width=1.5, dash='solid')
        ))
        
        fig_chart.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df['SMA_200'],
            name='200-Day Moving Average',
            line=dict(color='#64748B', width=1.5, dash='dash')
        ))

        fig_chart.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            template="plotly_white",
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickprefix="₹"),
            legend=dict(orientation="h", y=1.08, x=0.01)
        )
        
        st.plotly_chart(fig_chart, width='stretch')

    # ── MARKET CONTEXT ────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Market Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Broader Indian market indicators evaluated alongside stock features.</div>', unsafe_allow_html=True)
    
    raw_nifty = backend_data.get('raw_nifty', pd.DataFrame())
    raw_vix = backend_data.get('raw_vix', pd.DataFrame())

    if len(raw_nifty) < 2 or len(raw_vix) < 2:
        st.error("NIFTY 50 / India VIX history (data/raw) is missing; market context cannot be displayed.")
        st.stop()
    nifty_close = raw_nifty['Close'].iloc[-1]
    nifty_prev = raw_nifty['Close'].iloc[-2]
    nifty_chg = ((nifty_close / nifty_prev) - 1) * 100 if nifty_prev else 0.0

    vix_close = raw_vix['Close'].iloc[-1]
    vix_prev = raw_vix['Close'].iloc[-2]
    vix_chg = ((vix_close / vix_prev) - 1) * 100 if vix_prev else 0.0

    col_ctx1, col_ctx2 = st.columns(2)
    
    with col_ctx1:
        nifty_chg_icon = "+" if nifty_chg >= 0 else ""
        st.markdown(f"""
        <div class="stat-tile">
          <div class="card-label">NIFTY 50 INDEX (^NSEI)</div>
          <div class="stat-tile-val">{nifty_close:,.2f}</div>
          <div style="font-size: 0.88rem; font-weight: 600; color: {'#16A34A' if nifty_chg>=0 else '#DC2626'}; margin-top: 0.2rem;">
            {nifty_chg_icon}{nifty_chg:.2f}% (Daily return)
          </div>
          <div class="stat-tile-sub">Overall Indian stock market benchmark movement.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_ctx2:
        vix_chg_icon = "+" if vix_chg >= 0 else ""
        st.markdown(f"""
        <div class="stat-tile">
          <div class="card-label">INDIA VIX VOLATILITY (^INDIAVIX)</div>
          <div class="stat-tile-val">{vix_close:.2f}</div>
          <div style="font-size: 0.88rem; font-weight: 600; color: {'#DC2626' if vix_chg>=0 else '#16A34A'}; margin-top: 0.2rem;">
            {vix_chg_icon}{vix_chg:.2f}% (Daily change)
          </div>
          <div class="stat-tile-sub">Market volatility and investor fear indicator.</div>
        </div>
        """, unsafe_allow_html=True)

    # ── MODEL RELIABILITY ("How reliable is the model?") ──────────────────────
    st.markdown('<div class="section-heading">How reliable is the model?</div>', unsafe_allow_html=True)
    run = backend_data['run']
    test_split = run['split']['test']
    test_years = f"{test_split['start'][:4]}–{test_split['end'][:4]}"
    if run['feature_selection']['split'] == 'test':
        selection_note = "Note: the feature set was chosen by comparing scores on this same period, so these figures are likely somewhat optimistic."
    else:
        selection_note = f"The feature set was chosen on the {run['feature_selection']['split']} period; this test period was evaluated once."
    st.markdown(f'<div class="section-subheading">Out-of-sample evaluation on {test_split["n"]} held-out trading days ({test_years}) that were not used to train the model. {selection_note}</div>', unsafe_allow_html=True)
    
    model_metrics = backend_data['model_metrics']
    
    roc_auc_val = model_metrics['ROC-AUC']
    bal_acc_val = model_metrics['Balanced Accuracy']
    prec_val = model_metrics['Precision']
    f1_val = model_metrics['F1']

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="card-label">ROC-AUC SCORE</div>
          <div class="stat-tile-val">{roc_auc_val:.4f}</div>
          <div class="stat-tile-sub">0.50 = Random guessing · 1.00 = Perfect separation</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="card-label">BALANCED ACCURACY</div>
          <div class="stat-tile-val">{bal_acc_val:.2%}</div>
          <div class="stat-tile-sub">Overall direction accuracy accounting for class ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="card-label">PRECISION (UP PREDICTIONS)</div>
          <div class="stat-tile-val">{prec_val:.2%}</div>
          <div class="stat-tile-sub">Accuracy when model explicitly predicts UP</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        st.markdown(f"""
        <div class="stat-tile">
          <div class="card-label">F1 SCORE</div>
          <div class="stat-tile-val">{f1_val:.4f}</div>
          <div class="stat-tile-sub">Harmonic balance of precision and recall coverage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 1rem; font-size: 0.88rem; color: #475569;">
      <strong>Plain-English Verdict:</strong> Test ROC-AUC is {roc_auc_val:.4f} (0.5 = random guessing). {chance_statement(backend_data['run'])} {confidence_statement(backend_data['run'])} Performance can vary across market conditions.
    </div>
    """, unsafe_allow_html=True)

    # ── HISTORICAL BACKTEST ───────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Historical Strategy Performance</div>', unsafe_allow_html=True)
    bt = backend_data.get('backtest')
    if not bt:
        st.info(f"The historical backtest for run {run['run_id']} has not been produced yet, so no strategy results are shown.")
    else:
        head_ex = bt['headline_execution']
        bt_metrics = bt['variants'][head_ex]
        curve_path = PROJECT_ROOT / bt['curves'][head_ex]
        bt_curve = pd.read_csv(curve_path, index_col=0, parse_dates=True) if curve_path.exists() else None
        st.markdown(
            f'<div class="section-subheading">Long/cash simulation on held-out test data ({test_years}). '
            f'Signal at the close; <strong>{bt_metrics["Execution_Rule"]}</strong>. '
            f'Cost {bt["cost_per_side"]:.2%} per side (every entry and exit). Sharpe uses a '
            f'{bt["risk_free_rate_annual"]:.1%} risk-free rate (assumption: RBI 91-day T-bill level). '
            f'Buy &amp; hold is measured over exactly the same {bt_metrics["Days_Tested"]} periods.</div>',
            unsafe_allow_html=True)

        strat_ret = bt_metrics['Strategy_Total_Return']
        bnh_ret = bt_metrics['BnH_Total_Return']
        ret_color = "#16A34A" if strat_ret >= 0 else "#DC2626"

        col_b1, col_b2, col_b3, col_b4, col_b5, col_b6 = st.columns(6)

        with col_b1:
            st.markdown(f'<div class="stat-tile"><div class="card-label">STRATEGY RETURN</div><div class="stat-tile-val" style="color:{ret_color};">{strat_ret:+.2%}</div><div class="stat-tile-sub">After costs</div></div>', unsafe_allow_html=True)
        with col_b2:
            st.markdown(f'<div class="stat-tile"><div class="card-label">BUY & HOLD</div><div class="stat-tile-val">{bnh_ret:+.2%}</div><div class="stat-tile-sub">Same periods, after costs</div></div>', unsafe_allow_html=True)
        with col_b3:
            st.markdown(f'<div class="stat-tile"><div class="card-label">SHARPE RATIO</div><div class="stat-tile-val">{bt_metrics["Strategy_Sharpe"]:.2f}</div><div class="stat-tile-sub">Buy &amp; hold: {bt_metrics["BnH_Sharpe"]:.2f}</div></div>', unsafe_allow_html=True)
        with col_b4:
            st.markdown(f'<div class="stat-tile"><div class="card-label">MAX DRAWDOWN</div><div class="stat-tile-val" style="color:#DC2626;">{bt_metrics["Strategy_Max_Drawdown"]:.2%}</div><div class="stat-tile-sub">Buy &amp; hold: {bt_metrics["BnH_Max_Drawdown"]:.2%}</div></div>', unsafe_allow_html=True)
        with col_b5:
            st.markdown(f'<div class="stat-tile"><div class="card-label">WIN RATE</div><div class="stat-tile-val">{bt_metrics["Win_Rate_Trade"]:.1%}</div><div class="stat-tile-sub">Per trade · {bt_metrics["Win_Rate_Day"]:.1%} per day</div></div>', unsafe_allow_html=True)
        with col_b6:
            st.markdown(f'<div class="stat-tile"><div class="card-label">TRADES</div><div class="stat-tile-val">{bt_metrics["Trades"]}</div><div class="stat-tile-sub">{bt_metrics["Position_Changes"]} sides · {bt_metrics["Annual_Turnover"]:.0f}× turnover/yr</div></div>', unsafe_allow_html=True)

        if bt_curve is not None and not bt_curve.empty:
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt_curve.index, y=bt_curve['Strategy_Equity'],
                name=f'Model Strategy ({strat_ret:+.2%})', line=dict(color='#2563EB', width=2.5)
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_curve.index, y=bt_curve['BnH_Equity'],
                name=f'Buy & Hold, same periods ({bnh_ret:+.2%})', line=dict(color='#94A3B8', width=1.5, dash='dash')
            ))
            fig_bt.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_white",
                hovermode="x unified",
                yaxis=dict(tickprefix="₹", showgrid=True, gridcolor="#F1F5F9"),
                legend=dict(orientation="h", y=1.06, x=0.01)
            )
            st.plotly_chart(fig_bt, width='stretch')

        beats = strat_ret > bnh_ret
        st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 0.9rem 1.1rem; margin-top: 0.5rem; font-size: 0.88rem; color: #475569;">
          <strong>Plain-English Verdict:</strong> After costs the strategy {"beat" if beats else "did <strong>worse</strong> than"} buy &amp; hold over the same periods
          ({strat_ret:+.2%} vs {bnh_ret:+.2%}). It made {bt_metrics["Trades"]} round trips, paying {bt_metrics["Total_Cost_Paid"]:.1%} of capital in costs.
          Reliance's price moved {bt["window_price_return"]:+.2%} over the whole window; the difference from buy &amp; hold comes from
          {len(bt["sessions_without_prediction"])} sessions with no prediction (gaps in the input data), which both strategies sit out.
          This is one historical period and does not predict future returns.
        </div>
        """, unsafe_allow_html=True)

        # ── EXECUTION VARIANTS & COST SENSITIVITY ─────────────────────────────
        st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-top: 1rem;">Execution Assumption</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Execution": m["Execution_Rule"] + (" (headline)" if ex == head_ex else ""),
             "Strategy Return": f"{m['Strategy_Total_Return']:+.2%}", "Buy & Hold": f"{m['BnH_Total_Return']:+.2%}",
             "Sharpe": f"{m['Strategy_Sharpe']:.2f}", "Max Drawdown": f"{m['Strategy_Max_Drawdown']:.2%}",
             "Win Rate / Trade": f"{m['Win_Rate_Trade']:.1%}", "Trades": m["Trades"]}
            for ex, m in bt['variants'].items()
        ]), width='stretch', hide_index=True)

        st.markdown('<div style="font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-top: 1rem;">Transaction Cost Sensitivity</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size: 0.83rem; color: #64748B; margin-bottom: 0.6rem;">Headline execution, {len(bt["cost_grid"])} cost levels charged per side:</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Cost per Side": f"{g['cost_per_side']:.2%}" + (" (default)" if i == bt['default_index'] else ""),
             "Strategy Return": f"{g['Strategy_Total_Return']:+.2%}", "Buy & Hold": f"{g['BnH_Total_Return']:+.2%}",
             "Sharpe": f"{g['Strategy_Sharpe']:.2f}", "Max Drawdown": f"{g['Strategy_Max_Drawdown']:.2%}",
             "Win Rate / Trade": f"{g['Win_Rate_Trade']:.1%}", "Win Rate / Day": f"{g['Win_Rate_Day']:.1%}"}
            for i, g in enumerate(bt['cost_grid'])
        ]), width='stretch', hide_index=True)


# ==============================================================================
# TAB 2: ANALYSIS PAGE (In-depth Metrics & Ablation)
# ==============================================================================
with tab_analysis:
    st.markdown('<div class="section-heading">Detailed Experimental Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">In-depth feature ablation, out-of-sample confusion matrix, and performance curves.</div>', unsafe_allow_html=True)
    
    # Feature Ablation Progression
    st.markdown('### 1. Feature Ablation Progression (F0 → F5)')
    st.markdown("""
    To investigate which groups of indicators genuinely contribute predictive capacity, features were introduced cumulatively across 6 levels:
    - **F0 (Base):** 1d/5d/10d/20d Returns + High-Low & Close-Open Spreads (6 features)
    - **F1 (Trend):** F0 + Moving Average Distance Ratios (`Dist_SMA20/50/200`) (11 features)
    - **F2 (Momentum):** F1 + RSI, MACD Ratio, Rate of Change (15 features)
    - **F3 (Volatility):** F2 + ATR Ratio, Rolling Volatility, Bollinger %B & Width (19 features)
    - **F4 (Volume):** F3 + Volume Change & Volume Ratios (21 features)
    - **F5 (Market Context):** F4 + NIFTY 50 Returns/Volatility + India VIX Level/Change (27 features)
    """)
    
    selection = backend_data['run']['feature_selection']
    split_label = "Test" if selection['split'] == 'test' else "Validation"
    if selection.get('levels'):
        levels = list(selection['levels'])
        mlp_roc = [selection['levels'][lvl]['ROC-AUC'] for lvl in levels]
        lstm_levels = selection.get('lstm_levels', {})
        
        fig_abl = go.Figure()
        fig_abl.add_trace(go.Scatter(x=levels, y=mlp_roc, mode='lines+markers', name='Small MLP (ROC-AUC)', line=dict(color='#2563EB', width=2.5)))
        if lstm_levels:
            lstm_roc = [lstm_levels.get(lvl, {}).get('ROC-AUC', np.nan) for lvl in levels]
            fig_abl.add_trace(go.Scatter(x=levels, y=lstm_roc, mode='lines+markers', name='LSTM (ROC-AUC)', line=dict(color='#94A3B8', width=2, dash='dot')))
        fig_abl.add_hline(y=0.50, line_dash='dash', line_color='#E2E8F0', annotation_text='Random Baseline (0.50)')
        
        fig_abl.update_layout(
            height=360,
            template="plotly_white",
            title="Model ROC-AUC Progression across Feature Sets",
            xaxis_title="Feature Level",
            yaxis_title=f"{split_label} ROC-AUC Score"
        )
        st.plotly_chart(fig_abl, width='stretch')

        best_single = max(mlp_roc[:-1])
        if selection['split'] == 'test':
            selection_caveat = "These comparisons were made on the test period, which was then used to choose the final feature set."
        else:
            selection_caveat = f"These comparisons use the {selection['split']} period only; the test period played no part in choosing the feature set."
        st.markdown(f"""
        <div class="explanation-banner">
          <strong>Key Research Takeaway:</strong> Single-stock indicators alone ({levels[0]}–{levels[-2]}) reached a best {split_label.lower()} ROC-AUC of {best_single:.4f}. Incorporating <strong>Market Context ({levels[-1]})</strong> — NIFTY index movements and India VIX volatility — gave <strong>ROC-AUC {mlp_roc[-1]:.4f}</strong>; the selected set is <strong>{selection['selected_level']}</strong>. {selection_caveat}
        </div>
        """, unsafe_allow_html=True)

    # Confusion Matrix Display
    st.markdown('### 2. Out-of-Sample Confusion Matrix')
    cm_data = backend_data['model_metrics']['Confusion Matrix']  # [[TN, FP], [FN, TP]] on the test split
    (cm_tn, cm_fp), (cm_fn, cm_tp) = cm_data
    
    col_cm1, col_cm2 = st.columns([1, 2])
    with col_cm1:
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Direction", y="Actual Direction", color="Count"),
            x=['DOWN/CASH (0)', 'UP (1)'],
            y=['Actual DOWN (0)', 'Actual UP (1)'],
            text_auto=True,
            color_continuous_scale="Blues"
        )
        fig_cm.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, width='stretch')
        
    with col_cm2:
        st.markdown(f"""
        **Confusion Matrix Analysis ({cm_tn + cm_fp + cm_fn + cm_tp} Test Days):**
        - **True Negatives ({cm_tn} days):** Correctly identified Down/Cash days.
        - **True Positives ({cm_tp} days):** Correctly identified Up days.
        - **False Positives ({cm_fp} days):** Predicted Up when market was Down.
        - **False Negatives ({cm_fn} days):** Predicted Down when market was Up.
        
        *The model predicted UP on {(cm_tp + cm_fp) / (cm_tn + cm_fp + cm_fn + cm_tp):.1%} of test days (precision {backend_data['model_metrics']['Precision']:.1%}), holding cash otherwise.*
        """)


# ==============================================================================
# TAB 3: METHODOLOGY PAGE (Technical Rigor & Pipeline Specs)
# ==============================================================================
with tab_methodology:
    st.markdown('<div class="section-heading">Research Methodology & Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Technical specifications of the 10-year stationary pipeline, neural network structure, and validation protocols.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 1. Data Ingestion & Time Horizon
    - **Dataset Horizon:** 10 Years (October 2016 – October 2026).
    - **Raw Observations:** __RAW_SESSIONS__ raw daily trading sessions.
    - **Aligned Stationary Dataset:** __ROWS__ clean observations after 200-day indicator warm-up (calendar rule: __CAL_RULE__).
    - **Assets Ingested:** RELIANCE.NS (Target asset), NIFTY 50 (`^NSEI`), India VIX (`^INDIAVIX`).

    ### 2. 27 Scale-Invariant Stationary Features
    All nominal Rupee price levels were eliminated from feature matrix $X$ to prevent non-stationarity leakage:
    - **Returns (F0):** 1d, 5d, 10d, 20d Returns, High-Low Spread Ratio, Close-Open Spread Ratio.
    - **Trend Ratios (F1):** Distance to 20-day, 50-day, and 200-day SMAs (`Close / SMA - 1`), `SMA20_50_Ratio`, `SMA50_200_Ratio`.
    - **Momentum (F2):** 14-day RSI, Normalized MACD Line (`(EMA12-EMA26)/Close`), MACD Hist Ratio, 12-day Rate of Change.
    - **Volatility (F3):** Normalized ATR (`ATR14 / Close`), 20-day Rolling Volatility, Bollinger %B, Bollinger Bandwidth.
    - **Volume (F4):** 1-day Volume Change Ratio, Volume relative to 20-day SMA Volume (`Vol / Vol_SMA20`).
    - **Market Context (F5):** NIFTY 1d & 5d Returns, NIFTY 20-day Volatility, NIFTY Distance to SMA50, India VIX Level, India VIX 1d Change.

    ### 3. Model Architecture Specs
    - **Architecture:** PyTorch `ConfigurableMLP`
    - **Layer Dimensions:** 27 Input Features → 32 Hidden Neurons → 16 Hidden Neurons → 1 Output Logit.
    - **Regularization:** Dropout (0.20), L2 Weight Decay ($1\times 10^{-3}$), Early Stopping (patience = 10 epochs).
    - **Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) using class pos_weight ($\text{num\_neg}/\text{num\_pos}$).
    - **Optimizer:** Adam ($\text{lr} = 0.001$), `ReduceLROnPlateau` scheduler.
    - **Normalization:** `StandardScaler` fitted **exclusively** on Training split $X_{train}$.

    ### 4. Validation & Leakage Prevention
    - **Chronological Split:** __SPLIT__
    - **No Lookahead Bias:** Features at time $t$ use data available strictly at or before $t$. Signals at $t$ determine positions for $t \to t+1$ returns.

    ---

    ### Why Not Simply Use Accuracy?
    In financial direction prediction, accuracy alone is a misleading metric due to market regime shifts and class imbalances. For example, during a strong bull run, a naive model predicting "UP" every day can achieve 65%+ accuracy while possessing zero true predictive capacity. 

    This research uses **ROC-AUC** (Receiver Operating Characteristic Area Under Curve) and **Balanced Accuracy** as primary metrics because they evaluate the model's true ability to separate upward vs downward probability distributions independently of arbitrary decision thresholds.
    """
        .replace("__RAW_SESSIONS__", f"{run['data']['raw_sessions']:,}")
        .replace("__ROWS__", f"{run['data']['rows']:,}")
        .replace("__CAL_RULE__", run['data']['calendar_rule'])
        .replace("__SPLIT__", "; ".join(
            f"{name.capitalize()} {run['split'][key]['n']:,} samples ({run['split'][key]['start']} to {run['split'][key]['end']})"
            for key, name in (("train", "train"), ("val", "validation"), ("test", "test")))
            + f" — {run['split']['method']}."
            + (" The test period was also used to compare feature sets." if run['feature_selection']['split'] == 'test' else ""))
    )


# ==============================================================================
# TAB 4: ABOUT PAGE (Project Scope & Disclaimers)
# ==============================================================================
with tab_about:
    st.markdown('<div class="section-heading">About StockSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheading">Project overview, scope, and operational boundaries.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is this project?
    **StockSense** is an empirical financial machine learning project developed to investigate whether scale-invariant technical indicators, volume patterns, and broader market context (NIFTY 50 and India VIX) contain statistically significant predictive signal for next-day stock price direction.

    The project enforces rigorous scientific standards to eliminate common financial ML pitfalls:
    - **100% Stationary Data Pipeline:** Eliminating price-drift artifacts.
    - **Strict Data Isolation:** Preventing data leakage and lookahead bias.
    - **Vectorized Friction Modeling:** Accounting for real-world transaction costs and slippage.

    ---

    ### Important Limitations & Boundaries
    1. **Financial Markets are Extremely Noisy:** Price movements are influenced by macroeconomic news, earnings reports, geopolitical events, and institutional liquidity flows that technical indicators cannot capture.
    2. **Historical Performance $\neq$ Future Results:** Market regimes change over time. Statistical patterns observed in historical data may decay or reverse.
    3. **Model Predictions are Probabilistic:** Outputs represent statistical estimates of direction probability, not guaranteed forecasts.
    4. **Execution Friction:** Real-world trading incurs bid-ask spreads, order execution delays, and market impact costs.
    5. **Educational & Research Scope:** This application is strictly an academic research project and is not intended for live financial trading.
    """)

# ── GLOBAL DISCLAIMER FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-box">
  <strong>Disclaimer:</strong> StockSense is an academic financial machine learning research application. All content, predictions, metrics, and backtest results are presented strictly for educational and research purposes. This application does not constitute financial advice, investment advice, or trading recommendations. Past historical performance does not guarantee future results. Do not make real-money trading decisions based on this system.
</div>
""", unsafe_allow_html=True)
