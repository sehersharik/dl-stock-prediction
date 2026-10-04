import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import sys

# Add project root to sys path to import modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.inference.engine import InferenceEngine
from src.data.live_provider import YFinanceLiveProvider

# ==========================================
# CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="Financial ML Research System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional research aesthetic CSS (Restored previous clean look)
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; font-weight: 500; color: #333333; }
    .metric-value { font-family: 'Courier New', Courier, monospace; font-size: 1.5rem; font-weight: bold; }
    .disclaimer { font-size: 0.8rem; color: #666666; border-top: 1px solid #eeeeee; padding-top: 1rem; margin-top: 3rem; }
    .status-badge { display: inline-block; padding: 0.25em 0.4em; font-size: 75%; font-weight: 700; line-height: 1; text-align: center; white-space: nowrap; vertical-align: baseline; border-radius: 0.25rem; background-color: #6c757d; color: white; }
    .stButton button { width: auto; min-width: 200px; }
</style>
""", unsafe_allow_html=True)

TICKER = "RELIANCE.NS"
MODEL = "mlp"

# Initialize Session State
if 'live_result' not in st.session_state:
    st.session_state.live_result = None

# ==========================================
# DATA LOADING (HISTORICAL)
# ==========================================
@st.cache_data
def load_historical_data():
    data = {}
    features_path = PROJECT_ROOT / f"data/processed/{TICKER}_features.csv"
    if features_path.exists():
        data['features'] = pd.read_csv(features_path, index_col=0, parse_dates=True)
    
    metrics_path = PROJECT_ROOT / f"models/{TICKER}_{MODEL}_Target_Direction_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f: data['model_metrics'] = json.load(f)
            
    bt_metrics_path = PROJECT_ROOT / f"reports/{TICKER}_{MODEL}_backtest_metrics.json"
    if bt_metrics_path.exists():
        with open(bt_metrics_path, 'r') as f: data['backtest_metrics'] = json.load(f)
            
    bt_curve_path = PROJECT_ROOT / f"reports/{TICKER}_{MODEL}_backtest_curve.csv"
    if bt_curve_path.exists():
        data['backtest_curve'] = pd.read_csv(bt_curve_path, index_col=0, parse_dates=True)
        
    perm_path = PROJECT_ROOT / f"reports/{TICKER}_{MODEL}_perm_importance.csv"
    if perm_path.exists(): data['explainability'] = pd.read_csv(perm_path)
        
    ablation_path = PROJECT_ROOT / f"reports/{TICKER}_ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path, 'r') as f: data['ablation'] = json.load(f)
        
    preds_path = PROJECT_ROOT / f"data/predictions/{TICKER}_{MODEL}_Target_Direction_test_preds.csv"
    if preds_path.exists():
        data['latest_preds'] = pd.read_csv(preds_path, index_col=0, parse_dates=True)
        
    return data

hist_data = load_historical_data()

# ==========================================
# TOP-LEVEL HEADER
# ==========================================
st.markdown("## Deep Learning Based Stock Risk & Direction Prediction")
st.markdown(f"**Asset:** `{TICKER}` | **Mode:** Research & Live Inference")
st.markdown("---")

# ==========================================
# LIVE INFERENCE PANEL (COMPACT ADDITION)
# ==========================================
st.markdown("### LIVE MARKET INFERENCE")

if st.button("Refresh Latest Market Data"):
    config = load_config(str(PROJECT_ROOT / "config" / "settings.yaml"))
    provider = YFinanceLiveProvider()
    engine = InferenceEngine(config, provider, PROJECT_ROOT)
    
    with st.spinner("Fetching latest market data & running inference..."):
        try:
            st.session_state.live_result = engine.run_live_inference(TICKER, MODEL)
        except Exception as e:
            st.session_state.live_result = {"status": "Error", "message": str(e)}

res = st.session_state.live_result

if res is None:
    st.info("Click 'Refresh Latest Market Data' to pull the most recent trading session and run live inference.")
elif res.get("status") == "Error":
    # If it failed to fetch data completely (e.g. DNS error or Missing columns)
    st.warning("Live prediction temporarily unavailable because the latest market data is stale or incomplete.")
    st.markdown(f"**Reason:** {res.get('message', 'Unknown Error')}")
    if "data_timestamp" in res:
        st.markdown(f"**Latest Available Data Timestamp:** {res.get('data_timestamp').strftime('%Y-%m-%d %H:%M')}")
else:
    # Successfully generated a prediction on the latest available data
    timestamp = res.get('timestamp')
    latest_price = res.get('price', 0)
    open_price = res.get('open', 0)
    daily_change = (latest_price / open_price - 1) if open_price else 0
    vol = res.get('volume', 0)
    
    direction = res.get('predicted_direction', 'UNKNOWN')
    p_up = res.get('direction_prob', 0.0)
    p_down = 1 - p_up
    
    is_weekend = timestamp.weekday() >= 5 or (datetime.now() - timestamp).days >= 1
    status_str = "Latest Available Trading Session" if is_weekend else "Live / Latest"
    
    st.markdown(f"**Data Source:** `yfinance` | **Data Status:** {status_str} | **Last Updated:** {timestamp.strftime('%Y-%m-%d %H:%M')}")
    if is_weekend:
        st.caption(f"*Note: Market is likely closed. Using the latest available trading session from {timestamp.strftime('%d %b %Y')}*")

    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Latest Price", f"₹{latest_price:,.2f}", f"{daily_change:.2%}")
    l_col2.metric("Predicted Direction", direction)
    l_col3.metric("P(UP)", f"{p_up:.1%}")
    l_col4.metric("P(DOWN)", f"{p_down:.1%}")
    
    st.markdown(f"**Model:** `{res.get('model_version', 'v1.0')}` | **Feature Set:** `{res.get('feature_version', 'v1.0')}` | **Prediction Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")

st.markdown("---")

# ==========================================
# RESEARCH OVERVIEW (RESTORED DASHBOARD)
# ==========================================
st.markdown("### RESEARCH OVERVIEW")

features_df = hist_data.get('features', pd.DataFrame())

# 1. Historical Test-Set Prediction Summary
if not features_df.empty and 'latest_preds' in hist_data:
    latest_feat = features_df.iloc[-1]
    prev_feat = features_df.iloc[-2]
    latest_pred = hist_data['latest_preds'].iloc[-1]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    current_price = latest_feat['Close']
    price_change = (current_price / prev_feat['Close']) - 1
    prob_up = latest_pred['Prob']
    prob_down = 1 - prob_up
    dir_str = "UP" if prob_up >= 0.5 else "DOWN"
    risk_lvl = "ELEVATED" if prob_down > 0.6 else ("MODERATE" if prob_down > 0.4 else "LOW")
    
    col1.metric("Historical Close", f"₹{current_price:,.2f}", f"{price_change:.2%}")
    col2.metric("Test Direction", dir_str)
    col3.metric("Prob (Upward)", f"{prob_up:.1%}")
    col4.metric("Prob (Downside)", f"{prob_down:.1%}")
    col5.metric("Risk Level", risk_lvl)

# 2. Historical Price & Technical Indicators
st.markdown("#### Historical Price & Technical Indicators")
if not features_df.empty:
    plot_df = features_df.tail(150)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name='Price'))
    if 'SMA_50' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
    if 'BB_Upper' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Upper'], line=dict(color='gray', width=1, dash='dot'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BB_Lower'], line=dict(color='gray', width=1, dash='dot'), name='BB Lower'))
        
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# 3. Model Architecture & Explainability
col_mod, col_exp = st.columns([1, 2])

with col_mod:
    st.markdown("#### Model Architecture & Validation")
    metrics = hist_data.get('model_metrics', {})
    
    st.markdown(f"**Model:** Configurable {MODEL.upper()}")
    st.markdown(f"**Feature Set:** F4 (Volume Inclusion)")
    st.markdown(f"**Target:** Next-Day Direction (Binary)")
    
    if metrics:
        st.markdown("##### Out-of-Sample Test Metrics")
        st.code(f"ROC-AUC  : {metrics.get('ROC-AUC', 0):.4f}\nPR-AUC   : {metrics.get('PR-AUC', 0):.4f}\nF1 Score : {metrics.get('F1', 0):.4f}\nPrecision: {metrics.get('Precision', 0):.4f}", language="text")

with col_exp:
    st.markdown("#### Feature Explanability (Permutation)")
    if 'explainability' in hist_data:
        exp_df = hist_data['explainability'].head(10).sort_values('Importance_Mean', ascending=True)
        fig_exp = px.bar(exp_df, x='Importance_Mean', y='Feature', orientation='h', error_x='Importance_Std', template='plotly_white')
        fig_exp.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_exp, use_container_width=True)

# 4. Out-of-Sample Backtest
st.markdown("---")
st.markdown("#### Out-of-Sample Backtest")

if 'backtest_curve' in hist_data and 'backtest_metrics' in hist_data:
    bt_df = hist_data['backtest_curve']
    bt_metrics = hist_data['backtest_metrics']
    
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_col1.metric("Strategy Return", f"{bt_metrics.get('Strategy_Total_Return', 0):.2%}")
    b_col2.metric("Sharpe Ratio", f"{bt_metrics.get('Strategy_Sharpe', 0):.2f}")
    b_col3.metric("Max Drawdown", f"{bt_metrics.get('Strategy_Max_Drawdown', 0):.2%}")
    b_col4.metric("Win Rate", f"{bt_metrics.get('Win_Rate', 0):.2%}")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_Equity'], name='Strategy (MLP)', line=dict(color='blue', width=2)))
    fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['BnH_Equity'], name='Buy & Hold', line=dict(color='gray', width=1, dash='dash')))
    fig_bt.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white", yaxis_title="Equity")
    st.plotly_chart(fig_bt, use_container_width=True)

# 5. Feature Ablation
st.markdown("---")
st.markdown("#### Research: Feature Ablation Impact")
if 'ablation' in hist_data:
    abl = hist_data['ablation']
    if 'mlp' in abl:
        roc_vals = [abl['mlp'].get(f"F{i}", {}).get("ROC-AUC", None) for i in range(6)]
        f1_vals = [abl['mlp'].get(f"F{i}", {}).get("F1", None) for i in range(6)]
        
        fig_ab = go.Figure()
        fig_ab.add_trace(go.Scatter(x=[f"F{i}" for i in range(6)], y=roc_vals, mode='lines+markers', name='ROC-AUC'))
        fig_ab.add_trace(go.Scatter(x=[f"F{i}" for i in range(6)], y=f1_vals, mode='lines+markers', name='F1 Score'))
        fig_ab.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
        st.plotly_chart(fig_ab, use_container_width=True)

# ==========================================
# DISCLAIMER
# ==========================================
st.markdown("""
<div class='disclaimer'>
    <strong>SCIENTIFIC DISCLAIMER:</strong> This system is generated from available market data using a model trained on historical observations. It is a research/educational output and does not constitute investment or trading advice.
</div>
""", unsafe_allow_html=True)
