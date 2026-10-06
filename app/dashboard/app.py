import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import sys

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.inference.engine import InferenceEngine
from src.data.live_provider import YFinanceLiveProvider

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Direction Research — RELIANCE.NS",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Typography & spacing (editorial, not dashboard) ──────────────────────────
st.markdown("""
<style>
  /* Base */
  .block-container { padding: 2.5rem 3rem 3rem 3rem; max-width: 1100px; }
  body, p, li { font-family: 'Georgia', serif; color: #222; line-height: 1.65; }
  h1, h2, h3, h4 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    font-weight: 600; color: #111; letter-spacing: -0.02em; }

  /* Section titles */
  .section-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #888; margin-bottom: 0.25rem;
  }
  .section-title {
    font-size: 1.35rem; font-weight: 600; color: #111;
    margin-bottom: 0.4rem; line-height: 1.3;
  }
  .section-sub {
    font-size: 0.92rem; color: #555; margin-bottom: 1.4rem; line-height: 1.6;
  }

  /* Metric card — clean, flat */
  .metric-card {
    background: #f7f7f7; border-radius: 6px;
    padding: 1rem 1.2rem; margin-bottom: 0.6rem;
  }
  .metric-label { font-size: 0.78rem; color: #666; font-family: sans-serif;
                  text-transform: uppercase; letter-spacing: 0.06em; }
  .metric-human { font-size: 0.85rem; color: #444; margin: 0.15rem 0 0.4rem 0;
                  font-style: italic; }
  .metric-value { font-size: 1.45rem; font-weight: 700; color: #111;
                  font-family: 'Courier New', monospace; }
  .metric-tech  { font-size: 0.72rem; color: #999; margin-top: 0.2rem; }

  /* Inline note */
  .inline-note {
    font-size: 0.83rem; color: #666; font-style: italic;
    border-left: 3px solid #ddd; padding-left: 0.75rem;
    margin: 0.6rem 0 1rem 0;
  }

  /* Finding box */
  .finding {
    background: #fffbf0; border-left: 4px solid #d4a800;
    padding: 0.8rem 1.1rem; border-radius: 0 6px 6px 0;
    margin: 1rem 0; font-size: 0.9rem; color: #333;
  }

  /* Limitation box */
  .limit-box {
    background: #f9f9f9; border: 1px solid #e5e5e5;
    border-radius: 6px; padding: 1rem 1.4rem; margin-top: 0.5rem;
  }

  /* Divider */
  .soft-rule { border: none; border-top: 1px solid #e8e8e8; margin: 2.5rem 0; }

  /* Hero disclaimer */
  .hero-disclaimer {
    display: inline-block; font-size: 0.78rem; color: #888;
    background: #f0f0f0; padding: 0.25rem 0.7rem;
    border-radius: 4px; margin-top: 0.5rem;
  }

  /* Footer */
  .footer-note {
    font-size: 0.78rem; color: #999; border-top: 1px solid #e8e8e8;
    padding-top: 1.2rem; margin-top: 3rem; line-height: 1.7;
  }

  /* Tag pill */
  .tag { display: inline-block; background: #eee; color: #555;
         font-size: 0.72rem; padding: 0.2em 0.6em; border-radius: 4px;
         font-family: monospace; margin-right: 0.3rem; }

  /* Prediction direction block */
  .pred-up   { font-size: 2rem; font-weight: 700; color: #1a7f37; }
  .pred-down { font-size: 2rem; font-weight: 700; color: #c0392b; }
  .pred-neutral { font-size: 2rem; font-weight: 700; color: #555; }
</style>
""", unsafe_allow_html=True)

TICKER = "RELIANCE.NS"
MODEL  = "mlp"

# ── Session state ────────────────────────────────────────────────────────────
if 'live_result' not in st.session_state:
    st.session_state.live_result = None


# ── Data loading (no changes to logic) ─────────────────────────────────────
@st.cache_data
def load_historical_data():
    data = {}
    fp = PROJECT_ROOT / f"data/processed/{TICKER}_features.csv"
    if fp.exists():
        data['features'] = pd.read_csv(fp, index_col=0, parse_dates=True)

    mp = PROJECT_ROOT / f"models/{TICKER}_{MODEL}_Target_Direction_metrics.json"
    if mp.exists():
        with open(mp) as f: data['model_metrics'] = json.load(f)

    btp = PROJECT_ROOT / f"reports/{TICKER}_{MODEL}_backtest_metrics.json"
    if btp.exists():
        with open(btp) as f: data['backtest_metrics'] = json.load(f)

    btc = PROJECT_ROOT / f"reports/{TICKER}_{MODEL}_backtest_curve.csv"
    if btc.exists():
        data['backtest_curve'] = pd.read_csv(btc, index_col=0, parse_dates=True)

    pp = PROJECT_ROOT / f"reports/{TICKER}_{MODEL}_perm_importance.csv"
    if pp.exists(): data['explainability'] = pd.read_csv(pp)

    ap = PROJECT_ROOT / f"reports/{TICKER}_ablation_results.json"
    if ap.exists():
        with open(ap) as f: data['ablation'] = json.load(f)

    preds_p = PROJECT_ROOT / f"data/predictions/{TICKER}_{MODEL}_Target_Direction_test_preds.csv"
    if preds_p.exists():
        data['latest_preds'] = pd.read_csv(preds_p, index_col=0, parse_dates=True)

    return data

hist = load_historical_data()


# ════════════════════════════════════════════════════════════════════════════
# 1 ── HERO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Research Project · RELIANCE.NS · NSE</p>
<h1 style='font-size:2rem; margin-bottom: 0.4rem;'>
  Stock Direction Prediction Using Deep Learning
</h1>
<p style='font-size:1.05rem; color:#444; max-width:680px; line-height:1.65;'>
  This project studies whether patterns in historical prices, trading activity,
  and technical indicators can help a deep-learning model anticipate a stock's
  next-day direction — <em>up</em> or <em>down</em>.
</p>
<p style='font-size:0.92rem; color:#555; max-width:680px; margin-top:0.6rem;'>
  We don't just build a model — we run a controlled experiment to find out
  <strong>which types of information actually help</strong>. The findings are
  presented honestly, including where the model fell short.
</p>
<span class='hero-disclaimer'>Research experiment · Not investment advice</span>
""", unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 2 ── TODAY'S MODEL VIEW (LIVE INFERENCE)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Latest Market Data</p>
<p class='section-title'>What does the model see right now?</p>
<p class='section-sub'>
  Pull the latest available market data for RELIANCE.NS, run it through the
  trained model, and get a next-day direction estimate. The model is not
  retrained — it uses the weights saved after the original experiment.
</p>
""", unsafe_allow_html=True)

if st.button("Refresh Latest Market Data"):
    config = load_config(str(PROJECT_ROOT / "config" / "settings.yaml"))
    provider = YFinanceLiveProvider()
    engine   = InferenceEngine(config, provider, PROJECT_ROOT)
    with st.spinner("Fetching data and running inference…"):
        try:
            st.session_state.live_result = engine.run_live_inference(TICKER, MODEL)
        except Exception as e:
            st.session_state.live_result = {"status": "Error", "message": str(e)}

res = st.session_state.live_result

if res is None:
    st.markdown("""
    <div class='inline-note'>
      Click <strong>Refresh Latest Market Data</strong> above to load the most
      recent trading session and generate a prediction.
    </div>
    """, unsafe_allow_html=True)

elif res.get("status") == "Error":
    st.warning(
        f"Live data is temporarily unavailable — "
        f"{res.get('message', 'unknown error')}. "
        "The research sections below remain fully functional."
    )
else:
    ts          = res.get('timestamp')
    latest_px   = res.get('price', 0)
    open_px     = res.get('open',  0)
    daily_chg   = (latest_px / open_px - 1) if open_px else 0
    direction   = res.get('predicted_direction', 'UNKNOWN')
    p_up        = res.get('direction_prob', 0.0)
    p_down      = 1 - p_up
    is_stale    = (datetime.now() - ts).days >= 1
    data_label  = "Latest available trading session" if is_stale else "Live / latest"

    st.markdown(f"""
    <p style='font-size:0.82rem; color:#888; margin-bottom:1rem;'>
      Data source: <span class='tag'>yfinance</span>
      Status: <span class='tag'>{data_label}</span>
      Observation: <span class='tag'>{ts.strftime('%d %b %Y')}</span>
    </p>
    """, unsafe_allow_html=True)

    if is_stale:
        st.caption(
            f"The market is likely closed. Using data from the most recent "
            f"completed trading session: {ts.strftime('%d %b %Y')}."
        )

    lc1, lc2, lc3 = st.columns([1, 1, 2])
    with lc1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Latest Price</div>
          <div class='metric-value'>₹{latest_px:,.2f}</div>
          <div class='metric-tech'>Change from open: {daily_chg:+.2%}</div>
        </div>
        """, unsafe_allow_html=True)

    with lc2:
        dir_class = "pred-up" if direction == "UP" else ("pred-down" if direction == "DOWN" else "pred-neutral")
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Model's next-day estimate</div>
          <div class='{dir_class}'>{direction}</div>
          <div class='metric-tech'>P(Up) {p_up:.1%} · P(Down) {p_down:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with lc3:
        st.markdown(f"""
        <div class='inline-note'>
          <strong>What does this mean?</strong><br>
          The model currently sees a slightly higher probability of a
          <strong>{"downward" if direction == "DOWN" else "upward"}</strong>
          move tomorrow. This is a statistical estimate based on patterns in
          recent data — not a guarantee of what the market will do.
          <br><br>
          <em>Model: MLP · Feature set: F4 (price + volume indicators) ·
          Prediction generated: {datetime.now().strftime('%d %b %Y %H:%M')}</em>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 3 ── PRICE CHART
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Historical Context</p>
<p class='section-title'>How has the stock been moving?</p>
<p class='section-sub'>
  The chart shows RELIANCE.NS price over the last 150 trading days, alongside
  a 50-day moving average (the orange line) and Bollinger Bands (dotted grey),
  which show how far the price has been from its recent average.
</p>
""", unsafe_allow_html=True)

features_df = hist.get('features', pd.DataFrame())
if not features_df.empty:
    plot_df = features_df.tail(150)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'], name='Price',
        increasing_line_color='#2ecc71', decreasing_line_color='#e74c3c'
    ))
    if 'SMA_50' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['SMA_50'],
            line=dict(color='#e67e22', width=1.5), name='50-day average'
        ))
    if 'BB_Upper' in plot_df.columns:
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['BB_Upper'],
            line=dict(color='#aaa', width=1, dash='dot'), name='Upper band'
        ))
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['BB_Lower'],
            line=dict(color='#aaa', width=1, dash='dot'), name='Lower band',
            fill='tonexty', fillcolor='rgba(180,180,180,0.07)'
        ))
    fig.update_layout(
        height=430, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend=dict(orientation='h', y=-0.12, x=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("What are these indicators?"):
    st.markdown("""
    - **50-day moving average** — a smoothed view of the price trend over the last 50 trading days. When the current price is above it, the stock has generally been rising.
    - **Bollinger Bands** — show how far the price is from its recent average. Wider bands mean more volatility. Prices near the outer bands are statistically unusual.
    - The model also uses RSI (measures if a stock is overbought or oversold), MACD (tracks momentum shifts), and volume-based signals.
    """)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 4 ── MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Evaluation</p>
<p class='section-title'>How well does the model perform?</p>
<p class='section-sub'>
  The model was evaluated on data it had never seen during training — roughly
  the final 15% of the timeline. This is the closest we can get to measuring
  how it might behave going forward.
</p>
""", unsafe_allow_html=True)

metrics = hist.get('model_metrics', {})
if metrics:
    mc1, mc2, mc3, mc4 = st.columns(4)

    with mc1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Separating up vs down days</div>
          <div class='metric-value'>{metrics.get('ROC-AUC', 0):.2f}<span style='font-size:1rem; color:#888;'> / 1.00</span></div>
          <div class='metric-tech'>ROC-AUC · 0.5 = random · 1.0 = perfect</div>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>When it predicted UP, was it right?</div>
          <div class='metric-value'>{metrics.get('Precision', 0):.0%}</div>
          <div class='metric-tech'>Precision — correctness of positive predictions</div>
        </div>
        """, unsafe_allow_html=True)

    with mc3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Out of all up days, how many did it catch?</div>
          <div class='metric-value'>{metrics.get('Recall', 0):.0%}</div>
          <div class='metric-tech'>Recall — coverage of actual positive cases</div>
        </div>
        """, unsafe_allow_html=True)

    with mc4:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Balance of precision and coverage</div>
          <div class='metric-value'>{metrics.get('F1', 0):.2f}<span style='font-size:1rem; color:#888;'> / 1.00</span></div>
          <div class='metric-tech'>F1 Score — harmonic mean of precision & recall</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='inline-note'>
      <strong>How to read this:</strong> The model is very selective — when it
      predicts an upward day (73% of those predictions are correct), but it
      misses most of the actual up days (only catching about 28% of them).
      It is precise but conservative.
    </div>
    """, unsafe_allow_html=True)

with st.expander("Why do we test on unseen data?"):
    st.markdown("""
    A model can appear very accurate on the data it learned from — but that's
    like a student memorising answers rather than understanding the subject.
    By holding out the final 15% of the historical timeline and testing only
    on that, we get a fairer picture of real performance. All metrics shown
    here come from that held-out test period.
    """)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 5 ── BACKTEST
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Historical Simulation</p>
<p class='section-title'>Would these predictions have helped historically?</p>
<p class='section-sub'>
  Backtesting applies the model's historical predictions to a simulated
  trading strategy and measures how it would have performed — including
  transaction costs and slippage. This is a retrospective test, not a
  forecast of future performance.
</p>
""", unsafe_allow_html=True)

bt_metrics = hist.get('backtest_metrics', {})
bt_df      = hist.get('backtest_curve',  pd.DataFrame())

if bt_metrics and not bt_df.empty:
    bc1, bc2, bc3, bc4 = st.columns(4)

    bnh_ret = bt_metrics.get('BnH_Total_Return', 0)
    strat_ret = bt_metrics.get('Strategy_Total_Return', 0)

    with bc1:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>How much the strategy gained or lost</div>
          <div class='metric-value'>{strat_ret:.1%}</div>
          <div class='metric-tech'>Strategy total return · Buy & Hold: {bnh_ret:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with bc2:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Return relative to volatility taken</div>
          <div class='metric-value'>{bt_metrics.get('Strategy_Sharpe', 0):.2f}</div>
          <div class='metric-tech'>Sharpe ratio · higher is more efficient</div>
        </div>
        """, unsafe_allow_html=True)

    with bc3:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Largest drop from a previous high</div>
          <div class='metric-value'>{bt_metrics.get('Strategy_Max_Drawdown', 0):.1%}</div>
          <div class='metric-tech'>Maximum drawdown</div>
        </div>
        """, unsafe_allow_html=True)

    with bc4:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='metric-label'>Trades that ended profitably</div>
          <div class='metric-value'>{bt_metrics.get('Win_Rate', 0):.0%}</div>
          <div class='metric-tech'>Win rate · {int(bt_metrics.get('Total_Trades', 0))} total trades</div>
        </div>
        """, unsafe_allow_html=True)

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df['Strategy_Equity'],
        name='Model strategy', line=dict(color='#2980b9', width=2)
    ))
    fig_bt.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df['BnH_Equity'],
        name='Buy & Hold', line=dict(color='#bbb', width=1.5, dash='dash')
    ))
    fig_bt.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_white", yaxis_title="Portfolio value (normalised)",
        legend=dict(orientation='h', y=-0.15, x=0)
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("""
    <div class='inline-note'>
      <strong>What this chart tells us:</strong> The blue line is the simulated
      model strategy; the grey dashed line is simply buying and holding the stock.
      In this test period, Buy & Hold ended significantly higher. The model
      strategy stayed mostly in cash — trading only 14 times — resulting in
      very low drawdown but also missing most of the market's upside.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='finding'>
      <strong>Key finding:</strong> A good classification score does not
      automatically produce better trading returns. The model was highly
      selective (72% win rate on the trades it did take), but its caution
      meant it sat in cash for most of the period while the market rose.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 6 ── FEATURE ABLATION
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Experiment</p>
<p class='section-title'>What information actually helps the model?</p>
<p class='section-sub'>
  Rather than feeding every indicator into the model at once, we added groups
  of information step by step and measured whether performance improved. This
  is called a feature ablation study.
</p>
""", unsafe_allow_html=True)

abl_data = hist.get('ablation', {})
if 'mlp' in abl_data:
    # Feature group descriptions for humans
    group_desc = {
        'F0': ('Basic price & volume', 'Open, High, Low, Close, Volume, daily return'),
        'F1': ('+ Trend',             'Added moving averages (SMA/EMA)'),
        'F2': ('+ Momentum',          'Added RSI, MACD, Rate of Change'),
        'F3': ('+ Volatility',        'Added Bollinger Bands, ATR, rolling volatility'),
        'F4': ('+ Volume patterns',   'Added OBV, volume ratios, volume momentum'),
        'F5': ('+ Market context',    'Added NIFTY returns and India VIX'),
    }

    roc_vals = [abl_data['mlp'].get(f"F{i}", {}).get("ROC-AUC") for i in range(6)]
    f1_vals  = [abl_data['mlp'].get(f"F{i}", {}).get("F1")      for i in range(6)]
    labels   = [f"F{i}" for i in range(6)]

    # Table-style summary
    rows = []
    for i in range(6):
        fk = f"F{i}"
        r  = abl_data['mlp'].get(fk, {}).get("ROC-AUC", None)
        f1 = abl_data['mlp'].get(fk, {}).get("F1",      None)
        human, tech = group_desc[fk]
        rows.append({
            "Group": fk,
            "What we added": human,
            "Details": tech,
            "Score (ROC-AUC)": f"{r:.3f}" if r is not None else "—",
            "F1": f"{f1:.3f}" if f1 is not None else "—",
        })
    tbl = pd.DataFrame(rows).set_index("Group")
    st.dataframe(tbl, use_container_width=True)

    fig_ab = go.Figure()
    fig_ab.add_trace(go.Scatter(
        x=labels, y=roc_vals, mode='lines+markers',
        name='ROC-AUC (separating up vs down)',
        line=dict(color='#2980b9', width=2), marker=dict(size=8)
    ))
    fig_ab.add_trace(go.Scatter(
        x=labels, y=f1_vals, mode='lines+markers',
        name='F1 Score (precision + coverage balance)',
        line=dict(color='#e67e22', width=2, dash='dot'), marker=dict(size=8)
    ))
    fig_ab.add_hline(y=0.5, line_dash='dash', line_color='#ddd',
                     annotation_text='Random chance (0.50)', annotation_position='top left')
    fig_ab.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        template='plotly_white', yaxis_title='Score',
        legend=dict(orientation='h', y=-0.2, x=0)
    )
    st.plotly_chart(fig_ab, use_container_width=True)

    st.markdown("""
    <div class='inline-note'>
      <strong>How to read this:</strong> Each point represents training and
      testing the model with one more group of features added. Higher is better;
      0.50 is equivalent to random guessing.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='finding'>
      <strong>What we found:</strong> Adding trend indicators (moving averages)
      actually made performance <em>worse</em> compared to raw prices alone —
      suggesting they introduced redundant noise for this model. Performance
      improved noticeably only after volume-related features were introduced
      at F4, where the score reached 0.64. Adding market-context data (F5)
      produced no further improvement.
    </div>
    """, unsafe_allow_html=True)

with st.expander("Why run this experiment?"):
    st.markdown("""
    It's easy to assume that more data always helps. This experiment shows
    that's not always true — some indicators can actually hurt performance
    by adding noise. The ablation study gives us evidence about which types
    of information are genuinely useful for this model on this dataset,
    rather than relying on assumptions.
    """)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 7 ── FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Model Interpretation</p>
<p class='section-title'>Which signals mattered most?</p>
<p class='section-sub'>
  To understand what the model relied on, we measured how much its performance
  dropped when each feature's values were randomly shuffled. If shuffling a
  feature hurts performance, that feature was meaningful to the model.
</p>
""", unsafe_allow_html=True)

if 'explainability' in hist:
    exp_df = hist['explainability'].head(12).sort_values('Importance_Mean', ascending=True)
    fig_exp = px.bar(
        exp_df, x='Importance_Mean', y='Feature', orientation='h',
        error_x='Importance_Std', template='plotly_white',
        color_discrete_sequence=['#2980b9'],
        labels={'Importance_Mean': 'Drop in performance when shuffled', 'Feature': ''}
    )
    fig_exp.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("""
    <div class='inline-note'>
      <strong>Important caveat:</strong> This shows which features the model
      <em>associated</em> with better predictions. It does not mean these
      features <em>cause</em> the stock to move in any direction. Association
      is not causation.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 8 ── WHAT DID WE LEARN?
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Research Summary</p>
<p class='section-title'>What did this experiment teach us?</p>
""", unsafe_allow_html=True)

findings_col1, findings_col2 = st.columns(2)

with findings_col1:
    st.markdown("""
    <div class='metric-card' style='margin-bottom:1rem;'>
      <div class='metric-label'>01</div>
      <p style='font-weight:600; margin:0.3rem 0 0.3rem 0;'>Prediction is possible, but hard.</p>
      <p style='font-size:0.88rem; color:#555; margin:0;'>
        The model achieved a ROC-AUC of 0.64 on unseen data —
        meaningfully above random (0.50), but well below reliable.
        Stock markets are inherently noisy.
      </p>
    </div>
    <div class='metric-card' style='margin-bottom:1rem;'>
      <div class='metric-label'>02</div>
      <p style='font-weight:600; margin:0.3rem 0 0.3rem 0;'>More features did not always help.</p>
      <p style='font-size:0.88rem; color:#555; margin:0;'>
        Adding trend indicators (moving averages) actually reduced performance.
        The model improved only after volume patterns were included.
        This challenges the assumption that more data is always better.
      </p>
    </div>
    """, unsafe_allow_html=True)

with findings_col2:
    st.markdown("""
    <div class='metric-card' style='margin-bottom:1rem;'>
      <div class='metric-label'>03</div>
      <p style='font-weight:600; margin:0.3rem 0 0.3rem 0;'>Classification accuracy ≠ trading profit.</p>
      <p style='font-size:0.88rem; color:#555; margin:0;'>
        Despite a 73% precision score, the model's strategy returned only
        1.9% over the test period while Buy & Hold returned 15.4%. Being
        selective kept drawdowns low but missed most of the market's gains.
      </p>
    </div>
    <div class='metric-card' style='margin-bottom:1rem;'>
      <div class='metric-label'>04</div>
      <p style='font-weight:600; margin:0.3rem 0 0.3rem 0;'>Volume context was the most useful signal.</p>
      <p style='font-size:0.88rem; color:#555; margin:0;'>
        Both the ablation study and feature importance analysis pointed to
        volume-related features as the most informative for this model.
        This is consistent with the idea that price moves backed by high
        volume carry more information than price moves alone.
      </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 9 ── LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<p class='section-label'>Honest Boundaries</p>
<p class='section-title'>What this model cannot tell you</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class='limit-box'>
  <ul style='margin:0; padding-left:1.2rem; font-size:0.9rem; line-height:1.9; color:#444;'>
    <li>It cannot guarantee future price movements. Historical patterns change.</li>
    <li>The backtest covers a specific period. Results from a different period could differ substantially.</li>
    <li>The model was trained on one stock (RELIANCE.NS). It has not been tested on other assets.</li>
    <li>Live data from this provider is delayed, not real-time. Execution at modelled prices is not guaranteed.</li>
    <li>The model was not optimised with automated hyperparameter search — the architecture is a starting baseline.</li>
    <li>A strong historical score does not guarantee any future profitability.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='soft-rule'>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# 10 ── TECHNICAL DETAILS (EXPANDABLE)
# ════════════════════════════════════════════════════════════════════════════
with st.expander("Technical details — model architecture, validation methodology, metrics"):
    st.markdown(f"""
    **Model:** Multilayer Perceptron (MLP)
    — hidden layers `[128, 64, 32]` neurons, ReLU activations, Dropout 0.3.

    **Feature set used (F4):** OHLCV + daily returns, SMA (20/50/200), EMA (12/26/50),
    RSI, MACD, Bollinger Bands, ATR, rolling volatility, OBV, volume change, volume ratio.

    **Target:** Binary next-day direction — `1` if Close[t+1] > Close[t], else `0`.

    **Train / Validation / Test split:** Chronological 70 / 15 / 15.
    No random shuffling. Scaler fitted only on training data.

    **Loss function:** Binary Cross-Entropy with class weighting to handle imbalance.

    **Optimiser:** Adam, learning rate 0.001, ReduceLROnPlateau scheduler.

    **Early stopping:** Patience = 10 epochs on validation loss.

    **Explainability:** Permutation importance (SHAP was attempted; fell back to permutation
    due to tensor-dimension compatibility).

    **Backtest parameters:** 0.10% commission + 0.05% slippage per trade.
    Signal threshold: probability ≥ 0.50 → Long, else Cash.
    """)

    if metrics:
        st.markdown("**Full out-of-sample test metrics:**")
        st.json(metrics)

    if bt_metrics:
        st.markdown("**Full backtest metrics:**")
        st.json(bt_metrics)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer-note'>
  <strong>Disclaimer:</strong> This is a research and educational project built
  to study the predictive utility of technical indicators for stock direction
  classification. It does not constitute investment advice, financial advice, or
  a trading recommendation of any kind. Past model performance does not
  guarantee future results. Do not make financial decisions based on this system.
</div>
""", unsafe_allow_html=True)
