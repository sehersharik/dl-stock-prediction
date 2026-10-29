# StockSense: Next-Day Direction Prediction for Reliance Industries

> **Research prototype, not investment advice.** Nothing here is a recommendation to buy or sell any security.

**Live app:** https://stockpredictioncc.streamlit.app · **Stock:** Reliance Industries (`RELIANCE.NS`) · **Context:** NIFTY 50 (`^NSEI`), India VIX (`^INDIAVIX`) · **Models:** PyTorch MLP, LSTM/GRU, scikit-learn and XGBoost baselines

<!-- Generated file: edit docs/README.template.md, then run `python main.py readme`. Every number comes from reports/metrics.json (run {{run_id}}). -->

---

## Conclusion

{{conclusion}}

> ⚠️ **Backtest result:** {{backtest_headline}}

## Key Results

{{key_results}}

---

## The {{n_bugs}} Bugs Found and Fixed

An end-to-end audit (inspect, fix, re-measure) found these problems. Each one inflated the earlier results, distorted live predictions, or stopped the live app from working.

{{bugs}}

How the reported numbers changed:

{{audit_table}}

---

## Method

### Data
- Daily OHLCV for RELIANCE.NS, NIFTY 50 and India VIX from Yahoo Finance (`yfinance`), October 2016 to October 2026: {{raw_sessions}} raw Reliance sessions.
- **Trading calendar:** {{calendar_rule}}. This removes holiday placeholder bars (zero volume, no index data) that had silently blanked 50-day windows. A missing India VIX day leaves only that row and the next without VIX features.
- After the 200-day warm-up: **{{rows}} rows** with 27 features.

### Features (27, all scale-free)
All features at day *t* use data up to and including *t* (`src/features/generator.py`, shared by training and live inference).

| Group | Features | Definition |
|---|---|---|
| F0 Base | `Return_1d`, `Return_5d`, `Return_10d`, `Return_20d` | $C_t / C_{t-k} - 1$ |
| | `HL_Spread`, `CO_Spread` | $(H_t - L_t)/C_t$, $(C_t - O_t)/O_t$ |
| F1 Trend | `Dist_SMA20`, `Dist_SMA50`, `Dist_SMA200` | $C_t / \text{SMA}_k - 1$ |
| | `SMA20_50_Ratio`, `SMA50_200_Ratio` | $\text{SMA}_{20}/\text{SMA}_{50} - 1$, $\text{SMA}_{50}/\text{SMA}_{200} - 1$ |
| F2 Momentum | `RSI_14` | 14-day RSI ÷ 100 (0–1) |
| | `MACD_Ratio`, `MACD_Hist_Ratio` | MACD(12, 26, 9) line and histogram ÷ $C_t$ |
| | `ROC_12` | $C_t / C_{t-12} - 1$ |
| F3 Volatility | `ATR_Ratio` | 14-day ATR ÷ $C_t$ |
| | `Rolling_Vol_20` | 20-day std of daily returns (not annualized) |
| | `BB_PctB`, `BB_Width` | Bollinger(20, 2) %B and band width ÷ middle band |
| F4 Volume | `Vol_Change`, `Vol_Ratio_20` | $V_t/V_{t-1} - 1$, $V_t / \text{SMA}_{20}(V)$ |
| F5 Market context | `NIFTY_Return_1d`, `NIFTY_Return_5d` | NIFTY 1- and 5-day returns |
| | `NIFTY_Vol_20`, `NIFTY_Dist_SMA50` | 20-day std of NIFTY daily returns; NIFTY ÷ its 50-day SMA − 1 |
| | `VIX_Level`, `VIX_Change_1d` | India VIX close ÷ 100; 1-day change |

### Target
$y_t = 1$ if $C_{t+1}/C_t - 1 > 0$, else 0, where $t+1$ is the **next trading session** on the calendar above. It's computed before any rows are filtered, so a gap in the data can never turn "tomorrow" into "weeks later".

### Splits (fixed dates)
| Split | Rows | Dates | UP share |
|---|---|---|---|
| Train | {{n_train}} | {{train_start}} → {{train_end}} | {{up_train}} |
| Validation | {{n_val}} | {{val_start}} → {{val_end}} | {{up_val}} |
| Test | {{n_test}} | {{test_start}} → {{test_end}} | {{up_test}} |

Scalers are fitted on training rows only. Everything that involves a choice (feature set, early stopping, no-trade band) uses **validation only**. Each final model was scored on the test set once.

### Models
- **MLP** {{n_params}} parameters: 27 → 32 → 16 → 1, ReLU, dropout 0.2, Adam (lr 1e-3, weight decay 1e-3), class-weighted BCE, early stopping on validation loss (patience 10). Seed 42.
- **Baselines:** majority class, logistic regression, random forest, XGBoost (same features and splits).
- **Sequence models:** LSTM and GRU with 10- and 20-day windows, configurations fixed in advance.

### Evaluation
- ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall, F1 at threshold 0.5.
- **Uncertainty:** stationary block bootstrap (mean block 10 trading days, because predictions are autocorrelated), paired across models; circular-shift test against chance.
- **Calibration:** reliability diagram, Brier score and log loss against a constant base-rate forecast.

### Backtest
{{backtest_rules}}

---

## Detailed Results

### Classification on the test set ({{test_years}}, {{n_test}} days)

{{results_table}}

### Statistical uncertainty

{{uncertainty_block}}

### Sequence models: LSTM and GRU

{{sequence_block}}

### Probability calibration (MLP)

{{calibration_block}}

### Feature-set selection (ablation)

{{ablation_table}}

### What the model relies on (permutation importance)

Computed by shuffling each feature column and measuring the drop in ROC-AUC. It shows what the *model* uses, not what moves the market.

{{perm_block}}

### Backtest

{{backtest_table}}

#### Transaction cost sensitivity

{{sensitivity_table}}

{{winrate_note}}

#### No-trade band (chosen on validation)

{{band_block}}

---

## Limitations
- **One stock, one test period.** {{n_test}} test days give wide confidence intervals; a different period could look better or worse.
- **Many comparisons.** Several models and variants were evaluated on the same test period. Each final model was scored once, but with many models some will look "significant" by luck. Nothing here survives a multiple-comparison correction.
- **No walk-forward evaluation yet.** `src/evaluation/splitter.py` defines expanding-window folds, but results come from a single fixed split.
- **Risk-free rate is an assumption** (6.0% a year, about the RBI 91-day T-bill level). Sharpe is also reported at 0%, 5% and 7%.
- **Data quality:** Yahoo Finance data contains placeholder bars and missing index days (handled by the calendar rule above). Prices are as provided, without separate corporate-action checks.
- **Live app:** yfinance quotes are delayed. Predictions use the last *completed* session (today's bar counts only after 16:00 IST) and are withheld if any input is missing.

---

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # pinned versions; Python ≥ 3.12
python -m pytest -q                         # test suite

python main.py features --ticker RELIANCE.NS
python main.py targets  --ticker RELIANCE.NS
python main.py ablation --ticker RELIANCE.NS              # feature set chosen on validation
python main.py train    --ticker RELIANCE.NS --model mlp  # also: majority, logistic, random_forest, xgboost
python main.py sequence-models --ticker RELIANCE.NS
python main.py explain  --ticker RELIANCE.NS --model mlp --run <run_id> --split val
python main.py threshold-band  --ticker RELIANCE.NS --run <run_id>
python main.py metrics  --new-run <run_id> --active <run_id>   # builds reports/metrics.json
python main.py readme                                          # renders this file
streamlit run app/dashboard/app.py
```

On macOS without Homebrew's `libomp`, XGBoost can use PyTorch's copy: `DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib`.

## Repository Map

| Path | Purpose |
|---|---|
| `src/features/generator.py` | Calendar rule and the 27 features (shared by training and live inference) |
| `src/targets/generator.py` | Next-session direction target |
| `src/evaluation/splitter.py` | Fixed-date splits |
| `src/models/` | MLP, LSTM/GRU, baselines and training loops (model registry in `models/RELIANCE/<model>/<run_id>/`) |
| `src/backtesting/engine.py` | Long/cash backtest: execution variants, per-side costs, buy & hold over the same periods |
| `src/evaluation/uncertainty.py`, `calibration.py` | Block bootstrap, chance test, reliability and Brier score |
| `src/evaluation/metrics_registry.py` | Builds `reports/metrics.json`, the single source of every reported number |
| `src/inference/engine.py` | Live inference with the active run's model, scaler and feature list |
| `app/dashboard/app.py` | Streamlit dashboard |
| `archive/` | Frozen artifacts of earlier runs (kept for comparison) |
| `docs/learning-notes.md` | Plain-English notes for every audit phase |

---

*Research and educational use only. Historical results, including backtests, do not predict future returns.*
