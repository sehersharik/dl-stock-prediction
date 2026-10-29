# Deep Learning Based Stock Risk and Next-Day Direction Prediction System

> **A 10-Year Empirical Financial Machine Learning Framework**  
> **Target Asset:** RELIANCE Industries Ltd. (`RELIANCE.NS`) on National Stock Exchange of India (NSE)  
> **Market Context Assets:** NIFTY 50 Index (`^NSEI`) & India VIX Volatility Index (`^INDIAVIX`)  
> **Dataset Period:** October 2016 – October 2026 ({{raw_sessions}} raw daily sessions → {{rows}} aligned stationary samples)  
> **Validated Deep Learning Architecture:** Compact PyTorch MLP `[32, 16]` with L2 Regularization & Class-Weighted BCE Loss  
> **Reported results:** generated from `reports/metrics.json` (run `{{run_id}}`) | {{header_backtest}}

> ⚠️ **Backtest result:** {{backtest_headline}}

---

## Executive Summary & System Metadata

| Parameter | Value / Specification | Description / Justification |
|---|---|---|
| **Target Asset** | `RELIANCE.NS` | Major liquid blue-chip constituent of NIFTY 50 |
| **Context Feeds** | `^NSEI` (NIFTY 50) & `^INDIAVIX` (India VIX) | Parallel macro market return & volatility feeds |
| **Data Horizon** | 10 Years (2016 – 2026) | {{raw_sessions}} raw trading sessions |
| **Effective Dataset Size** | {{rows}} Aligned Observations | After 200-day indicator warmup and missing target drop; calendar rule: {{calendar_rule}} |
| **Feature Matrix ($X$)** | 27 Stationary Dimensionless Features | Returns, ratios, normalized momentum, volatility & macro context |
| **Target Vector ($y$)** | Next-Day Direction $y_t \in \{0, 1\}$ | $y_t = 1$ if $(P_{t+1}/P_t - 1) > 0$ else $0$ |
| **Train / Val / Test Split** | Chronological ({{split_method}}) | Train: {{n_train}} \| Val: {{n_val}} \| Test: {{n_test}} ({{test_years}}) |
| **Model Architecture** | PyTorch Small MLP `[32, 16]` | Input (27) $\to$ Linear(32) $\to$ ReLU $\to$ Drop(0.2) $\to$ Linear(16) $\to$ ReLU $\to$ Drop(0.2) $\to$ Out(1) |
| **Optimizer & Loss** | Adam ($\text{lr}=10^{-3}, \text{weight\_decay}=10^{-3}$) | `BCEWithLogitsLoss` with dynamic class `pos_weight` |
| **Out-of-Sample ROC-AUC** | **{{test_roc}}** | Test set ({{n_test}} trading days, {{test_years}}); {{selection_caveat_short}} |
| **Out-of-Sample Balanced Acc**| **{{test_balacc}}** | Precision: {{test_precision}} \| Recall: {{test_recall}} \| F1: {{test_f1}} |
{{exec_backtest_row}}
| **Test Suite** | `python -m pytest` | Includes leakage, stationary-column, splitter, live-inference parity and metrics-consistency tests |

---

## Table of Contents
1. [Project Overview & Purpose](#project-overview--purpose)
2. [Core Research Question & Theoretical Foundation](#core-research-question--theoretical-foundation)
3. [Chronological Problem Diagnostics (Why the 2-Year Baseline Failed)](#chronological-problem-diagnostics-why-the-2-year-baseline-failed)
4. [The 10-Year Rebuild Strategy (2016–2026)](#the-10-year-rebuild-strategy-20162026)
5. [Data Leakage Auditing & Mathematical Isolation](#data-leakage-auditing--mathematical-isolation)
6. [Complete 27 Stationary Feature Engineering Matrix](#complete-27-stationary-feature-engineering-matrix)
7. [Target Formulation & Downside Risk Classification](#target-formulation--downside-risk-classification)
8. [Deep Learning Model Architecture & Hyperparameters](#deep-learning-model-architecture--hyperparameters)
9. [Validation Strategy & Timeline Isolation](#validation-strategy--timeline-isolation)
10. [Empirical Benchmark Comparisons & Out-of-Sample Results](#empirical-benchmark-comparisons--out-of-sample-results)
11. [Controlled Feature Ablation Progression (F0 to F5)](#controlled-feature-ablation-progression-f0-to-f5)
12. [Model Explainability & Permutation Feature Importance](#model-explainability--permutation-feature-importance)
13. [Vectorized Backtesting Protocol & Execution Engine](#vectorized-backtesting-protocol--execution-engine)
14. [Vectorized Backtest Results vs Buy & Hold Benchmark](#vectorized-backtest-results-vs-buy--hold-benchmark)
15. [Transaction Cost Sensitivity & Stress Testing](#transaction-cost-sensitivity--stress-testing)
16. [Live Inference Engine & Real-Time Data Provider](#live-inference-engine--real-time-data-provider)
17. [StockSense Streamlit Web Application Architecture](#stocksense-streamlit-web-application-architecture)
18. [Complete Project Directory Tree & File Inventory](#complete-project-directory-tree--file-inventory)
19. [Installation & Virtual Environment Setup Guide](#installation--virtual-environment-setup-guide)
20. [CLI Command Reference (`main.py`)](#cli-command-reference-mainpy)
21. [Academic Viva Voce Q&A Preparation Summary](#academic-viva-voce-qa-preparation-summary)
22. [Project Documentation & Artifact Inventory](#project-documentation--artifact-inventory)
23. [Limitations, Methodological Takeaways & Future Directions](#limitations-methodological-takeaways--future-directions)
24. [Official Research & Educational Disclaimer](#official-research--educational-disclaimer)

---

## Project Overview & Purpose

This repository houses a financial machine learning research framework built in **Python, PyTorch, Scikit-Learn, Pandas, Plotly, and Streamlit**.

The system predicts **next-day price direction (UP vs DOWN/CASH)** for Indian equity securities on the National Stock Exchange (NSE), evaluated on **Reliance Industries Ltd. (`RELIANCE.NS`)**.

Unlike naive ML tutorials that feed raw Rupee stock prices into oversized networks and apply random train/test splits, this project addresses the fundamental statistical challenges of financial quantitative research:
* **Non-Stationarity:** Stock prices drift upward or downward over multi-year horizons, rendering raw price levels useless for predictive generalization.
* **Low Signal-to-Noise Ratio (SNR):** Daily stock returns are dominated by market noise. Unregularized models quickly overfit to transient fluctuations.
* **Look-Ahead Data Leakage:** Standard feature engineering often accidentally mixes future price information into past rows or fits data scalers across test partitions.
* **Regime Shifts:** Market behavior during bull markets differs radically from bear markets or volatility spikes.

---

## Core Research Question & Theoretical Foundation

### The Core Problem in Financial Machine Learning
Practitioners frequently concatenate dozens of raw technical indicators (`Close`, `Open`, `SMA_200`, `BB_Upper`) into a feature matrix, train a large neural network or tree ensemble, apply a random 80/20 train/test split, and report 85%+ accuracy. **This reported accuracy is a mathematical artifact of data leakage and autocorrelation.** Because stock prices are non-stationary, raw price features drift over time (e.g., RELIANCE trading at ₹500 in 2016 vs ₹1,400 in 2024). A model trained on raw price levels learns nominal price boundaries rather than market dynamics, leading to catastrophic failure out-of-sample.

### Core Research Question
> *"How do different logical groups of scale-invariant technical indicators and market-context features (Trend Ratios, Momentum Oscillators, Volatility Metrics, Volume Ratios, and Macro Volatility Context) impact the out-of-sample predictive capacity, walk-forward stability, and backtested risk-adjusted returns of deep learning models when predicting next-day stock direction?"*

---

## Chronological Problem Diagnostics (Why the 2-Year Baseline Failed)

During initial diagnostics on a 2-year baseline dataset (493 raw rows), we conducted a systematic audit and uncovered five critical methodological flaws:

1. **Extreme Overparameterization:** The original MLP had 14,561 trainable parameters trained on only 205 effective training samples after `SMA_200` warm-up. With 71 times more parameters than training samples, the network collapsed into outputting compressed, constant probabilities between 0.459 and 0.512.
2. **Non-Stationary Features & Severe Multicollinearity:** 15 of the 28 features were raw Rupee prices (`Close`, `Open`, `SMA_20`, `SMA_50`, `SMA_200`, `BB_Upper`, `BB_Lower`). 31 feature pairs had absolute correlations $|r| > 0.90$. Specifically, `SMA_20` and `BB_Middle` were exact duplicates ($r = 1.000$).
3. **Unloaded Market Context:** NIFTY and India VIX context DataFrames were never actually passed into the feature generator (`context_dfs={}`), making Market Context features identical to single-stock features.
4. **Batch Normalization Instability:** Batch Normalization with small batch sizes (~20 batches/epoch) produced highly noisy running mean and variance estimates, causing early stopping after just 4 epochs.
5. **Regime-Shifted Test Partition:** The 44-day test set spanned a sharp bull run (65.9% UP days) following a training set with 49% UP days, creating a severe regime mismatch on a tiny test sample.

---

## The 10-Year Rebuild Strategy (2016–2026)

To resolve these empirical failures, we executed a complete systemic rebuild:

```text
OLD PIPELINE (2-Year Baseline)         NEW REBUILT PIPELINE (10-Year Stationary Framework)
───────────────────────────────         ──────────────────────────────────────────────────
• 493 raw rows (2022–2024)             • {{raw_sessions}} raw rows (2016–2026: RELIANCE + NIFTY + VIX)
• 205 usable training samples          • {{rows}} clean aligned stationary samples
• 15/28 non-stationary Rupee features  • 27/27 stationary, scale-invariant features (0 raw prices)
• 14,561 MLP parameters                • {{n_params}} compact MLP parameters [32, 16]
• Probability collapse (0.459-0.512)   • Test probabilities {{prob_min}}-{{prob_max}} (std {{prob_std}})
• NIFTY/VIX context unpassed           • NIFTY + India VIX context fully integrated
• 44-day test sample (regime shifted)  • {{n_test}}-day out-of-sample test set ({{test_start}} to {{test_end}})
```

---

## Data Leakage Auditing & Mathematical Isolation

To guarantee mathematical integrity, the pipeline enforces strict isolation between features ($X$), targets ($y$), and validation partitions:

```text
Raw Historical Data ({{raw_sessions}} rows: RELIANCE.NS + ^NSEI + ^INDIAVIX)
                       │
                       ▼
Feature Generator ──► Compute 27 Stationary Features at time t
                       │  (Uses ONLY information available at or before t)
                       ▼
Target Generator  ──► Compute Target at time t using Close price at t+1
                       │  Target_Direction_t = 1 if (Close_{t+1} / Close_t - 1) > 0 else 0
                       ▼
Chronological Split ─► Train ({{n_train}} samples | {{train_start}} to {{train_end}})
                       Val   ({{n_val}} samples | {{val_start}} to {{val_end}})
                       Test  ({{n_test}} samples | {{test_start}} to {{test_end}})
                       │
                       ▼
Strict Scaler     ──► StandardScaler FIT ONLY on Train set
                       Transform Val and Test independently
```

### 4-Point Data Leakage Audit Checklist
- [x] **Target Leakage Protection:** Targets (`Next_Day_Return`, `Target_Direction`, `Target_Risk`) are computed separately in `TargetGenerator(df_features, df_prices)` and are physically excluded from feature matrix $X$.
- [x] **No Feature Lookahead:** Every indicator at row $t$ relies on rolling windows strictly ending at $t$ (verified mathematically via unit test `test_no_lookahead_bias`).
- [x] **Strict Scaler Isolation:** `StandardScaler` is fitted exclusively on $X_{train}$. $X_{val}$ and $X_{test}$ are transformed using parameters $(\mu_{train}, \sigma_{train})$.
- [x] **Temporal Isolation:** Train, Validation, and Test partitions follow a hard chronological sequence with zero random shuffling.

---

## Complete 27 Stationary Feature Engineering Matrix

The feature matrix $X$ contains **27 dimensionless, scale-invariant features** grouped into 6 logical categories:

| Group | Feature Name | Mathematical Definition / Formula | Plain-English Description |
|---|---|---|---|
| **F0: Base Returns** | `Return_1d` | $(P_t / P_{t-1}) - 1$ | 1-day percentage price return |
| | `Return_5d` | $(P_t / P_{t-5}) - 1$ | 5-day percentage price return |
| | `Return_10d` | $(P_t / P_{t-10}) - 1$ | 10-day percentage price return |
| | `Return_20d` | $(P_t / P_{t-20}) - 1$ | 20-day percentage price return |
| | `HL_Spread` | $(High_t - Low_t) / Close_t$ | Intraday price trading range ratio |
| | `CO_Spread` | $(Close_t - Open_t) / Open_t$ | Intraday Close-to-Open spread ratio |
| **F1: Trend Ratios** | `Dist_SMA20` | $(Close_t / SMA_{20,t}) - 1$ | Distance to short-term 20-day trend average |
| | `Dist_SMA50` | $(Close_t / SMA_{50,t}) - 1$ | Distance to medium-term 50-day trend average |
| | `Dist_SMA200` | $(Close_t / SMA_{200,t}) - 1$ | Distance to long-term 200-day trend average |
| | `SMA20_50_Ratio` | $(SMA_{20,t} / SMA_{50,t}) - 1$ | Short-to-medium trend crossover ratio |
| | `SMA50_200_Ratio` | $(SMA_{50,t} / SMA_{200,t}) - 1$ | Medium-to-long trend crossover ratio |
| **F2: Momentum** | `RSI_14` | $100 - \left(\frac{100}{1 + RS}\right)$ | 14-day Relative Strength Index (0–100) |
| | `MACD_Ratio` | $(EMA_{12} - EMA_{26}) / Close_t$ | MACD line normalized by Close price |
| | `MACD_Hist_Ratio` | $(MACD\_Line - Signal\_Line) / Close_t$ | MACD histogram normalized by Close price |
| | `ROC_12` | $(Close_t - Close_{t-12}) / Close_{t-12}$ | 12-day price velocity rate of change |
| **F3: Volatility** | `ATR_Ratio` | $ATR_{14,t} / Close_t$ | Average True Range normalized by Close price |
| | `Rolling_Vol_20` | $\text{Std}(R_{1d})_{20} \times \sqrt{252}$ | 20-day annualized rolling volatility |
| | `BB_PctB` | $(Close_t - Lower_t) / (Upper_t - Lower_t)$ | Bollinger %B bandwidth position (0 to 1) |
| | `BB_Width` | $(Upper_t - Lower_t) / Middle_t$ | Bollinger Bandwidth ratio (volatility squeeze) |
| **F4: Volume Ratios** | `Vol_Change` | $(Vol_t / Vol_{t-1}) - 1$ | 1-day percentage volume change ratio |
| | `Vol_Ratio_20` | $Vol_t / SMA_{20}(Vol)_t$ | Daily volume relative to 20-day average volume |
| **F5: Market Context**| `NIFTY_Return_1d` | $(NIFTY_t / NIFTY_{t-1}) - 1$ | 1-day percentage return of NIFTY 50 index |
| | `NIFTY_Return_5d` | $(NIFTY_t / NIFTY_{t-5}) - 1$ | 5-day percentage return of NIFTY 50 index |
| | `NIFTY_Vol_20` | $\text{Std}(R_{NIFTY})_{20} \times \sqrt{252}$ | 20-day rolling volatility of NIFTY 50 |
| | `NIFTY_Dist_SMA50` | $(NIFTY_t / SMA_{50}(NIFTY)_t) - 1$ | NIFTY 50 distance to its 50-day moving average |
| | `VIX_Level` | $VIX_{Close, t}$ | Absolute closing level of India VIX index |
| | `VIX_Change_1d` | $(VIX_t / VIX_{t-1}) - 1$ | 1-day percentage change in India VIX volatility |

---

## Target Formulation & Downside Risk Classification

Targets are constructed strictly using future price information at time $t+1$:

$$\text{Next\_Day\_Return}_t = \frac{P_{Close, t+1}}{P_{Close, t}} - 1$$

$$\text{Target\_Direction}_t = \begin{cases} 1 & \text{if } \text{Next\_Day\_Return}_t > 0 \\ 0 & \text{if } \text{Next\_Day\_Return}_t \le 0 \end{cases}$$

$$\text{Target\_Risk}_t = \begin{cases} 1 & \text{if } \text{Next\_Day\_Return}_t \le -0.02 \\ 0 & \text{if } \text{Next\_Day\_Return}_t > -0.02 \end{cases}$$

---

## Deep Learning Model Architecture & Hyperparameters

### ConfigurableMLP Architecture
```text
Input Layer (27 stationary features)
       │
       ▼
Linear(27 -> 32)  ──►  ReLU  ──►  Dropout(0.20)
       │
       ▼
Linear(32 -> 16)  ──►  ReLU  ──►  Dropout(0.20)
       │
       ▼
Linear(16 -> 1)   ──►  Logits (Binary Output)
```

### Hyperparameter Specifications (`config/settings.yaml`)
- **Architecture Dimensions:** `[32, 16]` (~1,400 trainable parameters)
- **Batch Normalization:** Excluded (avoids noisy batch statistics on small financial sequence batches)
- **Dropout Rate:** 0.20 per hidden layer
- **Optimizer:** Adam ($\text{learning\_rate} = 0.001$, $\text{weight\_decay} = 1\times 10^{-3}$ L2 regularization)
- **Loss Function:** `BCEWithLogitsLoss` with dynamic positive class weighting:
  $$\text{pos\_weight} = \frac{\text{num\_negative\_samples}}{\text{num\_positive\_samples}}$$
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=5, mode='min')
- **Epochs:** 60 max with Early Stopping (patience=10 on validation loss)
- **Batch Size:** 64

---

## Validation Strategy & Timeline Isolation

Financial time series exhibit strong temporal dependence and regime shifts. Random cross-validation is invalid. We use a hard chronological split:

### 1. Hard Chronological Split ({{split_method}})
- **Training Set:** {{n_train}} samples | {{train_start}} to {{train_end}} ({{up_train}} UP days)
- **Validation Set:** {{n_val}} samples | {{val_start}} to {{val_end}} ({{up_val}} UP days)
- **Test Set:** {{n_test}} samples | {{test_start}} to {{test_end}} ({{up_test}} UP days)

> **Walk-forward evaluation:** `src/evaluation/splitter.py` defines 5 expanding-window folds (print them with `python main.py splits`), but no per-fold model evaluation is implemented yet. Walk-forward results previously listed here were removed because no code in the repository produces them.

---

## Empirical Benchmark Comparisons & Out-of-Sample Results

### Out-of-Sample Test Set Performance ({{test_years}}, {{n_test}} days)

{{results_table}}

---

## Statistical Uncertainty (Test Set)

{{uncertainty_block}}

---

## Audit Trail: Previous Run vs Current Run

How the reported results changed when the bugs below were found and fixed. Both columns are rebuilt from saved artifacts by `python main.py metrics`.

{{audit_table}}

---

## Controlled Feature Ablation Progression (F0 to F5)

{{ablation_table}}

---

## Model Explainability & Permutation Feature Importance

Computed by shuffling each feature column and measuring the drop in model ROC-AUC. This shows what the *model* relies on, not what causes price moves.

{{perm_block}}

---

## Vectorized Backtesting Protocol & Execution Engine

### Protocol & Rules
- **Initial Capital:** ₹100,000
{{backtest_rules}}

---

## Vectorized Backtest Results vs Buy & Hold Benchmark

{{backtest_table}}

---

## Transaction Cost Sensitivity & Stress Testing

{{sensitivity_table}}

### No-Trade Band (chosen on validation)

{{band_block}}

{{winrate_note}}

---

## Live Inference Engine & Real-Time Data Provider

The `InferenceEngine` (`src/inference/engine.py`) operates as follows:
1. **2-Year Lookback Fetch:** `YFinanceLiveProvider` fetches 2 years of daily data for `RELIANCE.NS` to allow `SMA_200` indicator warmup.
2. **Context Polling:** Fetches parallel live data for `^NSEI` and `^INDIAVIX`.
3. **Stationary Feature Computation:** Passes buffers to `FeatureGenerator` to generate all 27 stationary features for the latest tick.
4. **Exact Preprocessing:** Loads the saved `scaler.joblib` (fitted on $X_{train}$) and transforms the latest feature vector.
5. **Model Forward Pass:** Passes scaled tensors through saved `.pt` weights to emit direction probability and decision recommendation (`UP` vs `DOWN/CASH`).

---

## StockSense Streamlit Web Application Architecture

Launch the application:
```bash
streamlit run app/dashboard/app.py
```

### Key UI Features
- **Stock Header & Status:** Real-time data session date and status indicators.
- **Next-Day Outlook Hero Card:** Prominent `▲ UP` / `▼ DOWN` badges and probability breakdown.
- **Top Signals Explanation:** Top 5 permutation importance features translated into plain English.
- **Price & Trend Chart:** Plotly chart with 50d/200d SMAs and `3M`, `6M`, `1Y`, `5Y`, `ALL` time range filters.
- **Market Context Panel:** NIFTY 50 benchmark return and India VIX volatility level readings.
- **Backtest Audit & Sensitivity:** Interactive performance comparison and 4-tier friction table.

---

## Complete Project Directory Tree & File Inventory

```text
dl-stock-prediction/
├── app/
│   └── dashboard/
│       └── app.py                      # Streamlit Web Application (StockSense UI)
├── config/
│   ├── logging.yaml                    # Logging configuration
│   └── settings.yaml                   # Global project hyperparameters
├── data/
│   ├── external/                       # External raw data files
│   ├── predictions/                    # Out-of-sample prediction CSV outputs
│   ├── processed/                      # Stationary feature X and target y CSVs
│   └── raw/                            # 10-year raw OHLCV CSVs (RELIANCE, NIFTY, VIX)
├── models/                             # Versioned model registry (.pt & scaler.joblib)
│   └── RELIANCE/
│       └── mlp/                        # Versioned model runs
├── reports/                            # Generated charts, metrics JSONs, ablation outputs
├── src/
│   ├── backtesting/
│   │   └── engine.py                   # Vectorized backtester with friction modeling
│   ├── data/
│   │   ├── ingester.py                 # YFinance defensive data ingester
│   │   └── live_provider.py            # Real-time data provider with context polling
│   ├── evaluation/
│   │   ├── metrics.py                  # Classification evaluation metrics
│   │   ├── plots.py                    # ROC, PR, and training history plotting
│   │   └── splitter.py                 # Time-series chronological & walk-forward splitter
│   ├── experiments/
│   │   └── feature_ablation.py         # F0-F5 feature ablation runner
│   ├── explainability/
│   │   └── explainer.py                # Permutation feature importance explainer
│   ├── features/
│   │   └── generator.py                # 27 stationary feature generation engine
│   ├── inference/
│   │   └── engine.py                   # Real-time live inference engine
│   ├── models/
│   │   ├── dl_models.py                # PyTorch ConfigurableMLP and ConfigurableLSTM
│   │   ├── dl_trainer.py               # Deep learning training loops & registry saving
│   │   └── trainer.py                  # Scikit-Learn baseline model trainer
│   └── targets/
│       └── generator.py                # Isolated target generator with leakage protection
├── tests/                              # Unit test suite (18/18 passing)
│   ├── test_baselines.py
│   ├── test_config.py
│   ├── test_data_provider.py
│   ├── test_features.py
│   ├── test_splitter.py
│   └── test_targets.py
├── main.py                             # Central CLI pipeline entry point
├── requirements.txt                    # Pinned python dependencies
├── VIVA_PREPARATION_GUIDE.md           # Comprehensive viva voce defense guide
└── README.md                           # Master documentation file
```

---

## Installation & Virtual Environment Setup Guide

```bash
# 1. Clone repository
git clone https://github.com/sehersharik/dl-stock-prediction.git
cd dl-stock-prediction

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## CLI Command Reference (`main.py`)

```bash
# Data Ingestion
python main.py ingest --ticker RELIANCE.NS

# Feature Generation (27 Stationary Features)
python main.py features --ticker RELIANCE.NS

# Target Generation
python main.py targets --ticker RELIANCE.NS

# Split Visualization & Walk-Forward Fold Logging
python main.py splits --ticker RELIANCE.NS

# Baseline Model Training
python main.py train --ticker RELIANCE.NS --model logistic
python main.py train --ticker RELIANCE.NS --model rf

# Deep Learning MLP Training
python main.py train --ticker RELIANCE.NS --model mlp

# Feature Ablation Experiment (F0 - F5)
python main.py ablation --ticker RELIANCE.NS

# Model Explainability (Permutation Importance)
python main.py explain --ticker RELIANCE.NS --model mlp

# Backtest Execution
python main.py backtest --ticker RELIANCE.NS --model mlp

# Live Market Inference
python main.py infer --ticker RELIANCE.NS --model mlp

# Run Full Test Suite
python -m pytest tests/ -v
```

---

## Academic Viva Voce Q&A Preparation Summary

1. **Q: Why are raw prices unsuitable for financial deep learning?**  
   *A:* Raw prices are non-stationary and drift over multi-year horizons. Models trained on raw prices learn nominal price thresholds rather than true market dynamics, causing catastrophic failure out-of-sample.

2. **Q: How does your pipeline prevent data leakage?**  
   *A:* Features at $t$ use data available strictly at or before $t$. Targets at $t$ use future return at $t+1$. Data scalers are fitted exclusively on the training partition $X_{train}$.

3. **Q: Why did adding NIFTY and VIX context change {{selection_split}} ROC-AUC from {{abl_prev}} to {{abl_last}}?**  
   *A:* Single-stock indicators lack macro regime context. Market index returns and volatility indices provide essential information regarding overall market direction and risk sentiment.

4. **Q: How should the backtest result be read, given modest direction accuracy?**  
   *A:* {{qa_backtest}}

---

## Project Documentation & Artifact Inventory

- **`VIVA_PREPARATION_GUIDE.md`**: Complete viva defense guide with formulas, Q&A, and code walkthroughs.
- **`reports/RELIANCE.NS_mlp_backtest_metrics.json`**: Official backtest JSON metrics output.
- **`reports/RELIANCE.NS_ablation_results.json`**: Numerical ablation study metrics across all models and feature sets.
- **`reports/RELIANCE.NS_mlp_perm_importance.csv`**: Permutation feature importance rankings.

---

## Limitations, Methodological Takeaways & Future Directions

### Limitations
1. **Macro Regime Drift:** Relationships learned during 2017–2023 training blocks may shift during major economic shocks.
2. **Execution Slippage:** Backtest assumes execution at daily close. Real-world institutional orders incur market impact costs.
3. **Single Asset Focus:** Evaluated primarily on `RELIANCE.NS`. Multi-asset cross-sectional testing remains a future objective.

### Key Research Takeaways
- **Stationarity is Essential:** Non-stationary features destroy neural network generalization.
- **Market Context Provides Signal:** Adding macro market feeds (NIFTY and VIX) gave the largest test-period improvement in the ablation (measured on the period also used for selection).
- **Risk Mitigation Drives Strategy Return:** Sitting in cash during drawdowns generates higher risk-adjusted returns than aggressive trading.

---

## Official Research & Educational Disclaimer

**This software is strictly for academic research and educational purposes.** It does not constitute financial advice, investment advice, or trading recommendations. Past backtest performance does not guarantee future returns. Do not deploy this code in live trading or real-money financial environments.
