# Deep Learning Based Stock Risk and Direction Prediction System

> **A 10-Year Empirical Financial Machine Learning Framework**  
> **Target Asset:** RELIANCE.NS (NSE India) | **Context:** NIFTY 50 (`^NSEI`) & India VIX (`^INDIAVIX`)  
> **Dataset Period:** 2016 – 2026 (2,475 raw trading days → 1,890 aligned stationary samples)  
> **Validated Model:** Small MLP `[32, 16]` with L2 Regularization & Weighted BCE Loss  
> **Status:** 18/18 Unit Tests Passing | GitHub & Streamlit Ready

---

## Table of Contents
1. [Project Overview & Purpose](#project-overview--purpose)
2. [Why This Project Exists & Core Research Question](#why-this-project-exists--core-research-question)
3. [The 10-Year Rebuild: Key Architectural Decisions & Why We Made Them](#the-10-year-rebuild-key-architectural-decisions--why-we-made-them)
4. [Data Pipeline & Leakage Auditing](#data-pipeline--leakage-auditing)
5. [Feature Engineering (27 Stationary Features)](#feature-engineering-27-stationary-features)
6. [Target Formulation & Mathematical Isolation](#target-formulation--mathematical-isolation)
7. [Model Architecture & Hyperparameter Design](#model-architecture--hyperparameter-design)
8. [Validation Strategy (Chronological & 5-Fold Walk-Forward)](#validation-strategy-chronological--5-fold-walk-forward)
9. [Experimental Results & Empirical Benchmark Comparisons](#experimental-results--empirical-benchmark-comparisons)
10. [Feature Ablation Study (F0 to F5 Progression)](#feature-ablation-study-f0-to-f5-progression)
11. [Model Explainability & Feature Importance](#model-explainability--feature-importance)
12. [Vectorized Backtest Audit & Sensitivity Analysis](#vectorized-backtest-audit--sensitivity-analysis)
13. [Live Inference Engine & Data Provider](#live-inference-engine--data-provider)
14. [Streamlit Interactive Dashboard](#streamlit-interactive-dashboard)
15. [Project Structure & Installation](#project-structure--installation)
16. [CLI Command Reference](#cli-command-reference)
17. [Project Documents & Artifacts Reference](#project-documents--artifacts-reference)
18. [Limitations, Research Takeaways & Future Work](#limitations-research-takeaways--future-work)
19. [Disclaimer](#disclaimer)

---

## Project Overview & Purpose

This repository contains a production-grade, end-to-end financial machine learning research framework built using **Python, PyTorch, Scikit-Learn, Pandas, and Streamlit**.

The system predicts **next-day price direction (UP vs DOWN/CASH)** for stock tickers on the National Stock Exchange of India (NSE), specifically evaluated on **Reliance Industries Ltd. (`RELIANCE.NS`)**. Unlike standard machine learning tutorials that use raw price levels and random data splits, this project addresses the fundamental statistical realities of financial markets: **non-stationarity, extreme noise, low signal-to-noise ratio, regime shifts, and look-ahead data leakage.**

### In 30 Seconds: What This Project Does
1. Downloads 10 years (2016–2026) of OHLCV data for RELIANCE.NS, NIFTY 50, and India VIX.
2. Transforms raw nominal price levels into **27 scale-invariant, stationary features** (ratios, returns, normalized oscillators, and market context).
3. Chronologically splits data into **70% Training (2017–2023), 15% Validation (2023–2024), and 15% Untouched Testing (2024–2026)**.
4. Trains a compact **PyTorch Multilayer Perceptron (MLP)** using class-weighted binary cross-entropy loss, dropout, and L2 weight decay.
5. Performs **5-Fold Walk-Forward Cross-Validation**, **Feature Ablation (F0–F5)**, and **Permutation Feature Importance**.
6. Simulates a **vectorized financial backtest** with transaction costs (0.10%) and slippage (0.05%).
7. Serves predictions in real time via a **Live Inference Engine** and an interactive **Streamlit Dashboard**.

---

## Why This Project Exists & Core Research Question

### The Core Problem in Financial Machine Learning
Practitioners often concatenate raw stock prices (`Close`, `Open`, `SMA_200`, `BB_Upper`) into a feature matrix, train a large neural network or tree model, run a random train/test split, and report 80%+ accuracy. **This accuracy is a mathematical illusion caused by data leakage and non-stationarity.** Raw price levels drift over time (e.g., RELIANCE trading at ₹500 in 2016 vs ₹1,400 in 2024). A model trained on raw prices learns nominal price thresholds rather than true market dynamics, causing catastrophic collapse when deployed out-of-sample.

### Core Research Question
> *"How do different groups of scale-invariant technical indicators and market-context features (Trend, Momentum, Volatility, Volume, and Macro Volatility Context) impact the out-of-sample predictive performance, walk-forward stability, and backtested risk-adjusted returns of deep learning models when predicting next-day stock direction?"*

---

## The 10-Year Rebuild: Key Architectural Decisions & Why We Made Them

During initial diagnostics on a 2-year dataset (493 raw rows), we discovered critical methodological flaws:
1. **Severe Overparameterization:** The initial MLP had 14,561 parameters trained on only 205 effective samples after `SMA_200` warm-up. This caused probability output collapse (probabilities compressed between 0.459 and 0.512).
2. **Non-Stationary Features & Multicollinearity:** 15 out of 28 features were raw price levels in Rupees with 31 feature pairs showing correlation $|r| > 0.90$ (e.g., `SMA_20` and `BB_Middle` were exact duplicates with $r=1.000$).
3. **Unloaded Market Context:** NIFTY and India VIX context data were never actually passed to feature generators, rendering market context features identical to single-stock features.
4. **BatchNorm Collapse:** Batch Normalization with small batch sizes (~20 batches/epoch) generated noisy running statistics, triggering early stopping after just 4 epochs.

### The Solution: Complete Pipeline Rebuild
We executed a systemic rebuild of the dataset and feature engineering pipeline:
* **Expanded Dataset Horizon:** Switched from 2 years to **10 years (2016–2026)**, expanding raw rows from 493 to **2,475**, producing **1,890 clean, aligned samples** after a 200-day indicator warm-up.
* **100% Stationary Transformation:** Completely eliminated all nominal Rupee price levels from the feature matrix $X$. Replaced them with dimensionless returns, moving-average distance ratios, normalized volatility metrics, and percentage bandwidths.
* **Integrated Market Context:** Ingested 10-year parallel historical data for `^NSEI` (NIFTY 50) and `^INDIAVIX` (India VIX) to calculate relative market return and regime volatility features.
* **Compact Neural Architecture:** Reduced MLP capacity to `[32, 16]` (approx. 1,400 parameters), removed Batch Normalization, added L2 weight decay ($1\times 10^{-3}$), and applied class-weighted BCE loss.

---

## Data Pipeline & Leakage Auditing

To guarantee scientific validity, the pipeline enforces strict isolation between features ($X$) and targets ($y$):

```text
Raw Historical Data (2,475 rows: RELIANCE + NIFTY + VIX)
                       │
                       ▼
Feature Generator ──► Compute 27 Stationary Features at time t
                       │  (Uses ONLY information available at or before t)
                       ▼
Target Generator  ──► Compute Target at time t using Close price at t+1
                       │  Target_Direction_t = 1 if (Close_{t+1} / Close_t - 1) > 0 else 0
                       ▼
Chronological Split ─► Train (70%: 1,323 samples | 2017-07-27 to 2023-10-19)
                       Val   (15%:   283 samples | 2023-10-20 to 2024-12-20)
                       Test  (15%:   284 samples | 2024-12-23 to 2026-09-10)
                       │
                       ▼
Strict Scaler     ──► StandardScaler FIT ONLY on Train set
                       Transform Val and Test independently
```

### Data Leakage Audit Checklist
- [x] **No Target Leakage:** Targets (`Next_Day_Return`, `Target_Direction`) are computed from raw prices in `TargetGenerator` and are physically excluded from feature matrix $X$.
- [x] **No Feature Lookahead:** Every indicator at row $t$ uses rolling windows strictly ending at $t$ (verified via `test_no_lookahead_bias`).
- [x] **No Scaler Leakage:** `StandardScaler` is fitted exclusively on $X_{train}$. $X_{val}$ and $X_{test}$ are transformed using train parameters.
- [x] **No Temporal Overlap:** Train, Validation, and Test sets follow a hard chronological sequence with zero random shuffling.

---

## Feature Engineering (27 Stationary Features)

The feature matrix $X$ contains **27 dimensionless, scale-invariant features** grouped into 6 logical categories:

| Group | Feature Name | Mathematical Definition / Description |
|---|---|---|
| **F0: Base Returns** | `Return_1d` | 1-day percentage price return: $(P_t / P_{t-1}) - 1$ |
| | `Return_5d` | 5-day percentage price return |
| | `Return_10d` | 10-day percentage price return |
| | `Return_20d` | 20-day percentage price return |
| | `HL_Spread` | Daily High-Low spread ratio: $(High_t - Low_t) / Close_t$ |
| | `CO_Spread` | Daily Close-Open spread ratio: $(Close_t - Open_t) / Open_t$ |
| **F1: Trend Ratios** | `Dist_SMA20` | Normalized distance to 20-day SMA: $(Close_t / SMA_{20,t}) - 1$ |
| | `Dist_SMA50` | Normalized distance to 50-day SMA: $(Close_t / SMA_{50,t}) - 1$ |
| | `Dist_SMA200` | Normalized distance to 200-day SMA: $(Close_t / SMA_{200,t}) - 1$ |
| | `SMA20_50_Ratio` | Fast/Medium trend ratio: $(SMA_{20,t} / SMA_{50,t}) - 1$ |
| | `SMA50_200_Ratio` | Medium/Slow trend ratio: $(SMA_{50,t} / SMA_{200,t}) - 1$ |
| **F2: Momentum** | `RSI_14` | 14-day Relative Strength Index (0–100 scale) |
| | `MACD_Ratio` | MACD Line normalized by Close price: $(EMA_{12} - EMA_{26}) / Close_t$ |
| | `MACD_Hist_Ratio` | MACD Histogram normalized by Close price |
| | `ROC_12` | 12-day Rate of Change percentage |
| **F3: Volatility** | `ATR_Ratio` | Average True Range normalized by Close price: $ATR_{14,t} / Close_t$ |
| | `Rolling_Vol_20` | 20-day annualized rolling volatility of 1-day returns |
| | `BB_PctB` | Bollinger Percent Bandwidth: $(Close_t - Lower_t) / (Upper_t - Lower_t)$ |
| | `BB_Width` | Bollinger Bandwidth ratio: $(Upper_t - Lower_t) / Middle_t$ |
| **F4: Volume Ratios** | `Vol_Change` | 1-day percentage volume change: $(Vol_t / Vol_{t-1}) - 1$ |
| | `Vol_Ratio_20` | Volume relative to 20-day average volume: $Vol_t / SMA_{20}(Vol)_t$ |
| **F5: Market Context** | `NIFTY_Return_1d` | 1-day return of NIFTY 50 index |
| | `NIFTY_Return_5d` | 5-day return of NIFTY 50 index |
| | `NIFTY_Vol_20` | 20-day rolling volatility of NIFTY 50 |
| | `NIFTY_Dist_SMA50` | NIFTY 50 distance to its own 50-day SMA |
| | `VIX_Level` | Absolute closing level of India VIX index |
| | `VIX_Change_1d` | 1-day percentage change in India VIX |

---

## Target Formulation & Mathematical Isolation

To prevent target leakage into features, `TargetGenerator` takes `df_features` ($X$) and `df_prices` ($P$) as separate inputs:

$$\text{Next\_Day\_Return}_t = \frac{P_{Close, t+1}}{P_{Close, t}} - 1$$

$$\text{Target\_Direction}_t = \begin{cases} 1 & \text{if } \text{Next\_Day\_Return}_t > 0 \\ 0 & \text{if } \text{Next\_Day\_Return}_t \le 0 \end{cases}$$

$$\text{Target\_Risk}_t = \begin{cases} 1 & \text{if } \text{Next\_Day\_Return}_t \le -0.02 \\ 0 & \text{if } \text{Next\_Day\_Return}_t > -0.02 \end{cases}$$

*The last observation in the dataset is dropped during training because its $t+1$ close price is unknown.*

---

## Model Architecture & Hyperparameter Design

### ConfigurableMLP Architecture
```text
Input Layer (27 stationary features)
       │
       ▼
Linear(27 -> 32)  ──►  ReLU  ──►  Dropout(0.2)
       │
       ▼
Linear(32 -> 16)  ──►  ReLU  ──►  Dropout(0.2)
       │
       ▼
Linear(16 -> 1)   ──►  Logits (Binary Output)
```

### Hyperparameter Configuration (`config/settings.yaml`)
- **Optimizer:** Adam ($\text{learning\_rate} = 0.001$, $\text{weight\_decay} = 1\times 10^{-3}$)
- **Loss Function:** `BCEWithLogitsLoss` with dynamic `pos_weight` ($\text{num\_neg} / \text{num\_pos}$) to handle class imbalance
- **Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=5)
- **Epochs:** 60 max with Early Stopping (patience=10 on validation loss)
- **Batch Size:** 64
- **Normalization:** Standard scaling fitted **only** on training data. No BatchNorm layers in network to avoid noisy batch statistics.

---

## Validation Strategy (Chronological & 5-Fold Walk-Forward)

Financial time series exhibit strong temporal dependence. Random k-fold cross-validation is invalid. We employ two complementary validation schemes:

### 1. Hard Chronological Split (70 / 15 / 15)
- **Training Set (70%):** 1,323 samples | 2017-07-27 to 2023-10-19 (Class balance: 51.9% UP)
- **Validation Set (15%):** 283 samples | 2023-10-20 to 2024-12-20 (Class balance: 50.9% UP)
- **Test Set (15%):** 284 samples | 2024-12-23 to 2026-09-10 (Class balance: 50.0% UP)

### 2. Expanding-Window Walk-Forward Validation (5 Folds)
To test model stability across changing market regimes, an expanding window initial training set (50% of data) is evaluated across 5 sequential out-of-sample folds:

```text
Fold 1 | Train: 2017-07-27 to 2022-02-18 | Val: 2022-02-21 to 2023-01-13 (189 days)
Fold 2 | Train: 2017-07-27 to 2023-01-13 | Val: 2023-01-16 to 2023-10-19 (189 days)
Fold 3 | Train: 2017-07-27 to 2023-10-19 | Val: 2023-10-20 to 2024-07-25 (189 days)
Fold 4 | Train: 2017-07-27 to 2024-07-25 | Val: 2024-07-26 to 2025-05-05 (189 days)
Fold 5 | Train: 2017-07-27 to 2025-05-05 | Val: 2025-05-06 to 2026-09-10 (191 days)
```

---

## Experimental Results & Empirical Benchmark Comparisons

### Out-of-Sample Test Set Performance (2024–2026, 284 days)

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | 50.00% | 50.00% | 50.00% | 100.0% | 0.6667 | 0.5000 | 0.5000 |
| **Logistic Regression** | 50.35% | 50.35% | 0.5035 | 0.5070 | 0.5052 | 0.5143 | 0.5098 |
| **Random Forest Baseline** | 52.82% | 52.82% | 0.5310 | 0.4789 | 0.5037 | 0.5414 | 0.5210 |
| **Small MLP [32, 16] (Final)** | **57.39%** | **57.39%** | **0.5868** | **0.5000** | **0.5399** | **0.5676** | **0.5301** |
| **LSTM [32, 16] (Sequence 30)** | 48.94% | 48.94% | 0.4851 | 0.3169 | 0.3834 | 0.4890 | 0.4881 |

### 5-Fold Walk-Forward Cross-Validation Results
- **Small MLP [32, 16]:** Mean Walk-Forward ROC-AUC = **0.5491 ± 0.010** *(Highest consistency across all 5 folds)*
- **Random Forest:** Mean Walk-Forward ROC-AUC = **0.5414 ± 0.031**
- **Logistic Regression:** Mean Walk-Forward ROC-AUC = **0.5143 ± 0.037**

### Key Finding
The **Small MLP** outperformed traditional baselines and sequential LSTM models. The LSTM struggled due to sequence length truncation on limited temporal samples (~1,300 training sequences), whereas the compact MLP effectively generalized across stationary cross-sectional features.

---

## Feature Ablation Study (F0 to F5 Progression)

To evaluate the predictive contribution of each feature group, we trained MLP and LSTM models across cumulative feature subsets:

| Feature Level | Feature Groups Included | Total Features | MLP ROC-AUC | MLP Accuracy | MLP F1 | LSTM ROC-AUC |
|---|---|---|---|---|---|---|
| **F0 (Base)** | Returns + Spreads | 6 | 0.4683 | 46.83% | 0.4702 | 0.4592 |
| **F1 (Trend)** | F0 + Moving Average Ratios | 11 | 0.4873 | 51.06% | 0.5775 | 0.5031 |
| **F2 (Momentum)**| F1 + RSI, MACD, ROC | 15 | 0.4829 | 45.42% | 0.5260 | 0.4588 |
| **F3 (Volatility)**| F2 + ATR, Volatility, Bollinger | 19 | 0.5148 | 49.30% | 0.6604 | 0.4835 |
| **F4 (Volume)** | F3 + Volume Changes & Ratios | 21 | 0.4964 | 48.59% | 0.4786 | 0.4747 |
| **F5 (Context)**| F4 + NIFTY 50 & India VIX | **27** | **0.5676** | **57.39%** | **0.5399** | 0.4890 |

### Research Conclusion from Ablation
Neither base returns (F0) nor single-stock indicators (F1–F4) provided sufficient signal for out-of-sample neural network generalization. Model performance broke out significantly only upon adding **Market Context (F5)** — incorporating NIFTY index movements and macro market volatility (India VIX) pushed ROC-AUC from 0.496 to **0.5676**.

---

## Model Explainability & Feature Importance

Permutation Feature Importance was computed on the untouched test set by shuffling each feature column and measuring the drop in model ROC-AUC score:

```text
Top Features by Permutation Importance (Drop in Test ROC-AUC):

1. Dist_SMA200       [+0.0337]  (Normalized distance to 200-day moving average)
2. NIFTY_Return_1d   [+0.0204]  (1-day NIFTY 50 market index return)
3. VIX_Change_1d     [+0.0160]  (1-day percentage change in India VIX volatility)
4. RSI_14            [+0.0126]  (14-day Relative Strength Index)
5. ROC_12            [+0.0119]  (12-day Rate of Change)
6. NIFTY_Return_5d   [+0.0079]  (5-day cumulative NIFTY market return)
7. Vol_Change        [+0.0076]  (1-day volume change ratio)
8. Return_1d         [+0.0073]  (1-day asset price return)
9. MACD_Ratio        [+0.0072]  (Normalized MACD line)
10. CO_Spread        [+0.0071]  (Close-to-Open daily spread)
```

> **Interpretation:** Long-term trend positioning (`Dist_SMA200`) and immediate market context (`NIFTY_Return_1d`, `VIX_Change_1d`) carry the strongest decision weight in the neural network's predictions.

---

## Vectorized Backtest Audit & Sensitivity Analysis

### Backtest Protocol
- **Out-of-Sample Period:** 284 trading days (2024-12-23 to 2026-09-10)
- **Initial Capital:** ₹100,000
- **Signal Threshold:** $\text{Probability} \ge 0.50 \implies \text{LONG (1)}$, else $\text{CASH (0)}$
- **Transaction Costs:** 0.10% per position change
- **Slippage Friction:** 0.05% per position change (Total friction: **0.15% per trade**)
- **Execution Timing:** Decision made at Close $T$ using features through $T$; return earned from Close $T$ to Close $T+1$.

### Official Backtest Performance Metrics

| Metric | Strategy (MLP [32, 16]) | Buy & Hold Benchmark |
|---|---|---|
| **Cumulative Total Return** | **+23.80%** | +5.13% |
| **Annualized Return** | **+20.78%** | +4.53% |
| **Annualized Volatility** | **11.45%** | 16.88% |
| **Sharpe Ratio** | **1.45** | 0.27 |
| **Maximum Drawdown** | **-5.27%** | -16.84% |
| **Win Rate** | **55.37%** (67 wins / 121 active days) | N/A |
| **Executed Trades** | **64** (32 Long entries, 32 Cash exits) | 1 (Buy at start) |
| **Time in Market** | **42.6%** (121 Long days, 162 Cash days) | 100.0% |

### Transaction Cost Sensitivity Analysis
To confirm that the strategy's profitability is robust and not an artifact of low friction assumptions, we stress-tested the backtest under four cost regimes:

| Cost Regime | Friction per Trade | Cumulative Return | Sharpe Ratio | Max Drawdown | Win Rate |
|---|---|---|---|---|---|
| **Zero Costs (Theoretical)** | 0.00% | **+36.25%** | 2.06 | -4.79% | 58.68% |
| **Default Friction (Official)** | **0.15%** (0.10% cost + 0.05% slip) | **+23.80%** | **1.45** | **-5.27%** | **55.37%** |
| **High Friction** | 0.25% (0.20% cost + 0.05% slip) | **+16.13%** | 1.04 | -6.80% | 54.55% |
| **Severe Stress Test** | 0.30% (0.20% cost + 0.10% slip) | **+12.47%** | 0.83 | -8.08% | 52.89% |

> **Key Audit Takeaway:** The strategy remains profitable and continues to beat Buy & Hold (+5.13%) even under severe 0.30% friction. The model's edge stems from **risk mitigation** — staying in cash during major market drawdowns (162 cash days) rather than chasing aggressive long leverage.

---

## Live Inference Engine & Data Provider

The project includes an independent `InferenceEngine` (`src/inference/engine.py`) and `YFinanceLiveProvider` (`src/data/live_provider.py`):

1. **2-Year Lookback Buffer:** Live provider fetches 2 years of daily data for the target stock to allow proper `SMA_200` warmup.
2. **Context Polling:** Fetches parallel live context feeds for `^NSEI` and `^INDIAVIX`.
3. **Stationary Transformation:** Applies identical `FeatureGenerator` rules to produce the 27 stationary features for the latest market tick.
4. **Exact Scaler Pipeline:** Loads the saved `scaler.joblib` (fitted on training data) and applies exact scaling.
5. **Model Forward Pass:** Pushes scaled features through saved `.pt` weights to emit direction probability and directional recommendation (`UP` vs `DOWN/CASH`).

---

## Streamlit Interactive Dashboard

An interactive dashboard (`app/dashboard/app.py`) provides progressive disclosure of all model outputs:

- **Executive Summary:** Live prediction, confidence score, and model status in plain English.
- **Model Diagnostic Audit:** Confusion matrix, precision/recall curves, and ROC-AUC metrics.
- **Feature Ablation Viewer:** Interactive plots showing feature group contributions (F0 through F5).
- **Backtest Analysis:** Equity curves, drawdown charts, and cost sensitivity comparisons.
- **Live Predictor Panel:** Single-click live tick fetching and inference execution.

To launch the dashboard locally:
```bash
streamlit run app/dashboard/app.py
```

---

## Project Structure & Installation

```text
dl-stock-prediction/
├── app/
│   └── dashboard/
│       └── app.py                      # Interactive Streamlit Web Application
├── config/
│   ├── logging.yaml                    # Logging configuration
│   └── settings.yaml                   # Global project hyperparameter configuration
├── data/
│   ├── predictions/                    # Out-of-sample test prediction CSVs
│   ├── processed/                      # Stationary feature X and target y datasets
│   └── raw/                            # 10-year raw OHLCV CSVs (RELIANCE, NIFTY, VIX)
├── models/                             # Structured registry for saved .pt models and scalers
│   └── RELIANCE/
│       └── mlp/                        # Versioned model run directories
├── reports/                            # Generated plots, metrics JSONs, and ablation outputs
├── src/
│   ├── backtesting/
│   │   └── engine.py                   # Vectorized backtester with friction modeling
│   ├── data/
│   │   ├── ingester.py                 # YFinance defensive data downloader
│   │   └── live_provider.py            # Real-time data provider with context polling
│   ├── evaluation/
│   ├── experiments/
│   │   └── feature_ablation.py         # Controlled F0-F5 ablation experiment runner
│   ├── explainability/
│   │   └── explainer.py                # Permutation importance and SHAP analysis
│   ├── features/
│   │   └── generator.py                # 27 stationary feature generation engine
│   ├── inference/
│   │   └── engine.py                   # Live inference engine
│   ├── models/
│   │   ├── dl_models.py                # PyTorch ConfigurableMLP and ConfigurableLSTM
│   │   ├── dl_trainer.py               # Training loops, early stopping, registry saving
│   │   └── trainer.py                  # Scikit-Learn baseline model trainer
│   └── targets/
│       └── generator.py                # Target formulation with leakage protection
├── tests/                              # Unit test suite (18 tests passing)
│   ├── test_baselines.py
│   ├── test_config.py
│   ├── test_data_provider.py
│   ├── test_features.py
│   ├── test_splitter.py
│   └── test_targets.py
├── main.py                             # Central CLI pipeline entry point
├── requirements.txt                    # Pinned project dependencies
├── VIVA_PREPARATION_GUIDE.md           # Complete academic viva Q&A preparation document
└── README.md                           # This master documentation file
```

### Installation
```bash
# 1. Clone repository
git clone https://github.com/sehersharik/dl-stock-prediction.git
cd dl-stock-prediction

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install pinned dependencies
pip install -r requirements.txt
```

---

## CLI Command Reference

The central CLI (`main.py`) controls the full pipeline lifecycle:

```bash
# 1. Download 10-year raw price data (Target, NIFTY, India VIX)
python main.py ingest --ticker RELIANCE.NS

# 2. Generate 27 stationary features
python main.py features --ticker RELIANCE.NS

# 3. Formulate binary direction and risk targets
python main.py targets --ticker RELIANCE.NS

# 4. Generate & plot chronological split & walk-forward folds
python main.py splits --ticker RELIANCE.NS

# 5. Train baseline models (Logistic Regression, Random Forest)
python main.py train --ticker RELIANCE.NS --model logistic
python main.py train --ticker RELIANCE.NS --model rf

# 6. Train Deep Learning MLP model
python main.py train --ticker RELIANCE.NS --model mlp

# 7. Run feature ablation experiment (F0 through F5)
python main.py ablation --ticker RELIANCE.NS

# 8. Compute Permutation Importance explainability
python main.py explain --ticker RELIANCE.NS --model mlp

# 9. Execute vectorized backtest with transaction costs
python main.py backtest --ticker RELIANCE.NS --model mlp

# 10. Run live market inference
python main.py infer --ticker RELIANCE.NS --model mlp

# 11. Run full test suite
python -m pytest tests/ -v
```

---

## Project Documents & Artifacts Reference

This repository includes detailed supplementary documentation:
- **`VIVA_PREPARATION_GUIDE.md`**: Complete, highly detailed academic project defense document containing a 30-second elevator pitch, methodological rationale, mathematical formulas, likely viva questions with answers, and step-by-step code walkthroughs.
- **`reports/RELIANCE.NS_mlp_backtest_metrics.json`**: Official JSON output of backtest performance metrics.
- **`reports/RELIANCE.NS_ablation_results.json`**: Complete numerical ablation study metrics across all models and feature groups.
- **`reports/RELIANCE.NS_mlp_perm_importance.csv`**: Permutation feature importance rankings.

---

## Limitations, Research Takeaways & Future Work

### Limitations
1. **Market Non-Stationarity:** Relationships learned over 2017–2023 may degrade during regime shifts (e.g., sudden geopolitical events or central bank interest rate shocks).
2. **Slippage & Execution:** Backtesting assumes execution at the exact daily closing price. Real-world institutional orders incur market impact and execution delays.
3. **Single Asset Focus:** Evaluated primarily on `RELIANCE.NS`. Multi-asset generalization across mid-cap or high-volatility stocks requires further cross-sectional testing.

### Core Research Takeaways
- **Stationarity is Non-Negotiable:** Raw price features cause neural network failure. Stationary ratios and relative distances are essential for financial deep learning.
- **Market Context Drives Edge:** Single-stock technical indicators (F0–F4) achieved max ROC-AUC of ~0.514. Incorporating broader market index returns and volatility (F5) improved ROC-AUC to **0.5676**.
- **Risk Avoidance Beats Aggressive Trading:** The model's outperformance (+23.80% vs +5.13%) comes from avoiding major drawdown periods by remaining in cash 57.4% of the time.

### Future Work
1. **Multi-Task Neural Networks:** Building a joint architecture predicting Next-Day Direction and Downside Risk simultaneously via a shared representation layer.
2. **Time-Series Attention Transformers:** Testing Temporal Fusion Transformers (TFT) on multi-scale stationary inputs.
3. **Dynamic Thresholding:** Replacing the static 0.50 probability cutoff with a dynamic volatility-adjusted threshold.

---

## Disclaimer

**This software is for research and educational purposes only.** It does not constitute financial advice, investment advice, or trading recommendations. Historical backtest performance does not guarantee future returns. Do not deploy this code in live trading or real-money environments.
