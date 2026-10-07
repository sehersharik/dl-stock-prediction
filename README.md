# Deep Learning Based Stock Risk and Next-Day Direction Prediction System

> **A 10-Year Empirical Financial Machine Learning Framework**  
> **Target Asset:** RELIANCE Industries Ltd. (`RELIANCE.NS`) on National Stock Exchange of India (NSE)  
> **Market Context Assets:** NIFTY 50 Index (`^NSEI`) & India VIX Volatility Index (`^INDIAVIX`)  
> **Dataset Period:** October 2016 – October 2026 (2,475 raw daily sessions → 1,890 aligned stationary samples)  
> **Validated Deep Learning Architecture:** Compact PyTorch MLP `[32, 16]` with L2 Regularization & Class-Weighted BCE Loss  
> **Verification Status:** 18/18 Unit Tests Passing | Out-of-Sample Test Backtest +23.80% vs +5.13% Buy & Hold (Sharpe 1.45)

---

## Executive Summary & System Metadata

| Parameter | Value / Specification | Description / Justification |
|---|---|---|
| **Target Asset** | `RELIANCE.NS` | Major liquid blue-chip constituent of NIFTY 50 |
| **Context Feeds** | `^NSEI` (NIFTY 50) & `^INDIAVIX` (India VIX) | Parallel macro market return & volatility feeds |
| **Data Horizon** | 10 Years (2016 – 2026) | 2,475 raw trading sessions |
| **Effective Dataset Size** | 1,890 Aligned Observations | After 200-day indicator warmup and missing target drop |
| **Feature Matrix ($X$)** | 27 Stationary Dimensionless Features | Returns, ratios, normalized momentum, volatility & macro context |
| **Target Vector ($y$)** | Next-Day Direction $y_t \in \{0, 1\}$ | $y_t = 1$ if $(P_{t+1}/P_t - 1) > 0$ else $0$ |
| **Train / Val / Test Split** | Chronological 70% / 15% / 15% | Train: 1,323 (2017–2023) \| Val: 283 (2023–2024) \| Test: 284 (2024–2026) |
| **Cross-Validation** | 5-Fold Expanding Window Walk-Forward | Evaluated on 189-day sequential folds |
| **Model Architecture** | PyTorch Small MLP `[32, 16]` | Input (27) $\to$ Linear(32) $\to$ ReLU $\to$ Drop(0.2) $\to$ Linear(16) $\to$ ReLU $\to$ Drop(0.2) $\to$ Out(1) |
| **Optimizer & Loss** | Adam ($\text{lr}=10^{-3}, \text{weight\_decay}=10^{-3}$) | `BCEWithLogitsLoss` with dynamic class `pos_weight` |
| **Out-of-Sample ROC-AUC** | **0.5676** | Untouched test set (284 trading days, 2024–2026) |
| **Out-of-Sample Balanced Acc**| **57.39%** | Precision: 58.68% \| Recall: 50.00% \| F1: 0.5399 |
| **Backtest Return (0.15% Cost)**| **+23.80%** (vs **+5.13%** Buy & Hold) | Sharpe: **1.45** \| Max Drawdown: **-5.27%** \| Win Rate: **55.37%** (64 trades) |
| **Test Suite Status** | **18 / 18 Tests Passing** | Includes leakage tests, stationary column tests & splitter tests |

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
• 493 raw rows (2022–2024)             • 2,475 raw rows (2016–2026: RELIANCE + NIFTY + VIX)
• 205 usable training samples          • 1,890 clean aligned stationary samples
• 15/28 non-stationary Rupee features  • 27/27 stationary, scale-invariant features (0 raw prices)
• 14,561 MLP parameters                • 1,400 compact MLP parameters [32, 16]
• Probability collapse (0.459-0.512)   • Well-distributed probabilities (0.461-0.654, std 0.028)
• NIFTY/VIX context unpassed           • NIFTY + India VIX context fully integrated
• 44-day test sample (regime shifted)  • 284-day out-of-sample test set (2024-12-23 to 2026-09-10)
```

---

## Data Leakage Auditing & Mathematical Isolation

To guarantee mathematical integrity, the pipeline enforces strict isolation between features ($X$), targets ($y$), and validation partitions:

```text
Raw Historical Data (2,475 rows: RELIANCE.NS + ^NSEI + ^INDIAVIX)
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

Financial time series exhibit strong temporal dependence and regime shifts. Random cross-validation is invalid. We employ two complementary validation schemes:

### 1. Hard Chronological Split (70 / 15 / 15)
- **Training Set (70%):** 1,323 samples | 2017-07-27 to 2023-10-19 (51.9% UP days)
- **Validation Set (15%):** 283 samples | 2023-10-20 to 2024-12-20 (50.9% UP days)
- **Test Set (15%):** 284 samples | 2024-12-23 to 2026-09-10 (50.0% UP days)

### 2. Expanding-Window Walk-Forward Validation (5 Folds)
An initial 50% training set expands sequentially across 5 out-of-sample evaluation folds:
- **Fold 1:** Train 2017-07-27 to 2022-02-18 | Val 2022-02-21 to 2023-01-13 (189 days)
- **Fold 2:** Train 2017-07-27 to 2023-01-13 | Val 2023-01-16 to 2023-10-19 (189 days)
- **Fold 3:** Train 2017-07-27 to 2023-10-19 | Val 2023-10-20 to 2024-07-25 (189 days)
- **Fold 4:** Train 2017-07-27 to 2024-07-25 | Val 2024-07-26 to 2025-05-05 (189 days)
- **Fold 5:** Train 2017-07-27 to 2025-05-05 | Val 2025-05-06 to 2026-09-10 (191 days)

---

## Empirical Benchmark Comparisons & Out-of-Sample Results

### Out-of-Sample Test Set Performance (2024–2026, 284 days)

| Model Architecture | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| **Majority Class Baseline** | 50.00% | 50.00% | 50.00% | 100.0% | 0.6667 | 0.5000 | 0.5000 |
| **Logistic Regression** | 50.35% | 50.35% | 50.35% | 50.70% | 0.5052 | 0.5143 | 0.5098 |
| **Random Forest Baseline** | 52.82% | 52.82% | 53.10% | 47.89% | 0.5037 | 0.5414 | 0.5210 |
| **Small MLP [32, 16] (Final)** | **57.39%** | **57.39%** | **58.68%** | **50.00%** | **0.5399** | **0.5676** | **0.5301** |
| **LSTM [32, 16] (Seq 30)** | 48.94% | 48.94% | 48.51% | 31.69% | 0.3834 | 0.4890 | 0.4881 |

### 5-Fold Walk-Forward Cross-Validation Performance
- **Small MLP [32, 16]:** Mean Walk-Forward ROC-AUC = **0.5491 ± 0.010** *(Most consistent architecture)*
- **Random Forest:** Mean Walk-Forward ROC-AUC = **0.5414 ± 0.031**
- **Logistic Regression:** Mean Walk-Forward ROC-AUC = **0.5143 ± 0.037**

---

## Controlled Feature Ablation Progression (F0 to F5)

| Feature Level | Feature Groups Included | Total Features | MLP ROC-AUC | MLP Accuracy | MLP F1 | LSTM ROC-AUC |
|---|---|---|---|---|---|---|
| **F0 (Base)** | Returns + Spreads | 6 | 0.4683 | 46.83% | 0.4702 | 0.4592 |
| **F1 (Trend)** | F0 + Moving Average Ratios | 11 | 0.4873 | 51.06% | 0.5775 | 0.5031 |
| **F2 (Momentum)**| F1 + RSI, MACD, ROC | 15 | 0.4829 | 45.42% | 0.5260 | 0.4588 |
| **F3 (Volatility)**| F2 + ATR, Volatility, Bollinger | 19 | 0.5148 | 49.30% | 0.6604 | 0.4835 |
| **F4 (Volume)** | F3 + Volume Changes & Ratios | 21 | 0.4964 | 48.59% | 0.4786 | 0.4747 |
| **F5 (Context)**| F4 + NIFTY 50 & India VIX | **27** | **0.5676** | **57.39%** | **0.5399** | 0.4890 |

---

## Model Explainability & Permutation Feature Importance

Computed by shuffling each feature column on the untouched test set and measuring the drop in model ROC-AUC:

```text
Top Features Ranked by Drop in Test ROC-AUC:

1. Dist_SMA200       [+0.0337]  (Normalized distance to 200-day moving average)
2. NIFTY_Return_1d   [+0.0204]  (1-day NIFTY 50 market index return)
3. VIX_Change_1d     [+0.0160]  (1-day percentage change in India VIX volatility)
4. RSI_14            [+0.0126]  (14-day Relative Strength Index)
5. ROC_12            [+0.0119]  (12-day Rate of Change momentum)
6. NIFTY_Return_5d   [+0.0079]  (5-day cumulative NIFTY market return)
7. Vol_Change        [+0.0076]  (1-day trading volume change ratio)
8. Return_1d         [+0.0073]  (1-day asset price return)
9. MACD_Ratio        [+0.0072]  (Normalized MACD line)
10. CO_Spread        [+0.0071]  (Intraday Close-to-Open spread)
```

---

## Vectorized Backtesting Protocol & Execution Engine

### Protocol & Rules
- **Out-of-Sample Test Period:** 284 trading days (2024-12-23 to 2026-09-10)
- **Initial Capital:** ₹100,000
- **Signal Threshold:** $\text{Probability} \ge 0.50 \implies \text{LONG (1)}$, else $\text{CASH (0)}$
- **Friction Modeling:** 0.10% transaction cost + 0.05% slippage (Total friction: **0.15% per position change**)
- **Execution Timing:** Decision made at Close $T$ using features through $T$; return realized from Close $T$ to Close $T+1$.

---

## Vectorized Backtest Results vs Buy & Hold Benchmark

| Metric | Model Strategy (MLP [32, 16]) | Buy & Hold Benchmark |
|---|---|---|
| **Cumulative Total Return** | **+23.80%** | +5.13% |
| **Annualized Return** | **+20.78%** | +4.53% |
| **Annualized Volatility** | **11.45%** | 16.88% |
| **Sharpe Ratio** | **1.45** | 0.27 |
| **Maximum Drawdown** | **-5.27%** | -16.84% |
| **Win Rate** | **55.37%** (67 wins / 121 active days) | N/A |
| **Total Trades Executed** | **64** (32 entries, 32 exits) | 1 (Buy at start) |
| **Market Exposure** | **42.6%** (121 Long days, 162 Cash days) | 100.0% |

---

## Transaction Cost Sensitivity & Stress Testing

| Cost Regime | Friction per Trade | Cumulative Return | Sharpe Ratio | Max Drawdown | Win Rate |
|---|---|---|---|---|---|
| **Zero Costs (Theoretical)** | 0.00% | **+36.25%** | 2.06 | -4.79% | 58.68% |
| **Default Friction (Official)** | **0.15%** (0.10% cost + 0.05% slip) | **+23.80%** | **1.45** | **-5.27%** | **55.37%** |
| **High Friction** | 0.25% (0.20% cost + 0.05% slip) | **+16.13%** | 1.04 | -6.80% | 54.55% |
| **Severe Stress Test** | 0.30% (0.20% cost + 0.10% slip) | **+12.47%** | 0.83 | -8.08% | 52.89% |

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

3. **Q: Why did adding NIFTY and VIX context improve ROC-AUC from 0.496 to 0.567?**  
   *A:* Single-stock indicators lack macro regime context. Market index returns and volatility indices provide essential information regarding overall market direction and risk sentiment.

4. **Q: Why did your backtest strategy outperform Buy & Hold (+23.80% vs +5.13%) despite modest direction accuracy?**  
   *A:* The edge comes from risk mitigation. The model spent 57.4% of trading days in cash, avoiding major market drawdowns while capturing upward momentum during active positions.

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
- **Market Context Provides Signal:** Macro market feeds (NIFTY and VIX) are critical for predictive breakout.
- **Risk Mitigation Drives Strategy Return:** Sitting in cash during drawdowns generates higher risk-adjusted returns than aggressive trading.

---

## Official Research & Educational Disclaimer

**This software is strictly for academic research and educational purposes.** It does not constitute financial advice, investment advice, or trading recommendations. Past backtest performance does not guarantee future returns. Do not deploy this code in live trading or real-money financial environments.
