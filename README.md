# Deep Learning Based Stock Risk and Direction Prediction System

A research-grade financial machine learning framework designed to empirically investigate the predictive utility of technical indicators on next-day asset movement and downside risk. This system provides an end-to-end pipeline covering strict chronological data ingestion, dynamic feature generation, sequence-based PyTorch deep learning, feature ablation, model explainability, vectorized backtesting, and live inference.

## Research Question
"How do different groups of technical indicators and market-context features affect the performance of deep-learning models for next-day stock direction and downside-risk prediction?"

## Why This Problem Matters
Financial time series are notoriously non-stationary and noisy. Practitioners frequently concatenate dozens of technical indicators into feature matrices, assuming that more data equates to higher predictive capacity. This project matters because it establishes a strict empirical framework to measure precisely which families of indicators contribute mathematically to model convergence, and which introduce destructive noise that causes overfitting.

## System Architecture
The pipeline enforces strict separation of concerns to prevent data leakage:
1. **Data Ingestion Layer:** Defensively polls Yahoo Finance, handling retries, timezone normalization, and duplicate-dropping.
2. **Feature & Target Generation:** Physically isolates `X` (Features at time $t$) from `y` (Targets at time $t+1$).
3. **Time-Series Validation:** Enforces forward-only chronological array splitting and fits standard scalers strictly on the training partition.
4. **Deep Learning Engine:** PyTorch-based training loops utilizing Early Stopping and Dynamic Class Weighting.
5. **Analytics & Backtesting:** Integrates SHAP/Permutation Explainability and vectorized strategy simulations.
6. **Live Inference:** Polling buffer that applies exact historical scaling transformations to live market ticks.

## Dataset
Evaluated strictly on `RELIANCE.NS` using Daily (`1d`) OHLCV sequences. Data was chronologically partitioned into a `70%` Training, `15%` Validation, and `15%` Test split.

## Feature Engineering
Indicators were strictly calculated using data available at or before time $t$:
* **RSI (Relative Strength Index):** A momentum oscillator measuring the speed and change of price movements.
* **MACD (Moving Average Convergence Divergence):** A trend-following momentum indicator showing the relationship between two moving averages (12 and 26 periods).
* **Bollinger Bands:** Volatility bands placed above and below a moving average (typically 20-day), expanding/contracting based on standard deviation.
* **EMA/SMA:** Exponential and Simple Moving Averages smoothing price data to identify trend directions over 20, 50, and 200 days.
* **ATR (Average True Range):** A measure of market volatility derived from the moving average of true ranges.
* **Volatility:** Rolling standard deviations of daily returns.
* **Volume Indicators:** On-Balance Volume (OBV) and volume moving averages capturing buying/selling pressure.
* **Market Context:** Integration of broader indices (e.g., NIFTY, India VIX) aligned securely without forward-filling errors.

## Target Definition
Targets were rigorously constructed to ensure zero leakage into the feature matrix:
* **Next-Day Direction:** A binary classification target mapped to `1` if the asset's return at $t+1$ is strictly greater than `0`, else `0`.
* **Downside-Risk Classification:** A binary threshold target mapped to `1` if the asset's return at $t+1$ crosses below a configured negative risk threshold (e.g., `-0.02`), else `0`.

## Experimental Design
To answer the research question, a controlled ablation framework iteratively tested models across expanding feature matrices:
* **F0 (Raw):** OHLCV + Returns
* **F1 (Trend):** F0 + SMA/EMA
* **F2 (Momentum):** F1 + RSI, MACD, ROC
* **F3 (Volatility):** F2 + ATR, Bollinger Bands, Rolling Volatility
* **F4 (Volume):** F3 + OBV, Volume Changes
* **F5 (Context):** F4 + Market Indices

## Models
The framework provisions five benchmark architectures:
* **Logistic Regression:** Linear baseline.
* **Random Forest:** Non-linear tree ensemble baseline.
* **XGBoost:** Gradient boosted tree baseline.
* **MLP (Multilayer Perceptron):** Configurable deep neural network (e.g., `[128, 64, 32]` dimensions with Batch Normalization and Dropout).
* **LSTM (Long Short-Term Memory):** Recurrent neural network mapping rolling 30-day chronological sequences to contextualize temporal market states.

## Validation
Randomized `train_test_split` methodologies are invalid for financial time series due to look-ahead bias and autocorrelation. This system natively supports:
* **Chronological Split:** A hard sequential cutoff mapping the earliest 70% of rows to Training, the subsequent 15% to Validation, and the final 15% to untouched Testing.
* **Walk-Forward Validation:** Expanding window loops generating sequential folds for robust hyperparameter optimization.

## Results
Based on testing `RELIANCE.NS` against the Next-Day Direction target:
* The **MLP** achieved its peak predictive capacity specifically on the **F4 (Volume)** dataset, recording an out-of-sample **ROC-AUC of 0.6391** and a **PR-AUC of 0.7821**. Adding Moving Averages (F1) actually degraded the model's performance compared to raw prices.
* The **LSTM** collapsed into predicting a single majority class across F0-F2 due to extreme noise, but immediately broke out upon the introduction of **Volatility (F3)** features, achieving an **F1-Score of 0.5000** and a Precision of 73.3%.

## Explainability
Permutation Importance was utilized to identify model dependencies (measured by the drop in Test ROC-AUC when a feature array is destroyed). For the optimal MLP, the most heavily weighted contributors to the model's predictions were:
1. `Vol_Change`
2. `Open`
3. `High`
4. `Bollinger Lower`

*(Note: These metrics indicate mathematical contribution to the model's decision boundaries, and do not signify causal market mechanisms).*

## Backtesting
The `test_preds.csv` outputs were piped into a vectorized financial backtester penalizing a `0.10%` transaction cost and `0.05%` slippage per state-change.
Using a strict `0.5` long/cash probability threshold, the MLP logged:
* **Strategy Return:** 1.94% (vs 15.39% Buy-and-Hold Benchmark)
* **Sharpe Ratio:** 1.44
* **Maximum Drawdown:** -2.34%
* **Win Rate:** 72.73% (across 14 executed trades)

The model severely underperformed the benchmark in absolute returns due to extreme conservatism (low recall), but functioned successfully as a high-precision, low-drawdown risk filter.

## Live Inference
An independent `LiveDataProvider` fetches latest quotes, dynamically splices them into a ~250-day rolling historical buffer, derives the feature engine natively, scales using the saved `StandardScaler.joblib`, and pushes the tensors through the `.pt` models to emit a realtime signal.

## Dashboard
An interactive Streamlit application (`app/dashboard/app.py`) parses the model registry and generates an aesthetic financial terminal outlining the dynamic predictions, metrics, backtest curves, and experimental ablation plots.

## Project Structure
```text
project/
├── app/
│   └── dashboard/
│       └── app.py
├── config/
│   ├── logging.yaml
│   └── settings.yaml
├── data/
│   ├── external/
│   ├── predictions/
│   ├── processed/
│   └── raw/
├── models/
│   └── {TICKER}/
│       └── {MODEL}/
│           └── {RUN_ID}/
├── notebooks/
├── reports/
├── src/
│   ├── backtesting/
│   ├── data/
│   ├── evaluation/
│   ├── experiments/
│   ├── explainability/
│   ├── features/
│   ├── inference/
│   ├── models/
│   └── targets/
├── tests/
├── main.py
├── requirements.txt
└── .env.example
```

## Installation
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Configuration
Core project variables (date ranges, target thresholds, sequence lengths) are controlled natively via `config/settings.yaml`. Live API keys must be mounted inside the `.env`.

## Running the Project
The central `main.py` CLI exposes the full pipeline:
```bash
python main.py ingest --ticker RELIANCE.NS
python main.py features --ticker RELIANCE.NS
python main.py targets --ticker RELIANCE.NS
python main.py splits --ticker RELIANCE.NS
python main.py train --ticker RELIANCE.NS --model mlp
python main.py backtest --ticker RELIANCE.NS --model mlp
python main.py infer --ticker RELIANCE.NS --model mlp
```

## Reproducing Experiments
To reproduce the empirical research study, invoke:
```bash
# Iterates F0-F5 permutations across MLP and LSTM
python main.py ablation --ticker RELIANCE.NS

# Extracts feature weights and SHAP equivalents
python main.py explain --ticker RELIANCE.NS --model mlp
```
Every execution securely deposits an immutable `metadata.json` payload inside `models/{ticker}/{model}/{run_id}` containing all hyperparameter signatures, array sizes, and git commits to guarantee absolute reproducibility.

## Limitations
* **Market Non-Stationarity:** Predictive relationships present in the 2016-2021 training block may decay or completely reverse in modern macroeconomic regimes.
* **Slippage Execution:** While penalized in backtesting, high-probability algorithmic signals often trigger concurrently across the institutional market. Executing bulk trades at theoretical closing prices without suffering heavy bid/ask slippage is improbable.
* **Delayed Feeds:** The fallback API implementation utilizes standard delayed metrics. Without integrating a premium Sub-Second WebSocket feed, live deployment is strictly theoretical.

## Future Work
1. Implementation of hyperparameter optimization sweeps (e.g., Ray Tune/Optuna).
2. Development of dual-head neural networks to predict Direction and Risk simultaneously.
3. Integration of sequential Transformer architectures (Time-Series Attention).

## Disclaimer
**This system is for research and educational purposes only.** It does not constitute investment advice, financial advice, trading advice, or any other sort of advice. The historical backtest performance shown does not guarantee future returns. Do not deploy these algorithms in live financial environments.
