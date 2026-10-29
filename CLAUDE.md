# StockSense
## Project Documentation & Cloud Code Handoff

**Project:** AI-Based Stock Direction and Risk Prediction  
**Repository:** https://github.com/sehersharik/dl-stock-prediction  
**Live application:** https://stockpredictioncc.streamlit.app  
**Primary stock:** Reliance Industries Limited (`RELIANCE.NS`)  
**Primary market:** Indian equities  
**Framework:** Streamlit  
**Machine learning framework:** PyTorch  
**Status:** Deployed prototype; further reliability, research, and evaluation improvements required.

---

# 1. Purpose of This Document

This document provides the technical and research context required for another AI coding assistant, particularly Cloud Code, to continue developing StockSense without losing the decisions, experiments, constraints, and issues established during previous development.

The project should not be treated as a generic stock prediction application. It is an undergraduate machine learning project intended to evolve into a credible, research-oriented financial forecasting system.

The objective is to improve:

- Predictive performance and generalization.
- Financial time-series validation.
- Feature engineering and market-context integration.
- Backtesting correctness and transparency.
- Probability reliability and uncertainty estimation.
- Model interpretability.
- Software architecture and reproducibility.
- Dashboard usability and reliability.
- Research quality and suitability for a technical portfolio or master's applications.

**The priority is trustworthy experimentation, not making the model appear more accurate than it is.**

Cloud Code must first understand the existing implementation before changing it.

---

# 2. Project Background and Evolution

The project began as a deep-learning-based stock direction prediction system using historical financial market data.

The initial objective was to determine whether historical market information could help predict the direction of the next trading day's stock return.

The project subsequently incorporated:

1. Historical stock market data.
2. Market-wide context from the NIFTY 50 index.
3. India VIX as a market-volatility indicator.
4. Technical indicators and stationary or scale-invariant features.
5. A PyTorch multilayer perceptron.
6. Chronological training, validation, and test periods.
7. Expanding-window walk-forward validation.
8. Comparisons against traditional machine learning models.
9. A historical trading-strategy backtest.
10. Transaction-cost sensitivity analysis.
11. A Streamlit dashboard.
12. A more polished fintech-style interface branded as StockSense.

The dashboard has previously been organized around the following sections:

- Overview
- Analysis
- Methodology
- About

The intended application combines model predictions, historical evaluation, market context, model explainability, and backtesting results.

These are the established design goals. Cloud Code must inspect the current repository to confirm which features remain implemented and which are incomplete.

---

# 3. Important Development Principles

Before making any changes, follow these rules.

## 3.1 Inspect before implementing

Read the existing repository, including:

- Training scripts.
- Feature-engineering modules.
- Data-fetching utilities.
- Model definitions.
- Model artifacts.
- Evaluation scripts.
- Backtesting implementation.
- Dashboard components.
- Tests.
- Dependency files.
- Deployment configuration.
- Existing documentation and experiment reports.

Do not assume the project follows a particular directory structure merely because this document describes one.

## 3.2 Preserve reproducibility

Every experiment should have a reproducible definition.

Record:

- Dataset and date range.
- Feature set.
- Target definition.
- Training and validation periods.
- Test period.
- Model architecture.
- Hyperparameters.
- Random seeds.
- Preprocessing.
- Decision threshold.
- Evaluation metrics.
- Transaction-cost assumptions, when applicable.

Do not overwrite historical experiment results without retaining the original information.

## 3.3 Protect the test set

The final test period must remain untouched during model selection.

Do not repeatedly change features, hyperparameters, thresholds, or strategy rules based on final-test performance and then present the same period as an independent test.

Use training and validation data for development. Reserve the final test period for the final evaluation of a frozen methodology.

## 3.4 Never fabricate performance

Do not:

- Invent missing metrics.
- Report training performance as test performance.
- Mix results from different experiments.
- Present a backtest as proof of future profitability.
- Hide failed experiments.
- Change the reported test period without explanation.
- Select only the most favorable model results.
- Silently replace old dashboard metrics with newer numbers.

If results conflict, investigate the source of the discrepancy before updating the dashboard or documentation.

## 3.5 Preserve existing functionality

Make focused changes.

Before editing, identify the affected components and their dependencies.

After editing:

1. Run relevant tests.
2. Check existing functionality.
3. Inspect the diff.
4. Report what changed.
5. Identify anything that remains unverified.

Do not commit or push changes unless explicitly authorized.

---

# 4. Technology Stack

The project has used the following technologies.

| Component | Technology | Purpose |
|---|---|---|
| Application interface | Streamlit | Interactive dashboard |
| Machine learning | PyTorch | Neural-network model |
| Data manipulation | pandas | Tabular time-series processing |
| Numerical computation | NumPy | Numerical transformations |
| Traditional ML benchmarks | scikit-learn | Baselines and comparison models |
| Historical market data | yfinance has been used | Historical market-data retrieval |
| Version control | Git and GitHub | Source-code management |
| Deployment | Streamlit Community Cloud or equivalent Streamlit hosting | Public application |

**Cloud Code must verify the current dependency versions and actual implementation.** Do not assume that every listed package is currently used in every execution path.

Additional dependencies may exist for plotting, technical indicators, testing, or explainability.

Inspect the repository's dependency files before installing or upgrading anything.

Avoid unnecessary package upgrades that could break the existing deployment.

---

# 5. Data Sources and Market Instruments

The project has worked with:

| Instrument | Identifier or symbol | Intended role |
|---|---|---|
| Reliance Industries | `RELIANCE.NS` | Primary stock being predicted |
| NIFTY 50 | Verify the exact configured ticker | Broad Indian equity-market context |
| India VIX | Verify the exact configured ticker | Market-implied volatility context |

The historical dataset has previously contained approximately 1,890–1,891 usable observations after a 200-day indicator warm-up period.

This is a historical figure, not a guarantee about the current dataset.

The project has used approximately ten years of historical market data, although the exact start date, end date, and data-fetching configuration must be confirmed from the code.

## 5.1 Data quality requirements

Inspect how the system handles:

- Missing observations.
- Duplicate dates.
- Market holidays.
- Different trading calendars.
- Missing OHLCV values.
- Incorrect or stale data.
- Adjusted versus unadjusted prices.
- Rolling-indicator warm-up periods.
- Corporate actions.
- Failed provider requests.
- Time-zone and date alignment.

When combining Reliance, NIFTY 50, and India VIX, align observations using the intended trading-date policy.

Do not assume that independently downloaded datasets have identical indexes.

Do not silently fill unavailable market-context data with arbitrary values.

If required data is missing, either apply a justified, training-consistent policy or withhold the prediction and display a clear explanation.

## 5.2 Data-provider limitations

The public application may depend on an external market-data provider.

Cloud Code must verify whether the current provider:

- Supplies sufficiently recent data.
- Includes the required instruments.
- Returns data in the expected format.
- Handles throttling or temporary failures.
- Requires authentication.
- Provides the latest completed trading session consistently.

A successful historical data download does not guarantee that live inference will work.

---

# 6. Prediction Objective

The primary prediction task is binary next-day stock-direction classification.

The target has previously been defined as:

`next_day_return = close.shift(-1) / close - 1`

The binary target is:

`target = 1 if next_day_return > 0 else 0`

In mathematical notation:

\[
r_{t+1} = \frac{C_{t+1}}{C_t} - 1
\]

\[
y_t =
\begin{cases}
1 & \text{if } r_{t+1} > 0 \\
0 & \text{otherwise}
\end{cases}
\]

Where:

- \(C_t\) is the closing price on trading day \(t\).
- \(C_{t+1}\) is the closing price on the next trading day.
- \(r_{t+1}\) is the next-day close-to-close return.
- \(y_t\) is the classification target.

The model estimates the probability that the next-day return will be positive.

## 6.1 Critical timing convention

The target definition alone does not establish when a prediction can actually be traded.

Cloud Code must verify the intended signal-generation and execution convention.

For example, if all features are calculated using day \(t\)'s closing data, the model produces a prediction after that information is available.

A backtest must then use an executable entry price and return interval consistent with that timing.

Do not assume that a prediction generated using the closing price can automatically be executed at that same closing price.

If the system uses a different timing convention, document it explicitly.

## 6.2 Prediction versus investment recommendation

A predicted probability is a model output, not a guarantee.

The application should communicate that:

- The prediction is uncertain.
- Historical relationships may change.
- Market prices can move against the prediction.
- Transaction costs can reduce returns.
- Historical backtest performance is not a forecast of future performance.

The system is intended for research and educational use, not as a guarantee of investment outcomes.

---

# 7. Feature Engineering

The model has previously used a 27-feature input representation.

These features have been grouped into six conceptual groups, F0–F5.

The precise feature names, transformations, and group membership must be retrieved from the training code or saved model metadata.

The feature families include:

- Stock returns and OHLCV-derived information.
- Trend indicators.
- Momentum indicators.
- Volatility indicators.
- Volume indicators.
- Broader market context.

The model's feature schema is a critical interface between data preparation and inference.

## 7.1 Market-context features

The trained model is known to expect these six features:

| Feature | Intended meaning |
|---|---|
| `NIFTY_Return_1d` | NIFTY 50 one-day return |
| `NIFTY_Return_5d` | NIFTY 50 five-day return |
| `NIFTY_Vol_20` | NIFTY 50 20-day rolling volatility |
| `NIFTY_Dist_SMA50` | NIFTY 50's distance from its 50-day moving average |
| `VIX_Level` | India VIX level |
| `VIX_Change_1d` | One-day change in India VIX |

The exact mathematical definitions must be obtained from the original training implementation.

For example, a distance-from-moving-average feature could be calculated as a ratio or a percentage difference. These are not interchangeable unless the training definition specifies that they are.

Likewise, volatility could be calculated using different return definitions, window conventions, or annualization assumptions.

**Do not independently recreate these features using guessed formulas.**

The live inference pipeline must reproduce the training definitions, transformations, and feature ordering.

## 7.2 Preventing data leakage

Inspect every feature for look-ahead leakage.

Questions to answer:

1. Does the feature use information available only after the prediction time?
2. Does any rolling calculation accidentally include future observations?
3. Are shifts applied in the correct direction?
4. Are missing values removed without misaligning features and labels?
5. Are NIFTY and VIX observations aligned correctly?
6. Are preprocessing parameters fitted only on training data?
7. Is any target-derived information included in the input?
8. Does the backtest use future prices to determine a signal that supposedly existed earlier?

Pay special attention to the relationship between feature date \(t\) and target date \(t+1\).

## 7.3 Feature-schema contract

Establish one authoritative definition of the model's expected input schema.

It should specify:

- Exact feature names.
- Exact feature ordering.
- Data types.
- Transformations.
- Units.
- Missing-value policy.
- Any scaler or other preprocessing parameters.
- Model artifact compatibility.

Where practical, share the feature-generation implementation between training and live inference.

If sharing the complete implementation is impractical, create deterministic parity tests that compare both pipelines on the same historical inputs.

Do not solve a schema mismatch by inserting zeros, arbitrary defaults, or unrelated replacement features.

---

# 8. Model Architecture

The primary deep-learning model has previously used a small multilayer perceptron implemented in PyTorch.

The documented architecture is:

\[
27 \rightarrow 32 \rightarrow 16 \rightarrow 1
\]

Conceptually:

1. Input layer: 27 features.
2. Hidden layer: 32 units.
3. Hidden layer: 16 units.
4. Output layer: one logit for binary classification.

Dropout with a rate of 0.2 has been used.

L2 regularization has also been used, with Adam as the optimizer and `BCEWithLogitsLoss` as the loss function.

The exact hidden-layer activations, batch configuration, learning rate, epoch count, early-stopping policy, and other hyperparameters must be confirmed from the source code.

## 8.1 Output interpretation

The network produces a logit \(z\).

Applying the sigmoid function converts the logit to a probability:

\[
p = \frac{1}{1+e^{-z}}
\]

The probability is intended to represent the estimated likelihood of a positive next-day return, subject to the limitations of the training process.

A default classification threshold of 0.5 may have been used in some evaluations. Verify the threshold in every experiment rather than assuming it is universal.

## 8.2 Regularization

The model uses regularization techniques intended to reduce overfitting.

However, the presence of dropout or L2 regularization does not establish that the model generalizes well.

Generalization must be evaluated on strictly later observations and compared against appropriate baselines.

---

# 9. Training and Validation Methodology

The project has previously used a chronological split:

- 70% training.
- 15% validation.
- 15% final test.

It has also used five-fold expanding-window walk-forward validation.

Verify the exact implementation and observation counts.

## 9.1 Why chronological splitting matters

Financial time series are not independent randomly ordered samples.

Randomly splitting the data can allow future market conditions to influence model development and make performance estimates unreliable.

The intended approach is to train on earlier observations and evaluate on later observations.

## 9.2 Walk-forward validation

The project has previously used five expanding-window folds.

Conceptually:

1. Train on an initial historical period.
2. Validate on the next chronological period.
3. Expand the training window.
4. Validate on a later period.
5. Repeat until all configured folds are evaluated.

The actual fold boundaries must be verified in the code and experiment reports.

For every fold, verify that:

- Training observations precede validation observations.
- Scaling and preprocessing are fitted only on the fold's training data.
- Feature selection does not use validation outcomes.
- Hyperparameter selection does not use the final test set.
- The validation window does not accidentally overlap training labels in a way that invalidates the evaluation.

## 9.3 Final test set

The final test set must remain separate from model selection.

Previously recorded backtest audit results identify a test period from **23 December 2024 to 10 September 2026**, containing 284 observations.

This period and observation count are historical recorded results and must be verified against the current code and data before being reused.

Do not assume the current repository still uses the same split.

If the test period has already influenced feature or strategy selection, disclose that limitation.

---

# 10. Benchmark Models

The project has compared the MLP against traditional machine learning baselines.

Previously reported five-fold expanding-window validation results were:

| Model | Mean ROC-AUC | Standard deviation | Mean balanced accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.5143 | 0.037 | 0.5126 |
| Gradient Boosting | 0.4815 | 0.066 | 0.4855 |
| Random Forest | 0.5414 | 0.031 | 0.5195 |
| Small MLP, [32, 16] | 0.5491 | 0.010 | 0.5487 |

These values are previously recorded results. They must not be treated as verified outputs of the current code until reproduced.

The recorded MLP fold ROC-AUC values were:

- Fold 1: 0.5619
- Fold 2: 0.5495
- Fold 3: 0.5423
- Fold 4: 0.5583
- Fold 5: 0.5335

These indicate modest discrimination rather than evidence of a highly accurate stock-prediction system.

## 10.1 Additional final-test benchmark

A separate previously recorded experiment reported:

| Model | Test ROC-AUC | Balanced accuracy | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Majority baseline | 0.5000 | 0.5000 | 0.6604 | 0.4930 |
| Previous-direction baseline | 0.4824 | 0.4824 | 0.4806 | 0.4824 |
| Logistic Regression | 0.5077 | 0.5043 | 0.5253 | 0.5070 |
| Gradient Boosting | 0.4878 | 0.4759 | 0.4915 | 0.4789 |
| Random Forest | 0.4991 | 0.4924 | 0.4667 | 0.4965 |
| Small MLP, [32, 16] | 0.5370 | 0.5442 | 0.4861 | 0.5458 |

The baseline and F1 definitions must be checked before these numbers are reused.

In particular, the majority baseline's F1 value should be verified against the underlying predictions, positive-class definition, and averaging convention.

The benchmark table comes from a different recorded evaluation than the dashboard metrics described in the next section.

Do not combine the tables into one experiment without establishing that they use the same data, split, preprocessing, and evaluation definitions.

---

# 11. Important: Resolve the Metrics Discrepancy

Previous dashboard implementation logs reported:

| Metric | Previously displayed value |
|---|---:|
| ROC-AUC | 0.5676 |
| Balanced accuracy | 57.39% |
| Precision | 58.68% |
| F1 | 0.54 |

A separate final-test benchmark reported:

| Metric | Previously reported value |
|---|---:|
| ROC-AUC | 0.5370 |
| Balanced accuracy | 54.42% |
| F1 | 0.4861 |
| Accuracy | 0.5458 |

These values must not be silently substituted for one another.

They may come from different experiment runs, splits, model artifacts, thresholds, or metric calculations.

The exact cause has not been established.

### Required investigation

Cloud Code must determine:

1. Which model artifact is loaded by the deployed application.
2. Which evaluation run produced each set of metrics.
3. Whether the dashboard displays training, validation, cross-validation, or test performance.
4. Whether the evaluation periods match.
5. Whether the preprocessing and feature sets match.
6. Whether thresholds differ.
7. Whether metric definitions or averaging conventions differ.
8. Whether the dashboard contains hardcoded or stale values.
9. Whether the metrics were generated from the current code and artifacts.
10. Whether the reported results can be reproduced.

Create a single, explicit source of truth for the dashboard's reported evaluation metrics.

Each metric should be associated with its experiment, dataset, evaluation period, model artifact, and methodology.

Do not change the dashboard's displayed metrics until the discrepancy has been investigated and the authoritative results have been identified.

---

# 12. Evaluation Metrics

The project has used several classification metrics.

## 12.1 ROC-AUC

ROC-AUC measures the ability of a model to rank positive examples above negative examples across classification thresholds.

A value near 0.5 indicates performance similar to random ranking.

ROC-AUC does not establish that a strategy will make money.

## 12.2 Accuracy

Accuracy measures the proportion of predictions classified correctly at a specified threshold.

It can be misleading when classes are imbalanced or when a majority-class prediction performs well.

## 12.3 Balanced accuracy

Balanced accuracy averages the recall of the positive and negative classes.

It is useful when class frequencies differ.

## 12.4 Precision

Precision measures the proportion of predicted positive cases that are actually positive.

Its interpretation depends on the positive-class definition and threshold.

## 12.5 F1 score

F1 combines precision and recall through their harmonic mean.

Document whether the reported value uses binary, macro, weighted, or another averaging convention.

## 12.6 Metrics to investigate next

For a research-oriented extension, consider:

- PR-AUC.
- Matthews correlation coefficient.
- Confusion matrices.
- Brier score.
- Log loss.
- Calibration curves.
- Confidence intervals.
- Fold-level performance distributions.
- Performance by market regime.

Not every metric needs to appear on the dashboard. Choose metrics that answer a specific research question.

---

# 13. Backtesting

The project includes a historical trading-strategy backtest.

Previously recorded results included:

| Metric | Recorded value |
|---|---:|
| Strategy return | +23.80% |
| Buy-and-hold return | +5.13% |
| Sharpe ratio | 1.45 |
| Maximum drawdown | -5.27% |
| Win rate | 55.37% |
| Number of trades | 64 |

These values are historical results from a previously recorded experiment, not verified current performance.

They must be reproduced from the actual implementation before being used in a final report or public-facing claims.

## 13.1 Transaction costs

Previously recorded transaction-cost sensitivity results were:

| Cost scenario | Strategy return |
|---|---:|
| 0.00% | +36.25% |
| 0.15% | +23.80% |
| 0.25% | +16.13% |
| 0.30% | +12.47% |

A previous implementation log described the 0.15% friction assumption as:

- 0.10% transaction cost.
- 0.05% slippage.

Confirm whether these costs are applied per transaction, per side, or per round trip.

The cost convention must be documented explicitly.

Do not assume that the recorded sensitivity table is valid for the current strategy until the calculations have been reproduced.

## 13.2 Backtest timing audit

A previous audit reported that:

- Predictions at close \(T\) were associated with returns from \(T\) to \(T+1\).
- The target shift was checked against raw prices for five consecutive dates.
- The one-day return feature matched raw closing-price returns to six decimal places.
- The final test period was recorded as 23 December 2024 to 10 September 2026.
- The test period contained 284 observations.
- The training period ended in October 2023.
- The validation period ended in December 2024.
- The recorded metrics were reproduced exactly.
- Eighteen tests passed.

These are prior audit findings, not proof that every aspect of the strategy is correct or that the current repository still has the same implementation.

### Verify the following independently

1. Signal timestamp.
2. Feature availability at the signal timestamp.
3. Entry price.
4. Exit price.
5. Return interval.
6. Position sizing.
7. Long, short, or cash behavior.
8. Whether the model can hold overlapping positions.
9. Transaction-cost calculation.
10. Slippage calculation.
11. Turnover.
12. Exposure.
13. Buy-and-hold benchmark alignment.
14. Annualization conventions.
15. Sharpe-ratio calculation.
16. Maximum drawdown calculation.
17. Treatment of missing observations.
18. Number of trades and win-rate definition.

A backtest that generates a signal using information unavailable at the simulated execution time is invalid.

## 13.3 Historical performance is not a forecast

Do not claim that the strategy will achieve the recorded returns in the future.

The backtest should be presented as a historical simulation with clearly documented assumptions and limitations.

Where possible, report benchmark-relative returns, drawdowns, volatility, turnover, and costs together rather than presenting cumulative return alone.

---

# 14. Streamlit Dashboard

The deployed application is:

https://stockpredictioncc.streamlit.app

The project has used a professional fintech-inspired interface branded as StockSense.

Previously discussed dashboard sections include:

- Overview.
- Analysis.
- Methodology.
- About.

The dashboard has also been designed to include elements such as:

- Prediction output.
- Predicted probability.
- Price charts.
- Market context.
- Model performance.
- Explainability.
- Historical backtesting.
- Sensitivity analysis.

Cloud Code must inspect the current application to determine which of these features are implemented, functional, and connected to the actual model outputs.

## 14.1 UI principles

Keep the interface clean and professional.

Prioritize:

- Readability.
- Consistent typography.
- Clear metric definitions.
- Explicit data timestamps.
- Distinction between live and historical results.
- Clear failure states.
- Accurate labels and units.
- Useful explanations instead of decorative metrics.

Do not redesign the interface unnecessarily when fixing a backend issue.

The current priority is to restore reliable live inference and reconcile displayed metrics.

---

# 15. Current Critical Bug: Live Feature-Schema Mismatch

The deployed application currently displays the following error:

> Live market provider status: Prediction unavailable: live feature schema does not match the trained model. Missing: {'VIX_Change_1d', 'VIX_Level', 'NIFTY_Dist_SMA50', 'NIFTY_Return_1d', 'NIFTY_Vol_20', 'NIFTY_Return_5d'}. Displaying verified model results.

This is the immediate engineering priority.

## 15.1 What the error means

The live inference pipeline is not supplying all the feature columns required by the trained model.

The missing columns are:

- `NIFTY_Return_1d`
- `NIFTY_Return_5d`
- `NIFTY_Vol_20`
- `NIFTY_Dist_SMA50`
- `VIX_Level`
- `VIX_Change_1d`

The system appears to withhold live prediction when the schema is incompatible and continue displaying previously calculated model results.

This behavior is preferable to silently generating predictions using incomplete inputs.

However, the root cause has not yet been conclusively identified.

Possible causes include:

- Live feature engineering differs from training feature engineering.
- NIFTY 50 or India VIX data is unavailable.
- Rolling indicators are not being calculated.
- Data alignment or missing-value handling removes the columns.
- The deployed code is outdated.
- The loaded model artifact and current code are incompatible.

These are hypotheses to investigate, not established facts.

## 15.2 Required debugging sequence

### Step 1: Inspect the trained model

Identify the exact model artifact currently loaded.

Retrieve its expected feature names and order from saved metadata or the training code.

If the model artifact does not store feature metadata, establish it from the authoritative training implementation.

### Step 2: Inspect live feature generation

Determine exactly which features the live pipeline produces.

Inspect whether the market-data provider returns the required NIFTY 50 and India VIX data.

Check the output after every major transformation:

- Data retrieval.
- Date alignment.
- Return calculation.
- Rolling-indicator calculation.
- Missing-value handling.
- Feature selection.
- Final inference input construction.

### Step 3: Reconcile training and inference

Use the same feature definitions and transformations as training.

Do not guess the formulas for the six missing features.

Ensure that the live pipeline uses sufficient historical observations for rolling calculations.

Ensure that all features are available at the prediction timestamp.

### Step 4: Enforce the schema

Validate:

- Feature presence.
- Feature order.
- Data types.
- Missing values.
- Numerical validity.
- Preprocessing compatibility.
- Model artifact compatibility.

Avoid inserting arbitrary zeros or substituting unrelated features.

If required market data is unavailable, withhold the prediction and explain why.

### Step 5: Add tests

Create regression tests using deterministic fixtures or mocked provider responses.

Tests should establish that:

1. Valid market data produces all required features.
2. Training and live feature definitions agree.
3. The expected feature ordering is preserved.
4. Missing market data produces a controlled error.
5. Stale data cannot silently appear current.
6. Valid features can be passed to the model successfully.
7. No future observations enter the input.
8. Existing historical evaluation results remain unchanged.

### Step 6: Verify deployment

Local tests are necessary but insufficient.

After the fix is ready and the deployment is updated, verify that the public application can fetch valid market data, construct the feature row, and perform inference.

Do not claim the deployment is fixed merely because the local code passes tests.

---

# 16. Live Prediction and Historical Evaluation Must Be Separate

The dashboard should distinguish three different concepts.

### A. Historical evaluation

Previously measured model performance on a specified dataset and period.

Examples include ROC-AUC, balanced accuracy, and historical backtest results.

These values should be reproducible and tied to a specific experiment.

### B. Latest available market data

The latest successfully retrieved observations for Reliance, NIFTY 50, and India VIX.

Display the relevant dates and identify stale or incomplete data.

### C. Live model inference

A prediction generated using the latest valid feature row and a compatible trained model.

This output must not be confused with historical evaluation metrics.

If live inference fails, the dashboard may still show historical evaluation results, but it must clearly indicate that no new prediction was generated.

If the market is closed, distinguish the latest available completed trading session from a current intraday observation.

---

# 17. Suggested Repository Organization

The following is a conceptual structure based on the project's previously discussed components.

**It is not a claim that the current repository has exactly this layout.** Inspect the real repository before moving or renaming anything.

```text
dl-stock-prediction/
│
├── app/
│   └── dashboard/
│       └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── backtesting/
│
├── models/
│   └── [saved model artifacts and metadata]
│
├── tests/
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_evaluation.py
│   └── test_backtesting.py
│
├── reports/
│   ├── experiments/
│   └── research/
│
├── requirements.txt
├── README.md
└── CLOUD_CODE_HANDOFF.md
```

Do not reorganize the repository just to match this example.

First understand the current imports, deployment paths, model-artifact locations, and tests.

Make structural changes only when there is a clear benefit and a migration plan.

---

# 18. Local Setup and Execution

Cloud Code must inspect the actual dependency and configuration files before finalizing setup instructions.

The following is a generic starting point, not a verified set of repository-specific commands.

## 18.1 Clone the repository

```bash
git clone https://github.com/sehersharik/dl-stock-prediction.git
cd dl-stock-prediction
```

## 18.2 Create a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 18.3 Install dependencies

If the repository contains `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Do not upgrade dependencies indiscriminately if the repository relies on older compatible versions.

## 18.4 Launch the dashboard

The previously discussed dashboard path is:

```text
app/dashboard/app.py
```

If that path still exists, the likely command is:

```bash
streamlit run app/dashboard/app.py
```

Confirm the actual entry point before running it.

## 18.5 Run tests

Inspect the existing test framework and project instructions first.

If the repository uses pytest, the test command may be:

```bash
pytest
```

Report the actual command, test count, failures, and any tests that could not run.

Do not report a successful test suite if the tests were not executed.

---

# 19. Research Roadmap

The project should evolve through controlled experiments rather than adding complexity without evidence.

## Phase 1: Audit and reliability

**Priority: Critical**

- Resolve the live feature-schema mismatch.
- Reconcile the conflicting dashboard and benchmark metrics.
- Verify the current model artifact.
- Reproduce the reported final-test metrics.
- Audit feature leakage and prediction timing.
- Audit backtesting assumptions.
- Establish reproducible experiment records.
- Add tests for feature parity and inference.

**Completion criterion:** The live pipeline is reliable, evaluation results have traceable sources, and the backtest timing is documented and tested.

## Phase 2: Establish a trustworthy benchmark

**Priority: High**

Run a controlled benchmark comparing:

- Majority-class baseline.
- Previous-direction baseline.
- Logistic Regression.
- Random Forest.
- Gradient Boosting.
- Existing MLP.

Use identical chronological evaluation periods wherever the comparison is intended to be direct.

Document any model-specific preprocessing.

Use walk-forward validation and preserve the final test set.

Report fold-level metrics rather than only a single average.

**Completion criterion:** The project has a reproducible model-comparison report with clearly defined baselines and evaluation periods.

## Phase 3: Statistical reliability

**Priority: High**

Investigate whether the observed performance is meaningfully better than the relevant baselines.

Potential analyses include:

- Confidence intervals.
- Fold-to-fold variability.
- Bootstrap analysis appropriate for dependent time-series observations.
- Sensitivity to the evaluation period.
- Performance stability across market conditions.

Avoid assuming that ordinary independent-sample statistical tests are automatically valid for financial time series.

**Completion criterion:** The report discusses both the observed performance and the uncertainty around it.

## Phase 4: Market-regime analysis

**Priority: Medium to High**

Investigate performance across different market conditions, such as:

- High versus low volatility.
- Upward versus downward market trends.
- Different volatility regimes.
- Large-movement versus ordinary trading periods.

Define regimes using information available at the relevant prediction time.

Avoid selecting regimes retrospectively to make the model look better.

**Completion criterion:** The report explains where the model performs relatively well or poorly, without overstating conclusions from small samples.

## Phase 5: Probability calibration

**Priority: Medium to High**

The model produces a probability, but that probability may not correspond to the actual frequency of positive outcomes.

Investigate:

- Reliability diagrams.
- Brier score.
- Log loss.
- Calibration using a temporally appropriate validation set.
- Whether calibration remains stable across later periods.

Do not calibrate on the final test set and then report the same test results as independent.

**Completion criterion:** The project can assess whether predicted probabilities are reliable, not merely whether classifications are correct.

## Phase 6: Downside-risk prediction

**Priority: Medium**

Explore a separate task for forecasting unusually negative next-day returns or downside risk.

Define the target using a documented, defensible rule.

Compare the risk model against simple baselines.

Evaluate precision-recall behavior, false alarms, missed events, and calibration where appropriate.

Treat this as a separate research experiment rather than silently changing the original direction-prediction target.

**Completion criterion:** A documented experiment establishes whether downside-risk forecasting adds useful information.

## Phase 7: Sequence models, if justified

**Priority: Optional**

Potential models include:

- LSTM.
- GRU.
- Temporal convolutional networks.
- Transformer-based time-series models.

Do not add these merely because they are more sophisticated.

First establish whether the current MLP and traditional baselines have a reliable, reproducible evaluation.

The dataset has previously contained approximately 1,900 usable observations, which is relatively small for complex deep-learning models.

A more complicated architecture could overfit and perform worse.

**Completion criterion:** A sequence model is retained only if it provides a defensible improvement under the same chronological evaluation protocol.

## Phase 8: Dashboard improvements

**Priority: After the data and evaluation foundations are reliable**

Potential improvements include:

- Historical prediction log.
- Prediction timestamps.
- Latest market-data timestamps.
- Model-version information.
- Confidence or probability reliability visualization.
- Performance by market regime.
- Backtest assumptions panel.
- Data-provider health status.
- Clear distinction between live inference and historical metrics.

Do not prioritize decorative interface changes over correctness.

## Phase 9: Research report

Prepare a technical report describing:

1. Research question.
2. Motivation.
3. Dataset.
4. Feature engineering.
5. Prediction target.
6. Experimental methodology.
7. Baseline models.
8. Deep-learning architecture.
9. Walk-forward validation.
10. Final test results.
11. Backtesting methodology.
12. Statistical uncertainty.
13. Limitations.
14. Failure cases.
15. Future work.

Include negative results and failed experiments where they are informative.

Do not frame modest predictive performance as a breakthrough.

---

# 20. Immediate Task List for Cloud Code

Work through the following in order.

### Task 1 — Repository reconnaissance

Inspect the actual files, model artifacts, tests, and documentation.

Return a concise map of the project architecture and identify the source of truth for training, inference, metrics, and backtesting.

Do not make changes during the initial inspection.

### Task 2 — Live feature-schema bug

Identify why the six required NIFTY/VIX features are absent from the live feature DataFrame.

Fix the root cause using the established training definitions.

Add deterministic regression tests.

### Task 3 — Metric reconciliation

Trace the previously recorded dashboard metrics:

- ROC-AUC: 0.5676.
- Balanced accuracy: 57.39%.
- Precision: 58.68%.
- F1: 0.54.

Compare them with the separately recorded final-test results:

- ROC-AUC: 0.5370.
- Balanced accuracy: 54.42%.
- F1: 0.4861.
- Accuracy: 0.5458.

Identify which runs produced these results and whether they can be reproduced.

Do not silently overwrite the dashboard.

### Task 4 — Backtest audit

Verify prediction timing, executable prices, return alignment, position changes, transaction costs, benchmark comparability, and performance calculations.

Reproduce the recorded results if possible.

Document discrepancies rather than hiding them.

### Task 5 — Baseline benchmark

Only after the evaluation methodology is reliable, run a controlled comparison of the existing models.

Preserve the final test set.

### Task 6 — Research enhancements

Proceed to uncertainty analysis, regime analysis, calibration, and downside-risk prediction in separate, documented experiments.

Do not implement every enhancement simultaneously.

---

# 21. Required Working Style for Cloud Code

For each task:

1. State what you are investigating.
2. Inspect the relevant source files.
3. Explain the root cause before proposing a substantial change.
4. Make the smallest appropriate change.
5. Add or update tests.
6. Run the relevant tests.
7. Inspect the final diff.
8. Summarize the changes and their effects.
9. Identify remaining risks and unverified assumptions.

When results are uncertain, state that explicitly.

If a required file, artifact, or data source is missing, report the missing dependency instead of inventing its contents.

Do not retrain the model unnecessarily when the problem is only in the dashboard or live-data pipeline.

Do not alter the target definition, feature schema, model architecture, evaluation split, or trading rules without explaining why the change is necessary.

Do not commit, push, or deploy changes without explicit authorization.

Never expose secrets, API keys, credentials, or private configuration values in logs or documentation.

---

# 22. Definition of Done

StockSense should not be considered research-ready merely because the dashboard loads.

The following criteria should be satisfied progressively.

## Engineering

- [ ] Repository architecture is documented.
- [ ] Setup instructions are verified.
- [ ] Dependencies are documented.
- [ ] Training and inference schemas are consistent.
- [ ] Missing market data is handled safely.
- [ ] Live inference passes regression tests.
- [ ] Model artifacts are traceable to training runs.
- [ ] Dashboard status accurately reflects inference availability.

## Evaluation

- [ ] Target definition is documented.
- [ ] Chronological splits are reproducible.
- [ ] Walk-forward validation is reproducible.
- [ ] Baselines use comparable evaluation periods.
- [ ] Metrics are traceable to specific experiments.
- [ ] Dashboard metrics match the authoritative evaluation.
- [ ] Final-test data is protected from model selection.
- [ ] Data leakage checks are documented.

## Backtesting

- [ ] Signal timing is explicit.
- [ ] Entry and exit assumptions are documented.
- [ ] Costs and slippage are implemented consistently.
- [ ] Buy-and-hold comparison is aligned.
- [ ] Return and drawdown calculations are tested.
- [ ] Historical results are reproducible.
- [ ] Performance claims are appropriately qualified.

## Research

- [ ] Limitations are acknowledged.
- [ ] Statistical uncertainty is investigated.
- [ ] Market-regime analysis is conducted if justified.
- [ ] Probability calibration is assessed if probabilities are presented as meaningful confidence estimates.
- [ ] New models are compared under the same evaluation methodology.
- [ ] Results are recorded in a research report.

---

# 23. Final Context

StockSense is a developing machine-learning research project, not a proven profitable trading system.

Its most important current problems are not necessarily the sophistication of its neural network. They are the reliability of live feature generation, the consistency of reported metrics, and the reproducibility of its evaluation methodology.

The recorded experiments show modest classification performance, and the final-test benchmark differs from the previously displayed dashboard metrics. These discrepancies need to be resolved before making stronger claims about model quality.

The next stage should therefore focus on:

1. Fixing the live feature-schema mismatch.
2. Reconciling the reported metrics.
3. Reproducing the backtest.
4. Establishing a trustworthy benchmark.
5. Quantifying uncertainty.
6. Extending the research only when the evidence justifies it.

**The guiding principle is simple: make StockSense more correct, reproducible, and scientifically defensible before making it more complicated.**
