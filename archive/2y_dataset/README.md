# Archived: 2-year dataset artifacts (2022-01-01 → 2024-01-01)

These files came from an earlier version of the project, trained on about 2 years of RELIANCE.NS data. Its test set had **44 days**. The current pipeline uses about 10 years of data and a **284-day** test set (2024-12-23 → 2026-09-10).

They are kept for the record but are **not comparable** with current results: they use a different dataset, split and test size. Nothing in the app or pipeline reads them.

Moved here on 2026-10-29 during the Phase 2 cleanup. Paths below mirror where each file used to be:

| File | Why archived |
|---|---|
| `data/raw/RELIANCE.NS_2022-01-01_2024-01-01.csv` (+ `_meta.json`) | The 2-year raw price download |
| `models/RELIANCE.NS_majority_Target_Direction{.joblib,_scaler.joblib,_metrics.json}` | Majority baseline fitted on the 2-year data (205 train / 44 test rows) |
| `models/RELIANCE.NS_lstm_Target_Direction{.pt,_metrics.json}` | LSTM from the 2-year era (44 test rows). The 10-year LSTM runs are in `models/RELIANCE/lstm/` |
| `data/predictions/RELIANCE.NS_majority_Target_Direction_{train,val,test}_preds.csv` | Majority-baseline predictions on the 2-year split |
| `reports/baseline_comparison.csv` | Baseline table computed on the 44-day test set (same majority PR-AUC, 0.6591). No code in the current repo produces it |
