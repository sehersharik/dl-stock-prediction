# Frozen snapshot: run 20261007_003042 (the model deployed until Phase 3)

This is a copy of the working artifacts as they were before the Phase 3 retrain (2026-10-29). It keeps the old run reproducible and comparable after `data/processed/`, `data/predictions/`, `models/` and `reports/` are regenerated. The paths mirror the original locations.

- **MLP model:** `models/RELIANCE.NS_mlp_Target_Direction.pt`. It is byte-identical to `models/RELIANCE/mlp/20261007_003042/model.pt`, and that registry folder stays untouched.
- **Data:** `data/processed/RELIANCE.NS_X.csv` / `_y.csv`. These have 1,890 rows and still contain the holiday gaps and the 19 multi-session targets described in `docs/learning-notes.md` (Phase 3).
- **Split:** 70/15/15 by row. Train ends 2023-10-19, validation ends 2024-12-20, and the test set runs 2024-12-23 → 2026-09-10 (284 rows).
- **Feature set:** all 27 features (F5). This set was chosen by comparing **test** ROC-AUC in `reports/RELIANCE.NS_ablation_results.json`, so the test score is optimistic.
- **Baselines:** `models/RELIANCE.NS_{logistic,random_forest}_*`. The majority/LSTM files from the 2-year era are in `archive/2y_dataset/`.
- **Backtest and explainability:** `reports/`, produced from this run's test predictions.
