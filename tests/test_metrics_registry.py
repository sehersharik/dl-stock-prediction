import json
import re
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import evaluate_classification
from src.evaluation.metrics_registry import active_run, load_metrics
from src.evaluation.readme_render import README_PATH, TEMPLATE_PATH, render_text
from src.evaluation.splitter import TimeSeriesSplitter
from src.utils.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestMetricsRegistry(unittest.TestCase):
    """reports/metrics.json is the single source of reported results."""

    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics()

    def test_readme_is_rendered_from_active_run(self):
        expected = render_text(TEMPLATE_PATH.read_text(), active_run(self.metrics))
        self.assertEqual(README_PATH.read_text(), expected,
                         "README.md is stale: run `python main.py readme` after `python main.py metrics`")

    def test_reported_test_metrics_match_saved_predictions(self):
        for run_id, run in self.metrics["runs"].items():
            preds = pd.read_csv(PROJECT_ROOT / run["artifacts"]["test_preds"], index_col=0)
            recomputed = evaluate_classification(preds["True"], preds["Pred"], preds["Prob"])
            reported = run["metrics"]["mlp"]["test"]
            for k in ("ROC-AUC", "PR-AUC", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1"):
                self.assertAlmostEqual(reported[k], recomputed[k], places=10, msg=f"{run_id} {k}")
            self.assertEqual(len(preds), run["split"]["test"]["n"], run_id)

    def test_active_run_artifacts_exist(self):
        run = active_run(self.metrics)
        for key in ("model", "scaler", "test_preds"):
            self.assertTrue((PROJECT_ROOT / run["artifacts"][key]).exists(), key)

    def test_validation_selected_runs_never_scored_test_during_ablation(self):
        for run_id, run in self.metrics["runs"].items():
            sel = run["feature_selection"]
            if sel["split"] != "validation":
                continue
            for lvl, v in sel["levels"].items():
                run_dir = PROJECT_ROOT / "models" / "RELIANCE" / "mlp" / v["run_id"]
                meta = json.loads((run_dir / "metadata.json").read_text())
                self.assertIsNone(meta["metrics"]["test"], f"ablation run {lvl} scored the test set")
                self.assertFalse((run_dir / "test_preds.csv").exists(), f"ablation run {lvl} saved test predictions")

    def test_fixed_split_matches_config(self):
        sd = load_config()["training"]["split_dates"]
        for run in self.metrics["runs"].values():
            if not run["split"]["method"].startswith("fixed dates"):
                continue
            self.assertLessEqual(run["split"]["train"]["end"], sd["train_end"])
            self.assertGreater(run["split"]["val"]["start"], sd["train_end"])
            self.assertLessEqual(run["split"]["val"]["end"], sd["val_end"])
            self.assertGreater(run["split"]["test"]["start"], sd["val_end"])
            self.assertLessEqual(run["split"]["test"]["end"], sd["test_end"])

    def test_dashboard_has_no_hardcoded_metric_values(self):
        source = (PROJECT_ROOT / "app" / "dashboard" / "app.py").read_text()
        literals = re.findall(r"(?<![\w.])0\.\d{4,}(?!\w)", source)
        self.assertEqual(literals, [], f"hardcoded metric-like literals in app.py: {literals}")


class TestFixedDateSplit(unittest.TestCase):
    def test_date_boundaries_are_inclusive_and_drop_rows_after_test_end(self):
        idx = pd.bdate_range("2024-01-01", periods=20)
        X = pd.DataFrame({"a": range(20)}, index=idx)
        y = pd.Series(range(20), index=idx)
        cfg = {"training": {"split_dates": {"train_end": str(idx[9].date()), "val_end": str(idx[14].date()),
                                            "test_end": str(idx[17].date())}}}
        (Xtr, _), (Xv, _), (Xte, _) = TimeSeriesSplitter(cfg).split_train_val_test(X, y)
        self.assertEqual((len(Xtr), len(Xv), len(Xte)), (10, 5, 3))
        self.assertEqual(Xte.index[-1], idx[17])


if __name__ == '__main__':
    unittest.main()
