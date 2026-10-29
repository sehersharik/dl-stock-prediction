"""Uncertainty for test-set metrics on dependent daily observations.

- Stationary block bootstrap (Politis & Romano, 1994): resample runs of consecutive
  days with geometric block lengths (mean `mean_block`), wrapping circularly. The same
  resampled indices are used for every model, so paired differences are valid.
- Circular-shift test against chance: shift the labels relative to the predictions by
  every offset of at least `min_shift` days. This keeps the autocorrelation of both
  series but breaks their alignment, giving a null distribution for "no skill".
"""
import numpy as np


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), ties handled by average ranks."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return np.nan
    return float(((pred[pos] == 1).mean() + (pred[neg] == 0).mean()) / 2)


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    restart = rng.random(n) < 1.0 / mean_block
    starts = rng.integers(n, size=n)
    for t in range(1, n):
        idx[t] = starts[t] if restart[t] else (idx[t - 1] + 1) % n
    return idx


def _summary(estimate: float, draws: np.ndarray, null: float = None) -> dict:
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out = {"estimate": float(estimate), "ci95": [float(lo), float(hi)], "boot_std": float(draws.std(ddof=1))}
    if null is not None:
        out["excludes_null"] = bool(lo > null or hi < null)
        out["share_draws_at_or_below_null"] = float((draws <= null).mean())
    return out


def bootstrap(y: np.ndarray, models: dict, reference: str, n_boot: int = 5000, mean_block: float = 10.0, seed: int = 42) -> dict:
    """models: {name: (prob, pred)} on the same days as y. Returns CIs per model and
    paired differences against `reference`."""
    rng = np.random.default_rng(seed)
    names = list(models)
    auc = {m: np.empty(n_boot) for m in names}
    bacc = {m: np.empty(n_boot) for m in names}
    for b in range(n_boot):
        i = stationary_bootstrap_indices(len(y), mean_block, rng)
        yb = y[i]
        for m in names:
            prob, pred = models[m]
            auc[m][b] = roc_auc(yb, prob[i])
            bacc[m][b] = balanced_accuracy(yb, pred[i])
    out = {"method": "stationary block bootstrap (Politis & Romano), percentile 95% CI",
           "n_boot": n_boot, "mean_block_length": mean_block, "seed": seed, "n_obs": int(len(y)), "models": {}, "differences": {}}
    for m in names:
        prob, pred = models[m]
        out["models"][m] = {"ROC-AUC": _summary(roc_auc(y, prob), auc[m], null=0.5),
                            "Balanced Accuracy": _summary(balanced_accuracy(y, pred), bacc[m], null=0.5)}
    rp, rd = models[reference]
    for m in names:
        if m == reference:
            continue
        prob, pred = models[m]
        out["differences"][f"{reference} - {m}"] = {
            "ROC-AUC": _summary(roc_auc(y, rp) - roc_auc(y, prob), auc[reference] - auc[m], null=0.0),
            "Balanced Accuracy": _summary(balanced_accuracy(y, rd) - balanced_accuracy(y, pred), bacc[reference] - bacc[m], null=0.0),
        }
    return out


def circular_shift_test(y: np.ndarray, prob: np.ndarray, pred: np.ndarray, min_shift: int = 20) -> dict:
    """One-sided p-values for 'better than chance' from all circular shifts of the labels."""
    n = len(y)
    shifts = range(min_shift, n - min_shift + 1)
    auc_null = np.array([roc_auc(np.roll(y, s), prob) for s in shifts])
    bacc_null = np.array([balanced_accuracy(np.roll(y, s), pred) for s in shifts])
    auc_obs, bacc_obs = roc_auc(y, prob), balanced_accuracy(y, pred)
    return {"method": f"circular shift of labels vs predictions, shifts {min_shift}..{n - min_shift}",
            "n_shifts": len(auc_null),
            "ROC-AUC": {"observed": auc_obs, "p_value": float((1 + (auc_null >= auc_obs).sum()) / (1 + len(auc_null))),
                        "null_95th_pct": float(np.percentile(auc_null, 95))},
            "Balanced Accuracy": {"observed": bacc_obs, "p_value": float((1 + (bacc_null >= bacc_obs).sum()) / (1 + len(bacc_null))),
                                  "null_95th_pct": float(np.percentile(bacc_null, 95))}}
