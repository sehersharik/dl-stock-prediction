"""Probability quality for the direction model: reliability, Brier score and log loss.

Descriptive only (nothing is fitted). Each split is compared with a no-skill reference that
always predicts the training-set UP share, so "how good is a Brier score of 0.25?" has an
answer: the Brier skill score is 1 - Brier / Brier_reference (0 = no better than the
constant forecast, < 0 = worse).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def wilson_interval(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    phat = k / n
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) / denom
    return [float(centre - half), float(centre + half)]


def reliability_bins(y, p, n_bins=5):
    """Equal-count (quantile) bins: the probabilities span a narrow range, so fixed-width
    bins would leave most bins empty."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-12
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        k, n = int(y[m].sum()), int(m.sum())
        bins.append({"p_low": float(lo), "p_high": float(min(hi, p.max())), "n": n,
                     "mean_predicted": float(p[m].mean()) if n else float("nan"),
                     "observed_up_rate": float(k / n) if n else float("nan"),
                     "observed_ci95": wilson_interval(k, n)})
    return bins


def summarize(y, p, reference_rate, n_bins=5):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ref = np.full_like(p, reference_rate)
    b, b_ref = brier(y, p), brier(y, ref)
    ll, ll_ref = log_loss(y, p), log_loss(y, ref)
    return {"n": int(len(y)), "observed_up_rate": float(y.mean()),
            "prob_min": float(p.min()), "prob_max": float(p.max()), "prob_mean": float(p.mean()), "prob_std": float(p.std(ddof=1)),
            "share_within_0.45_0.55": float(((p > 0.45) & (p < 0.55)).mean()),
            "brier": b, "brier_reference": b_ref, "brier_skill_score": float(1 - b / b_ref),
            "log_loss": ll, "log_loss_reference": ll_ref,
            "bins": reliability_bins(y, p, n_bins)}


def plot_reliability(results: dict, probs: dict, path, title: str):
    """results: {split: summarize(...) output}; probs: {split: predicted probabilities}."""
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Perfect calibration")
    colors = {"validation": "#2563EB", "test": "#DC2626"}
    for split, r in results.items():
        xs = [b["mean_predicted"] for b in r["bins"]]
        ys = [b["observed_up_rate"] for b in r["bins"]]
        err = np.array([[y - b["observed_ci95"][0], b["observed_ci95"][1] - y] for b, y in zip(r["bins"], ys)]).T
        ax.errorbar(xs, ys, yerr=err, marker="o", capsize=3, color=colors.get(split), label=f"{split} (Brier {r['brier']:.4f})")
        axh.hist(probs[split], bins=np.arange(0.3, 0.7001, 0.005), alpha=0.5, color=colors.get(split),
                 label=f"{split}: P(UP) {r['prob_min']:.3f}–{r['prob_max']:.3f}")
    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel("Observed share of UP days (95% Wilson CI)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    axh.set_xlabel("Predicted P(UP)")
    axh.set_ylabel("Days")
    axh.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
