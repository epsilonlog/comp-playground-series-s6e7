"""EDA and diagnostics plotting. Claude-owned since 2026-08-30.

Contract — every function:
    * takes a Polars DataFrame (converts internally if a library needs otherwise)
    * returns a ``matplotlib.figure.Figure`` — built directly, never through pyplot,
      so notebooks display each figure exactly once and no global state accumulates
    * never calls ``plt.show()`` or saves; the caller decides
    * never mutates the input frame
    * docstring names the decision the figure informs

The point isn't pretty charts. A panel that cannot fail teaches nothing — every panel
here has a reading that would change a decision.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from s6e7 import folds, io, metric

__all__ = [
    "categorical_grid",
    "correlation",
    "fold_distribution",
    "importance",
    "missingness",
    "numeric_grid",
    "oof_diagnostics",
    "target_overview",
    "train_test_shift",
]

PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#64B5CD")
TRAIN_COLOR, TEST_COLOR = "#4C72B0", "#C44E52"


def _grid(
    n: int, ncols: int = 3, panel: tuple[float, float] = (4.2, 3.2)
) -> tuple[Figure, list[Axes]]:
    """Figure with n subplots in a grid, unused axes hidden, flat list returned."""
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig = Figure(figsize=(ncols * panel[0], nrows * panel[1]), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False).ravel().tolist()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


def _declared_levels(col: str) -> tuple[str, ...] | None:
    return {**io.ORDINAL_LEVELS, **io.NOMINAL_LEVELS}.get(col)


# --- Tier 1 --------------------------------------------------------------------


def target_overview(df: pl.DataFrame, target: str) -> Figure:
    """Decision: fold scheme and metric expectations — how imbalanced, what is chance.

    Class counts with shares annotated; chance balanced accuracy (1/K) in the title so
    every later score is read against the real floor, not 0.5.
    """
    counts = df[target].value_counts().sort("count", descending=True)
    labels, values = counts[target].to_list(), counts["count"].to_list()
    k = len(labels)

    fig, (ax,) = _grid(1, panel=(6.0, 3.6))
    bars = ax.bar(labels, values, color=PALETTE[:k])
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v:,}\n{100 * v / df.height:.1f}%",
            (bar.get_x() + bar.get_width() / 2, v),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("rows")
    ax.set_title(f"{target} — {df.height:,} rows, K={k}, chance BA = 1/{k} = {1 / k:.3f}")
    return fig


def missingness(df: pl.DataFrame) -> Figure:
    """Decision: whether missingness is structural (a feature) or independent noise.

    Left: null share per column. Right: pairwise co-occurrence *lift* — observed
    co-null rate over the independence expectation ``p_a·p_b``. Lift ≈ 1 everywhere
    means independent nulls; a block of lift ≫ 1 means a shared mask, which is usually
    predictive on its own. Equal counts alone cannot distinguish the two (LEARNING.md).
    """
    cols = [c for c in df.columns if df[c].null_count() > 0]
    fig, axes = _grid(2, ncols=2, panel=(5.4, 0.42 * max(len(cols), 8)))
    if not cols:
        axes[0].set_title("no nulls anywhere")
        axes[1].set_visible(False)
        return fig

    shares = np.array([df[c].null_count() / df.height for c in cols])
    order = np.argsort(shares)
    axes[0].barh([cols[i] for i in order], shares[order] * 100, color=TRAIN_COLOR)
    axes[0].set_xlabel("% null")
    axes[0].set_title(f"null share per column ({df.height:,} rows)")

    if len(cols) >= 2:
        mask = df.select(pl.col(c).is_null().cast(pl.Float64) for c in cols).to_numpy()
        p = mask.mean(axis=0)
        observed = (mask.T @ mask) / mask.shape[0]
        lift = observed / np.outer(p, p)
        np.fill_diagonal(lift, np.nan)
        im = axes[1].imshow(lift, cmap="coolwarm", vmin=0.0, vmax=2.0)
        axes[1].set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
        axes[1].set_yticks(range(len(cols)), cols, fontsize=8)
        axes[1].set_title("co-null lift (1 = independent)")
        fig.colorbar(im, ax=axes[1], shrink=0.8)
    else:
        axes[1].set_visible(False)
    return fig


def numeric_grid(df: pl.DataFrame, cols: Sequence[str], target: str | None = None) -> Figure:
    """Decision: which numerics separate the classes, and which distributions look coded.

    Per column: density histogram, per-class overlay when a target is given. Bins span
    the 0.5-99.5 percentile so one outlier cannot flatten the picture; the null share
    sits in each subtitle because a null is a value the histogram cannot show.
    """
    classes = df[target].drop_nulls().unique().sort().to_list() if target else [None]
    fig, axes = _grid(len(cols))
    for ax, col in zip(axes, cols, strict=False):
        values = df[col].drop_nulls().drop_nans().to_numpy()
        lo, hi = np.quantile(values, [0.005, 0.995])
        bins = np.linspace(lo, hi, 60)
        for i, cls in enumerate(classes):
            sel = (
                values
                if cls is None
                else df.filter(pl.col(target) == cls)[col].drop_nulls().drop_nans().to_numpy()
            )
            hist, edges = np.histogram(sel, bins=bins, density=True)
            ax.stairs(
                hist,
                edges,
                color=PALETTE[i % len(PALETTE)],
                label=str(cls),
                fill=cls is None,
                alpha=0.9,
            )
        null_pct = 100 * df[col].null_count() / df.height
        ax.set_title(f"{col}  (null {null_pct:.1f}%)", fontsize=10)
        ax.set_yticks([])
    if target:
        axes[0].legend(fontsize=8)
    return fig


def categorical_grid(df: pl.DataFrame, cols: Sequence[str], target: str) -> Figure:
    """Decision: encoding per column — which levels move the target, and by how much.

    Bars: level shares (declared order where one exists, else by count; nulls shown as
    their own level). Markers: per-class rate *deviation from the global rate* in
    percentage points on a twin axis — deviation, because 8.47% vs 8.51% on a 0-100%
    axis is invisible and the deviation is the entire content.
    """
    classes = df[target].drop_nulls().unique().sort().to_list()
    global_rate = {cls: float((df[target] == cls).mean()) for cls in classes}
    fig, axes = _grid(len(cols), panel=(4.6, 3.4))

    for ax, col in zip(axes, cols, strict=False):
        agg = (
            df.group_by(col)
            .agg(
                pl.len().alias("n"),
                *[(pl.col(target) == cls).mean().alias(str(cls)) for cls in classes],
            )
            .with_columns(pl.col(col).cast(pl.String).fill_null("(null)"))
        )
        declared = _declared_levels(col)
        if declared:
            rank = {lvl: i for i, lvl in enumerate(declared)}
            agg = agg.sort(pl.col(col).replace_strict(rank, default=len(rank)))
        else:
            agg = agg.sort("n", descending=True).head(15)

        levels = agg[col].to_list()
        x = np.arange(len(levels))
        ax.bar(x, (agg["n"] / df.height * 100).to_numpy(), color="#CCCCCC", width=0.6)
        ax.set_ylabel("% of rows", fontsize=8)
        ax.set_xticks(x, levels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{col}  ({agg.height} levels)", fontsize=10)

        twin = ax.twinx()
        for i, cls in enumerate(classes):
            dev = (agg[str(cls)].to_numpy() - global_rate[cls]) * 100
            twin.plot(
                x, dev, marker="o", ms=4, lw=1.2, color=PALETTE[i % len(PALETTE)], label=str(cls)
            )
        twin.axhline(0, color="black", lw=0.6, ls="--")
        twin.set_ylabel("rate - global (pp)", fontsize=8)
        if ax is axes[0]:
            twin.legend(fontsize=7, loc="upper right")
    return fig


def correlation(df: pl.DataFrame, cols: Sequence[str], method: str = "spearman") -> Figure:
    """Decision: which features are redundant. Read r² not r — the cut sits near |r|=0.95.

    Complete cases across `cols` (n in the title); Spearman via ranks by default.
    """
    frame = df.select(cols).drop_nulls()
    matrix = frame.to_numpy().astype(np.float64)
    if method == "spearman":
        matrix = frame.select(pl.col(c).rank() for c in cols).to_numpy().astype(np.float64)
    corr = np.corrcoef(matrix, rowvar=False)
    corr[np.triu_indices_from(corr)] = np.nan

    fig, (ax,) = _grid(1, panel=(0.65 * len(cols) + 2.5, 0.55 * len(cols) + 2.0))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i):
            ax.annotate(f"{corr[i, j]:.2f}", (j, i), ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.set_title(f"{method} correlation — {frame.height:,} complete rows")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig


# --- Tier 2: the ones that actually win competitions ----------------------------


def train_test_shift(train: pl.DataFrame, test: pl.DataFrame, cols: Sequence[str]) -> Figure:
    """Decision: whether CV can be trusted — the visual companion to adversarial AUC.

    Per column, train/test distributions overlaid with a distance in the subtitle (KS
    for numerics, total-variation for categoricals), sorted worst-first. Remember the
    limit this has: 13 clean marginals still hid a joint AUC of 0.65 — this shows
    *what* differs, the adversarial AUC decides *whether* anything does.
    """
    from scipy.stats import ks_2samp

    stats: list[tuple[str, float, bool]] = []
    for col in cols:
        numeric = train.schema[col].is_numeric()
        if numeric:
            a = train[col].drop_nulls().drop_nans().to_numpy()
            b = test[col].drop_nulls().drop_nans().to_numpy()
            stats.append((col, float(ks_2samp(a, b).statistic), numeric))
        else:
            pa = train[col].fill_null("(null)").value_counts(normalize=True)
            pb = test[col].fill_null("(null)").value_counts(normalize=True)
            joined = pa.join(pb, on=col, how="full", coalesce=True).fill_null(0.0)
            tv = 0.5 * float((joined["proportion"] - joined["proportion_right"]).abs().sum())
            stats.append((col, tv, numeric))
    stats.sort(key=lambda t: t[1], reverse=True)

    fig, axes = _grid(len(stats))
    for ax, (col, dist, numeric) in zip(axes, stats, strict=False):
        if numeric:
            a = train[col].drop_nulls().drop_nans().to_numpy()
            b = test[col].drop_nulls().drop_nans().to_numpy()
            lo, hi = np.quantile(np.concatenate([a[:200_000], b[:200_000]]), [0.005, 0.995])
            bins = np.linspace(lo, hi, 60)
            for arr, color, label in ((a, TRAIN_COLOR, "train"), (b, TEST_COLOR, "test")):
                hist, edges = np.histogram(arr, bins=bins, density=True)
                ax.stairs(hist, edges, color=color, label=label)
            ax.set_title(f"{col} — KS {dist:.4f}", fontsize=10)
        else:
            pa = train[col].fill_null("(null)").value_counts(normalize=True)
            pb = test[col].fill_null("(null)").value_counts(normalize=True)
            # Full join: a level present in only one file must still get a bar — a
            # brand-new test-only category is the most alarming categorical shift.
            merged = (
                pa.join(pb, on=col, how="full", coalesce=True)
                .fill_null(0.0)
                .sort("proportion", descending=True)
            )
            x = np.arange(merged.height)
            ax.bar(x - 0.2, merged["proportion"] * 100, 0.4, color=TRAIN_COLOR, label="train")
            ax.bar(x + 0.2, merged["proportion_right"] * 100, 0.4, color=TEST_COLOR, label="test")
            ax.set_xticks(x, merged[col].to_list(), rotation=30, ha="right", fontsize=8)
            ax.set_title(f"{col} — TV {dist:.4f}", fontsize=10)
        ax.set_yticks([])
    axes[0].legend(fontsize=8)
    return fig


def fold_distribution(
    df: pl.DataFrame,
    fold: NDArray[np.integer],
    target: str,
    *,
    test: pl.DataFrame | None = None,
) -> Figure:
    """Decision: did the freeze produce what the design promised — and how far is test.

    Left, the stratified dimension as a receipt: per-fold class counts as *deviation
    from the class mean, in rows* (a 0-100% axis would show five identical bars).
    Right, the unstratified dimension, where the information is: null-bucket shares as
    deviation from pooled train, with a ±2 SE band (inside = binomial drift, outside =
    bug) and test as the dashed reference — the honest answer to "are my folds like
    test?" is "no, deliberately, and this is how far".
    """
    fold = np.asarray(fold)
    fold_ids = np.unique(fold)
    classes = df[target].drop_nulls().unique().sort().to_list()
    y = df[target].to_numpy()
    n_fold = float(np.mean([(fold == k).sum() for k in fold_ids]))

    fig, axes = _grid(2, ncols=2, panel=(5.8, 3.8))

    counts = np.array(
        [[(y[fold == k] == cls).sum() for k in fold_ids] for cls in classes], dtype=float
    )
    dev = counts - counts.mean(axis=1, keepdims=True)
    width = 0.8 / len(fold_ids)
    x = np.arange(len(classes))
    for j, k in enumerate(fold_ids):
        axes[0].bar(x + (j - len(fold_ids) / 2 + 0.5) * width, dev[:, j], width, label=f"fold {k}")
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_xticks(
        x,
        [f"{c}\n~{m:,.0f}/fold" for c, m in zip(classes, counts.mean(axis=1), strict=True)],
        fontsize=8,
    )
    axes[0].set_ylabel("rows - class mean")
    axes[0].set_title(f"stratified dimension (receipt) — {len(fold_ids)} folds of ~{n_fold:,.0f}")
    axes[0].legend(fontsize=7)

    buckets = folds.null_bucket(df).to_numpy()
    bucket_ids = list(range(folds.NULL_BUCKET_CAP + 1))
    labels = [f"k={b}" if b < folds.NULL_BUCKET_CAP else f"k≥{b}" for b in bucket_ids]
    pooled = np.array([(buckets == b).mean() for b in bucket_ids])
    share = np.array([[(buckets[fold == k] == b).mean() for k in fold_ids] for b in bucket_ids])
    dev_pp = (share - pooled[:, None]) * 100
    band = 2 * np.sqrt(pooled * (1 - pooled) / n_fold) * 100

    x = np.arange(len(bucket_ids))
    for i in x:
        axes[1].fill_between([i - 0.42, i + 0.42], -band[i], band[i], color="#DDDDDD", zorder=0)
    for j, _k in enumerate(fold_ids):
        axes[1].bar(x + (j - len(fold_ids) / 2 + 0.5) * width, dev_pp[:, j], width, zorder=2)
    if test is not None:
        tb = folds.null_bucket(test).to_numpy()
        test_dev = np.array([(tb == b).mean() - pooled[b] for b in bucket_ids]) * 100
        axes[1].plot(x, test_dev, "D--", color="black", ms=5, lw=1.2, label="test", zorder=3)
        axes[1].legend(fontsize=8)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xticks(
        x, [f"{lbl}\n({100 * p:.2f}%)" for lbl, p in zip(labels, pooled, strict=True)], fontsize=8
    )
    axes[1].set_ylabel("share - pooled train (pp)")
    axes[1].set_title("unstratified dimension — grey band = ±2 SE")
    return fig


def oof_diagnostics(
    y_true: NDArray[Any],
    oof: NDArray[np.floating],
    *,
    class_names: Sequence[str] | None = None,
) -> Figure:
    """Decision: what to fix next — which class leaks recall, and whether the
    probabilities can be trusted by the multiplier search.

    Confusion (row-normalised, diagonal = recall); reliability per class (the multiplier
    search assumes roughly multiplicative distortion — a crossing curve says calibrate
    instead); per-class recall against chance; confidence split by correctness.
    """
    names = (
        list(class_names)
        if class_names is not None
        else (
            list(io.CLASSES)
            if oof.shape[1] == len(io.CLASSES)
            else [str(i) for i in range(oof.shape[1])]
        )
    )
    k = len(names)
    y_arr = np.asarray(y_true)
    if not np.issubdtype(y_arr.dtype, np.integer):
        lookup = {name: i for i, name in enumerate(names)}
        y_arr = np.array([lookup[str(v)] for v in y_arr], dtype=np.int64)
    pred = oof.argmax(axis=1)

    fig, axes = _grid(4, ncols=2, panel=(5.2, 3.8))

    matrix, _ = metric.confusion(y_arr, pred, labels=range(k))
    recall_matrix = matrix / matrix.sum(axis=1, keepdims=True)
    im = axes[0].imshow(recall_matrix, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            axes[0].annotate(
                f"{recall_matrix[i, j]:.3f}",
                (j, i),
                ha="center",
                va="center",
                color="white" if recall_matrix[i, j] > 0.6 else "black",
                fontsize=9,
            )
    axes[0].set_xticks(range(k), names, fontsize=8)
    axes[0].set_yticks(range(k), names, fontsize=8)
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("true")
    ba = metric.balanced_accuracy(y_arr, pred)
    axes[0].set_title(f"confusion (row-normalised) — BA {ba:.5f}")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    edges = np.linspace(0, 1, 11)
    for i, name in enumerate(names):
        p = oof[:, i]
        which = np.digitize(p, edges[1:-1])
        centers, observed = [], []
        for b in range(10):
            sel = which == b
            if sel.sum() >= 50:
                centers.append(float(p[sel].mean()))
                observed.append(float((y_arr[sel] == i).mean()))
        axes[1].plot(
            centers, observed, marker="o", ms=3, lw=1.2, color=PALETTE[i % len(PALETTE)], label=name
        )
    axes[1].plot([0, 1], [0, 1], ls="--", color="black", lw=0.8)
    axes[1].set_xlabel("predicted probability")
    axes[1].set_ylabel("observed rate")
    axes[1].set_title("reliability per class")
    axes[1].legend(fontsize=8)

    recalls = np.diag(recall_matrix)
    axes[2].bar(names, recalls, color=[PALETTE[i % len(PALETTE)] for i in range(k)])
    axes[2].axhline(1 / k, ls="--", color="black", lw=0.8, label=f"chance 1/{k}")
    for i, r in enumerate(recalls):
        axes[2].annotate(f"{r:.3f}", (i, r), ha="center", va="bottom", fontsize=9)
    axes[2].set_ylim(0, 1.1)
    axes[2].set_title("per-class recall — the macro average weighs these equally")
    axes[2].legend(fontsize=8)

    confidence = oof.max(axis=1)
    correct = pred == y_arr
    bins = np.linspace(1 / k, 1, 40)
    for sel, color, label in ((correct, "#55A868", "correct"), (~correct, "#C44E52", "wrong")):
        hist, e = np.histogram(confidence[sel], bins=bins, density=True)
        axes[3].stairs(hist, e, color=color, label=label, fill=True, alpha=0.4)
    axes[3].set_xlabel("max predicted probability")
    axes[3].set_yticks([])
    axes[3].set_title("confidence by correctness")
    axes[3].legend(fontsize=8)
    return fig


def experiment_compare(
    experiments: pl.DataFrame, *, n_folds: int = 5, resolution: float = 0.001
) -> Figure:
    """Decision: which experiments moved the needle, and whether the trajectory has
    flattened — the when-to-stop dial.

    Points are cv_mean with ±2·cv_std/√n_folds error bars; the grey band is the best
    score ± the harness resolution, so any point inside it is statistically the same
    experiment as the best. Black diamonds are public LB where recorded — they should
    ride ~0.001-0.002 below CV (the standing prediction); a point breaking that pattern
    is the alarm, not the gap itself.
    """
    mean = experiments["cv_mean"].to_numpy().astype(np.float64)
    err = 2.0 * experiments["cv_std"].to_numpy().astype(np.float64) / math.sqrt(n_folds)
    labels = experiments["exp_id"].to_list()
    x = np.arange(len(labels))
    best = float(mean.max())

    fig, (ax,) = _grid(1, panel=(max(6.5, 0.9 * len(labels) + 2.5), 4.0))
    ax.axhspan(
        best - resolution,
        best + resolution,
        color="#EEEEEE",
        zorder=0,
        label=f"best ± {resolution} (resolution)",
    )
    ax.errorbar(
        x, mean, yerr=err, fmt="o", color=TRAIN_COLOR, capsize=3, zorder=2, label="cv_mean ± 2 SE"
    )
    if "lb_public" in experiments.columns:
        lb = experiments["lb_public"].cast(pl.Float64, strict=False).to_numpy()
        has_lb = np.isfinite(lb.astype(np.float64))
        if has_lb.any():
            ax.plot(x[has_lb], lb[has_lb], "D", color="black", ms=5, zorder=3, label="public LB")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("experiment trajectory — flat inside the band means the axis is mined out")
    ax.legend(fontsize=8, loc="lower right")
    return fig


def resolution_demo(
    se: float = 0.00103, *, effects: Sequence[float] = (0.0005, 0.002, 0.005)
) -> Figure:
    """Decision: which effect sizes this harness can confirm, and which it cannot.

    The bell is the sampling distribution of `cv_mean` when NOTHING changed — pure
    validation luck, width `se` (~0.001 here, derived in LEARNING.md). Judge a candidate
    gain by how far up the tail it sits: inside ~2 SE it is indistinguishable from luck
    on a single run; far outside, a result. Paired comparisons on the frozen folds
    resolve finer than this single-score picture — the bell is the conservative bar.
    """
    x = np.linspace(-3.5 * se, max(effects) + 3.0 * se, 600)
    pdf = np.exp(-0.5 * (x / se) ** 2)

    fig, (ax,) = _grid(1, panel=(7.5, 3.8))
    ax.fill_between(x, pdf, color="#CCD9EA", label="cv_mean wobble when nothing changed")
    for e, color in zip(effects, PALETTE[1:], strict=False):
        z = e / se
        verdict = "luck" if z < 2 else ("borderline" if z < 3 else "real")
        ax.axvline(e, color=color, lw=1.5)
        ax.annotate(
            f"+{e:.4f}\n{z:.1f} SE - {verdict}",
            (e, 1.04),
            ha="center",
            fontsize=8,
            color=color,
        )
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xlabel("change in cv_mean")
    ax.set_title(f"what a gain must clear - SE(cv_mean) = {se:.5f}")
    ax.legend(fontsize=8, loc="upper right")
    return fig


def importance(model: Any, feature_names: Sequence[str], top: int = 30) -> Figure:
    """Decision: what to prune, and whether a leak exists (one feature towering is one).

    Accepts a single fitted model or a sequence (the five fold models): mean ± std
    across folds, because a feature important in one fold only is noise. Note the
    known bias: split/gain importance inflates high-cardinality features — the
    adversarial run ranked a chance-level column first on exactly that (LEARNING.md).
    """
    models = model if isinstance(model, (list, tuple)) else [model]
    values = np.vstack([np.asarray(m.feature_importances_, dtype=float) for m in models])
    mean, std = values.mean(axis=0), values.std(axis=0)
    order = np.argsort(mean)[::-1][:top][::-1]

    fig, (ax,) = _grid(1, panel=(6.5, 0.32 * len(order) + 1.2))
    ax.barh(
        [feature_names[i] for i in order],
        mean[order],
        xerr=std[order] if len(models) > 1 else None,
        color=TRAIN_COLOR,
    )
    kind = getattr(models[0], "importance_type", "importance")
    ax.set_xlabel(f"{kind} (mean of {len(models)} model{'s' if len(models) > 1 else ''})")
    ax.set_title(f"top {len(order)} features")
    return fig
