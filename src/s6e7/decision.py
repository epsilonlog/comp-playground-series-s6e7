"""The decision rule: from probabilities to labels. Applied once, after model choice.

Argmax is optimal for plain accuracy only. Balanced accuracy wants
``argmax_k m_k · q_k(x)`` with multipliers near 1/prior — but calibration error bends
the derived value, so the multipliers are *searched* on OOF instead (LEARNING.md).
Scale invariance fixes ``m[0] = 1``, leaving K-1 free parameters; at K=3 that is a
2-dimensional search a coarse-to-fine log grid covers exactly.

The overfitting question is answered by construction, not by hope:

- ``search`` maximises the metric on the rows it is given — an *in-sample* number.
- ``cross_fit`` searches on four folds and scores on the fifth, so the reported score
  never reads a row its own search saw — the honest number.

The gap between the two is a direct measurement of how much the rule overfits. With 2
parameters against ~550k rows per search it should be ~0; if it ever is not, the rule
earned nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from s6e7 import io, metric

#: Search space: multipliers 1..100 on a log grid, per free class. 1/prior for the
#: rarest class here is ~17, comfortably interior.
GRID_LO: float = 1.0
GRID_HI: float = 100.0
GRID_SIZE: int = 25


@dataclass(frozen=True, slots=True)
class CrossFitResult:
    """Cross-fitted rule labels and the bookkeeping to judge them."""

    labels: NDArray[np.int64]
    fold_scores: tuple[float, ...]
    fold_multipliers: tuple[tuple[float, ...], ...]


def apply(proba: NDArray[np.floating], multipliers: NDArray[np.floating]) -> NDArray[np.int64]:
    """Labels under the rule: argmax of the reweighted probabilities."""
    return np.asarray((proba * multipliers).argmax(axis=1), dtype=np.int64)


def search(
    proba: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    size: int = GRID_SIZE,
) -> tuple[NDArray[np.float64], float]:
    """Best multipliers for these rows, by coarse-to-fine log-grid. K=3 only.

    Returns ``(multipliers, in_sample_score)``. The score is in-sample by definition —
    never report it as the experiment's score; that is what `cross_fit` is for.
    """
    if proba.shape[1] != 3:
        msg = f"grid search is written for K=3, got K={proba.shape[1]}"
        raise ValueError(msg)

    def score(a: float, b: float) -> float:
        m = np.array([1.0, a, b])
        return metric.balanced_accuracy(y, apply(proba, m))

    grid = np.logspace(np.log10(GRID_LO), np.log10(GRID_HI), size)
    best_a, best_b, best = 1.0, 1.0, -1.0
    for a in grid:
        for b in grid:
            s = score(a, b)
            if s > best:
                best_a, best_b, best = float(a), float(b), s

    # Refine around the coarse winner: one grid-cell width in each direction.
    step = (GRID_HI / GRID_LO) ** (1.0 / (size - 1))
    for a in np.logspace(np.log10(best_a / step), np.log10(best_a * step), size):
        for b in np.logspace(np.log10(best_b / step), np.log10(best_b * step), size):
            s = score(float(a), float(b))
            if s > best:
                best_a, best_b, best = float(a), float(b), s

    return np.array([1.0, best_a, best_b]), best


def cross_fit(
    proba: NDArray[np.floating],
    y: NDArray[np.integer],
    fold: NDArray[np.integer],
) -> CrossFitResult:
    """Search the rule on four folds, apply it to the fifth. The honest estimate.

    The multipliers per fold should agree closely — five nearly-identical searches on
    80% samples of the same data. Wild disagreement means the rule is chasing noise.
    """
    labels = np.empty(len(y), dtype=np.int64)
    scores: list[float] = []
    fold_ms: list[tuple[float, ...]] = []
    for k in np.unique(fold):
        held_out = fold == k
        m, _ = search(proba[~held_out], y[~held_out])
        labels[held_out] = apply(proba[held_out], m)
        scores.append(metric.balanced_accuracy(y[held_out], labels[held_out]))
        fold_ms.append(tuple(round(float(v), 3) for v in m))
    return CrossFitResult(labels=labels, fold_scores=tuple(scores), fold_multipliers=tuple(fold_ms))


def prior_multipliers(y: NDArray[np.integer]) -> NDArray[np.float64]:
    """The theory-optimal multipliers, ``1/pi_k`` scaled so ``m[0] = 1``.

    Optimal *exactly* when ``proba`` is the true posterior. Balanced accuracy pays
    ``1/(K*n_k)`` per correct row of class k, so a correct row of a class that is
    ``1/pi_k`` times rarer is worth ``1/pi_k`` times more — the multiplier is an
    exchange rate, not a tuning knob. Any distance between this and `search`'s answer
    is calibration error (or landscape flatness), never a better exchange rate.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    prior = counts / counts.sum()
    return np.asarray(prior[0] / prior, dtype=np.float64)


def landscape(
    proba: NDArray[np.floating],
    y: NDArray[np.integer],
    *,
    size: int = 41,
    lo: float = GRID_LO,
    hi: float = GRID_HI,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Balanced accuracy over the whole ``(m1, m2)`` grid — the surface `search` climbs.

    Returns ``(m1_values, m2_values, ba)`` with ``ba[i, j]`` scored at
    ``(1, m1_values[i], m2_values[j])``.

    Exact, and ~15x faster than calling `apply` per grid point, because the label
    depends on the multipliers only through two log thresholds. With
    ``a = log q1 - log q0``, ``b = log q2 - log q0``, ``d = log q1 - log q2`` per row and
    ``la = log m1``, ``lb = log m2``:

    * label 0  <=>  ``a <= -la``  and  ``b <= -lb``
    * else label 1  <=>  ``d >= lb - la``,  else label 2

    So one grid point costs three scalar comparisons per row and no argmax at all.
    """
    if proba.shape[1] != 3:
        msg = f"landscape is written for K=3, got K={proba.shape[1]}"
        raise ValueError(msg)
    q = np.clip(np.asarray(proba, dtype=np.float64), 1e-15, None)
    logq = np.log(q)
    y_arr = np.asarray(y, dtype=np.int64)

    a, b, d = logq[:, 1] - logq[:, 0], logq[:, 2] - logq[:, 0], logq[:, 1] - logq[:, 2]
    groups = [(a[y_arr == k], b[y_arr == k], d[y_arr == k]) for k in range(3)]
    n_k = np.array([len(g[0]) for g in groups], dtype=np.float64)

    grid = np.logspace(np.log10(lo), np.log10(hi), size)
    ba = np.empty((size, size), dtype=np.float64)
    (a0, b0, _), (a1, b1, d1), (a2, b2, d2) = groups
    for i, m1 in enumerate(grid):
        la = float(np.log(m1))
        for j, m2 in enumerate(grid):
            lb = float(np.log(m2))
            tau = lb - la
            hit0 = np.count_nonzero((a0 <= -la) & (b0 <= -lb))
            hit1 = np.count_nonzero(((a1 > -la) | (b1 > -lb)) & (d1 >= tau))
            hit2 = np.count_nonzero(((a2 > -la) | (b2 > -lb)) & (d2 < tau))
            ba[i, j] = (hit0 / n_k[0] + hit1 / n_k[1] + hit2 / n_k[2]) / 3.0
    return grid, grid, ba


def exchange_table(y: NDArray[np.integer]) -> pl.DataFrame:
    """Why the multipliers are what they are: what one correct row is *worth* per class.

    ``balanced_accuracy = (1/K) * sum_k hit_k / n_k``, so one more correct row of class k
    adds exactly ``1/(K*n_k)`` — a fixed price per class, independent of the model. The
    last column is that price relative to the majority class, and it is the multiplier:
    the metric will trade ``ratio`` majority rows for one minority row and break even.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64))
    k = len(counts)
    value = 1.0 / (k * counts)
    return pl.DataFrame(
        {
            "class": list(io.CLASSES[:k]),
            "n_rows": counts.tolist(),
            "prior": (counts / counts.sum()).tolist(),
            "ba_per_correct_row": value.tolist(),
            "worth_vs_majority": (value / value[0]).tolist(),
        }
    )


def rule_effect(
    proba: NDArray[np.floating],
    y: NDArray[np.integer],
    multipliers: NDArray[np.floating],
) -> pl.DataFrame:
    """What the rule actually buys, per class: recall before (argmax) and after.

    The shape to expect from a prior correction is a large majority-class recall loss
    paying for larger minority-class gains — the trade `exchange_table` prices. If the
    majority loss is *not* the biggest number here, the multipliers are doing something
    other than correcting the prior.
    """
    y_arr = np.asarray(y, dtype=np.int64)
    before = np.asarray(proba).argmax(axis=1)
    after = apply(proba, multipliers)
    rows = []
    for k in range(proba.shape[1]):
        mask = y_arr == k
        r_before = float((before[mask] == k).mean())
        r_after = float((after[mask] == k).mean())
        rows.append(
            {
                "class": io.CLASSES[k],
                "n_rows": int(mask.sum()),
                "recall_argmax": r_before,
                "recall_rule": r_after,
                "delta": r_after - r_before,
                "rows_gained": int((after[mask] == k).sum() - (before[mask] == k).sum()),
            }
        )
    return pl.DataFrame(rows)
