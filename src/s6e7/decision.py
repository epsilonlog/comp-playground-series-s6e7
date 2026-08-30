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
from numpy.typing import NDArray

from s6e7 import metric

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
