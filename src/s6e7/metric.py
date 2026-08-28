"""The competition metric: balanced accuracy.

Submissions are scored on the macro-average of per-class recall. This module is the
single scoring call site — `cv.py`, the decision-rule search, and `oof_diagnostics`
all go through it.

`sklearn.metrics.balanced_accuracy_score` is the reference; `tests/test_metric.py`
asserts agreement with it. We reimplement rather than call it because the decision-rule
search evaluates this hundreds of times over ~690k rows, and because the behaviour when
a class is absent from `y_true` needs pinning down explicitly rather than inheriting a
warning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def confusion(
    y_true: ArrayLike, y_pred: ArrayLike, *, labels: ArrayLike | None = None
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Confusion matrix and the label set it is indexed by.

    Rows are true classes, columns predicted. Labels default to the sorted union of
    those present in either array, matching ``sklearn.metrics.confusion_matrix``.
    """
    true_arr = np.asarray(y_true).ravel()
    pred_arr = np.asarray(y_pred).ravel()
    if true_arr.shape != pred_arr.shape:
        msg = f"length mismatch: y_true has {true_arr.size}, y_pred has {pred_arr.size}"
        raise ValueError(msg)

    label_arr = np.union1d(true_arr, pred_arr) if labels is None else np.asarray(labels).ravel()
    n = label_arr.size
    if n == 0:
        return np.zeros((0, 0), dtype=np.int64), label_arr

    # searchsorted maps arbitrary labels (ints, strings) to 0..n-1 positions.
    true_idx = np.searchsorted(label_arr, true_arr)
    pred_idx = np.searchsorted(label_arr, pred_arr)

    flat = np.bincount(true_idx * n + pred_idx, minlength=n * n)
    return flat.reshape(n, n).astype(np.int64), label_arr


def balanced_accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, *, labels: ArrayLike | None = None, adjusted: bool = False
) -> float:
    """Macro-average of per-class recall.

    Each class contributes equally regardless of its size, so the floor for a
    degenerate always-one-class prediction is ``1 / n_classes``, not the majority rate.

    Classes with no support in ``y_true`` have undefined recall and are dropped from the
    average — this mirrors sklearn, which additionally emits a warning. Pass ``labels``
    to pin the class set explicitly; absent classes are still dropped, since their
    recall cannot be computed at all.

    With ``adjusted=True`` the score is rescaled so chance performance is 0.0.
    """
    matrix, _ = confusion(y_true, y_pred, labels=labels)
    support = matrix.sum(axis=1)

    present = support > 0
    if not present.any():
        return float("nan")

    recall = np.diag(matrix)[present] / support[present]
    score = float(recall.mean())

    if adjusted:
        chance = 1.0 / int(present.sum())
        score = (score - chance) / (1.0 - chance)
    return score
