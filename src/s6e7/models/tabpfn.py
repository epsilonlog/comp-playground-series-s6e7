"""TabPFN builder — in-context learning as the third model family.

TabPFN (Prior Labs, ``pip install tabpfn``) is a transformer pre-trained on synthetic
tabular tasks: there is no gradient step on our data. ``fit`` stores the training rows
as *context*, and ``predict_proba`` attends from each query row to that context. It is
the most different learner available to this project, which is exactly what the blend
measured itself missing (notebook 07: GBDT error correlation 0.885, the FT-Transformer
the only measured source of decorrelation). Used here as a **baseline**: library
defaults, no tuning.

Two decisions the wrapper makes, both recorded in the params so the ledger can see them:

- **Context size.** The default TabPFN-3 checkpoint admits 1,000,000 x 200, so the
  full 552k fit rows are *allowed*; but attention over the context is quadratic in
  wall clock and memory, and this is a baseline. ``max_fit_rows`` takes a
  class-proportional (stratified) subsample of the fit rows as the context — the same
  priors as the full data, so the model's probabilities keep their meaning. Raise it in
  a config once the first run has priced a fold.
- **Prior correction.** ``balance_probabilities=True`` divides each predicted
  probability by the class prior observed in the context and renormalises — the
  zero-parameter ``1/pi`` rule notebook 07 measured at 0.94932 on the LGBM OOF, applied
  inside ``predict_proba``. The OOF argmax is therefore the metric-correct decision,
  comparable with exp_0004/exp_0017 without a separate rule run.

NaN is handled natively by TabPFN, so the baseline matrix goes in as is; the six
categorical code columns are declared through ``categorical_features_indices``, and any
extra columns (the exact-value TE columns ``cv.run`` appends when a config names an
encoder) are numeric.

The model weights are gated: a free Prior Labs account, licence acceptance, and a
``TABPFN_TOKEN`` environment variable (headless). Neither the library nor the weights are
installed locally — training happens on Kaggle GPU (``notebooks/kaggle_tabpfn/``) — so
the import is deferred to fit time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from s6e7 import features, io
from s6e7.config import SEED
from s6e7.protocols import Classifier
from s6e7.registry import register

TABPFN_PARAMS: dict[str, Any] = {
    "max_fit_rows": 100_000,
    "n_estimators": 8,  # library default
    "balance_probabilities": True,
    "fit_mode": "fit_with_cache",  # context encoded once, reused by every predict call
    "ignore_pretraining_limits": False,
    "device": "auto",
    "predict_chunk": 100_000,  # rows per predict_proba call; the library chunks further
    "seed": SEED,
}


def stratified_context(
    y: NDArray[np.integer[Any]], max_rows: int, *, seed: int = SEED
) -> NDArray[np.int64]:
    """Sorted indices of a class-proportional subsample of at most `max_rows` rows.

    Every row when there are no more than `max_rows`. Otherwise a stratified draw, so the
    context carries the same class priors as the data it stands in for.
    """
    n = len(y)
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)
    from sklearn.model_selection import train_test_split

    keep, _ = train_test_split(
        np.arange(n, dtype=np.int64), train_size=max_rows, stratify=y, random_state=seed
    )
    return np.sort(np.asarray(keep, dtype=np.int64))


class TabPFNContext:
    """TabPFN behind the harness protocol: float32 matrix in, probabilities out."""

    def __init__(self, params: dict[str, Any]) -> None:
        self._p: dict[str, Any] = {**TABPFN_PARAMS, **params}
        self._model: Any = None
        self.n_context_: int = 0

    @property
    def classes_(self) -> NDArray[np.int64]:
        return np.arange(len(io.CLASSES), dtype=np.int64)

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]) -> TabPFNContext:
        from tabpfn import TabPFNClassifier

        p = self._p
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        idx = stratified_context(y_arr, int(p["max_fit_rows"]), seed=int(p["seed"]))
        self.n_context_ = int(idx.size)
        self._model = TabPFNClassifier(
            n_estimators=int(p["n_estimators"]),
            balance_probabilities=bool(p["balance_probabilities"]),
            fit_mode=p["fit_mode"],
            ignore_pretraining_limits=bool(p["ignore_pretraining_limits"]),
            device=p["device"],
            random_state=int(p["seed"]),
            categorical_features_indices=list(features.CATEGORICAL_IDX),
        )
        self._model.fit(X_arr[idx], y_arr[idx])
        if list(self._model.classes_) != list(range(len(io.CLASSES))):
            msg = f"context is missing a class: classes_={list(self._model.classes_)}"
            raise RuntimeError(msg)
        return self

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        if self._model is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        X_arr = np.asarray(X, dtype=np.float32)
        chunk = int(self._p["predict_chunk"])
        parts = [
            np.asarray(self._model.predict_proba(X_arr[i : i + chunk]), dtype=np.float64)
            for i in range(0, len(X_arr), chunk)
        ]
        return np.vstack(parts)


@register("tabpfn")
def build_tabpfn(params: dict[str, Any]) -> Classifier:
    return TabPFNContext(params)
