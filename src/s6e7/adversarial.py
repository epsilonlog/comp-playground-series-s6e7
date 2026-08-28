"""Adversarial validation — can a classifier tell a train row from a test row?

Workflow step 3. The fold design rests on a premise: train and test are draws from one
generator, so a validation fold is a fair stand-in for the test set. This is the test of
that premise, and it is the only one that looks at *all* features jointly. EDA compared
marginals one column at a time; a shift can hide in the joint distribution with every
marginal intact.

Reading the result:

    AUC ~ 0.50   distributions match. Random folds are fair. Proceed.
    AUC ~ 0.55   mild shift. Read the importances; usually one column drifting.
    AUC > 0.70   real shift. Either drop the leaking columns or design folds that
                 reproduce the difference, and revisit before freezing.

**Gain importances are only meaningful when the AUC says there is something to explain.**
At AUC 0.50 the model found nothing and the importance ranking is a ranking of noise —
some column has to come top. Read the AUC first, the table second.

`id` is excluded by default. Competition ids are assigned per file, so train and test
occupy disjoint ranges and a single split on `id` separates them perfectly — AUC 1.0 that
says nothing whatever about the feature distributions. Check id ranges separately.

This module trains a model before the CV harness exists, which CLAUDE.md rule 2 otherwise
forbids. Adversarial validation *is* part of the harness: it is what licenses the fold
design. No competition model is built here and nothing it produces is submitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from s6e7 import io
from s6e7.config import N_JOBS, SEED

if TYPE_CHECKING:
    import pandas as pd

#: Deliberately unremarkable. This model is a measuring instrument, not a contender —
#: tuning it would only make it better at finding a difference that should not exist.
DEFAULT_PARAMS: Final[dict[str, Any]] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "n_jobs": N_JOBS,
    "random_state": SEED,
    "verbosity": -1,
}

#: Above this, the i.i.d. premise is dead and fold design has to change.
SHIFT_THRESHOLD: Final[float] = 0.55


@dataclass(frozen=True, slots=True)
class AdversarialResult:
    """Everything the run produced. `auc` is the headline; the rest is the audit trail."""

    auc: float
    fold_aucs: tuple[float, ...]
    importance: pl.DataFrame
    oof: np.ndarray
    n_train: int
    n_test: int
    features: tuple[str, ...]

    @property
    def auc_std(self) -> float:
        """Spread across folds. A large spread on a near-0.5 AUC is just noise."""
        return float(np.std(self.fold_aucs))

    @property
    def shifted(self) -> bool:
        return self.auc > SHIFT_THRESHOLD

    def report(self) -> str:
        """One human-readable block. Printed by the notebook, not parsed by anything."""
        verdict = (
            f"SHIFT — AUC {self.auc:.4f} > {SHIFT_THRESHOLD}. Read the importances; "
            "revisit fold design before freezing."
            if self.shifted
            else f"NO SHIFT — AUC {self.auc:.4f}. Train and test are one distribution; "
            "random stratified folds are a fair stand-in for test."
        )
        rows = "\n".join(
            f"  {r['feature']:<24} {r['gain_pct']:6.2f}%  {r['split']:>7,}"
            for r in self.importance.iter_rows(named=True)
        )
        return (
            f"adversarial validation: {self.n_train:,} train vs {self.n_test:,} test, "
            f"{len(self.features)} features\n"
            f"  OOF AUC   {self.auc:.4f}   (per-fold sd {self.auc_std:.4f})\n"
            f"  per fold  {', '.join(f'{a:.4f}' for a in self.fold_aucs)}\n\n"
            f"{verdict}\n\n"
            f"  {'feature':<24} {'gain':>7}  {'splits':>7}\n{rows}"
        )


def build_matrix(
    train: pl.DataFrame,
    test: pl.DataFrame,
    features: tuple[str, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stack train and test into one design matrix with an is-test label.

    Categoricals are cast on the *combined* frame so both halves share one level set —
    LightGBM aligns pandas category dtypes between fit and predict, and any level it has
    not seen becomes missing. Casting each half separately would risk two different
    integer codings for the same word, which is a bug that produces a plausible number.
    """
    cats = [c for c in features if c in io.CATEGORICAL_COLS]
    combined = pl.concat(
        [train.select(features), test.select(features)],
        how="vertical",
    ).with_columns(pl.col(c).cast(pl.Categorical) for c in cats)

    y = np.concatenate([np.zeros(train.height, dtype=np.int8), np.ones(test.height, dtype=np.int8)])
    return combined.to_pandas(), y


def run(
    train: pl.DataFrame | None = None,
    test: pl.DataFrame | None = None,
    *,
    features: tuple[str, ...] = io.FEATURE_COLS,
    n_splits: int = 5,
    seed: int = SEED,
    params: dict[str, Any] | None = None,
    sample_frac: float | None = None,
) -> AdversarialResult:
    """Cross-validate a train-vs-test classifier and return its OOF AUC and importances.

    The folds here are throwaway, stratified on the is-test label. They have nothing to do
    with the competition folds frozen in `folds.py` — this runs *before* those exist, and
    its answer is what decides whether their design is sound.

    `sample_frac` subsamples both halves for a fast smoke run; leave it None for the real
    answer, since a subsample can only weaken evidence of a shift.
    """
    train = io.load_train() if train is None else train
    test = io.load_test() if test is None else test
    if sample_frac is not None:
        train = train.sample(fraction=sample_frac, seed=seed)
        test = test.sample(fraction=sample_frac, seed=seed)

    import lightgbm as lgb

    X, y = build_matrix(train, test, features)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs: list[float] = []
    gains = np.zeros(len(features), dtype=np.float64)
    splits = np.zeros(len(features), dtype=np.float64)

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, val_idx in splitter.split(X, y):
        model = lgb.LGBMClassifier(**(params or DEFAULT_PARAMS))
        model.fit(X.iloc[fit_idx], y[fit_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[val_idx], oof[val_idx])))
        gains += model.booster_.feature_importance(importance_type="gain")
        splits += model.booster_.feature_importance(importance_type="split")

    total = gains.sum()
    importance = (
        pl.DataFrame(
            {
                "feature": list(features),
                "gain": gains,
                "gain_pct": 100.0 * gains / total if total else np.zeros_like(gains),
                "split": splits.astype(np.int64),
            }
        )
        .sort("gain", descending=True)
    )

    return AdversarialResult(
        auc=float(roc_auc_score(y, oof)),
        fold_aucs=tuple(fold_aucs),
        importance=importance,
        oof=oof,
        n_train=train.height,
        n_test=test.height,
        features=tuple(features),
    )
