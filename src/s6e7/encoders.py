"""Transforms that must be **fitted**, and therefore live inside the fold loop.

`features.py` holds *declared* transforms only — mappings fixed by the schema, safe to
build once on the whole frame. Anything that reads the target is the opposite kind of
object: fitted on the fold's training rows, applied to the held-out rows, discarded
(CLAUDE.md rule 3). This module holds those, behind the same registry-dict idiom the
models use.

The only entry so far is exact-value target encoding, and it exists because notebook 06
measured the signal it reaches: a repeated numeric value in this synthetic dataset
behaves like a high-cardinality *category*, and per-value class rates replicate across
independent halves of the data (`sleep_duration` r = 0.94) with a residual larger than
the class's own base rate. A 255-bin histogram destroys that before the first split, so
no amount of capacity recovers it — encoding is the only route.

Two leaks are possible here and both are closed by construction:

1. **Across the fold.** The encoder is fitted on the fold's training rows only, so the
   held-out rows' labels never enter their own encoding.
2. **Within the training rows.** A row's own label contributes to its value's mean, so a
   naive encoding hands the model a peek at each training label — worst on rare values,
   where the mean *is* the label. `fit_transform_inner` closes it: the training matrix is
   built from an inner K-fold, so no row is ever encoded by a statistic it helped
   compute. The gap between that and the naive version is what "target-encoding
   overfitting" means; with smoothing and ~1,000 rows per value here it is small, but
   correctness should not depend on the cardinality happening to be low.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final, Protocol

import numpy as np
import polars as pl
from numpy.typing import NDArray

from s6e7 import io
from s6e7.config import SEED

Matrix = tuple[NDArray[np.float32], list[str]]

#: The three numeric columns whose per-value signal replicated in notebook 06 section 4
#: (`eda.exact_value_signal`). The other four read r ~ 0 and are deliberately excluded:
#: encoding them would add nine noise columns and cost the comparison its resolution.
EXACT_VALUE_COLUMNS: Final[tuple[str, ...]] = ("sleep_duration", "water_intake", "heart_rate")

#: m-estimate weight: a value's encoding is pulled toward the global class rate as if it
#: carried `SMOOTHING` extra rows at that rate. The columns above average ~1,000 rows per
#: value, so this is a no-op where the signal is and a guard rail in the tail.
SMOOTHING: Final[float] = 20.0


class FoldEncoder(Protocol):
    """What `cv.run` drives, once per fold. Fitted objects are never reused across folds."""

    def fit(self, df: pl.DataFrame, y: NDArray[np.integer]) -> FoldEncoder: ...

    def transform(self, df: pl.DataFrame) -> Matrix: ...

    def fit_transform_inner(self, df: pl.DataFrame, y: NDArray[np.integer]) -> Matrix: ...


@dataclass(slots=True)
class ExactValueTargetEncoder:
    """Per-value class rates as columns: one per (column, class), m-estimate smoothed.

    An unseen value falls back to the global class rate. A **null** value gets its own
    encoding rather than the fallback — missingness here is informative (notebook 01's
    `bmi_is_null`, notebook 06's null-bin recall crater), so the null group is just
    another category.
    """

    columns: tuple[str, ...] = EXACT_VALUE_COLUMNS
    smoothing: float = SMOOTHING
    inner_splits: int = 5
    seed: int = SEED
    _prior: NDArray[np.float64] = field(default_factory=lambda: np.empty(0), init=False)
    #: column -> (sorted known values, table of shape (n_values, K), null-row encoding)
    _maps: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = field(
        default_factory=dict, init=False
    )

    @property
    def names(self) -> list[str]:
        return [f"te_{col}__{cls}" for col in self.columns for cls in io.CLASSES]

    def fit(self, df: pl.DataFrame, y: NDArray[np.integer]) -> ExactValueTargetEncoder:
        y_arr = np.asarray(y, dtype=np.int64)
        n_classes = len(io.CLASSES)
        counts = np.bincount(y_arr, minlength=n_classes)
        self._prior = counts / counts.sum()
        self._maps = {}
        for col in self.columns:
            values = df[col].cast(pl.Float64).to_numpy()
            is_null = np.isnan(values)
            known, inverse = np.unique(values[~is_null], return_inverse=True)
            hits = np.zeros((known.size, n_classes), dtype=np.float64)
            for k in range(n_classes):
                hits[:, k] = np.bincount(
                    inverse, weights=(y_arr[~is_null] == k), minlength=known.size
                )
            n_per_value = hits.sum(axis=1, keepdims=True)
            table = (hits + self.smoothing * self._prior) / (n_per_value + self.smoothing)

            null_hits = np.array(
                [(y_arr[is_null] == k).sum() for k in range(n_classes)], dtype=np.float64
            )
            null_te = (null_hits + self.smoothing * self._prior) / (
                null_hits.sum() + self.smoothing
            )
            self._maps[col] = (known, table, null_te)
        return self

    def transform(self, df: pl.DataFrame) -> Matrix:
        if not self._maps:
            msg = "encoder is not fitted; call fit (or fit_transform_inner) first"
            raise RuntimeError(msg)
        blocks = [self._encode_column(df, col) for col in self.columns]
        return np.hstack(blocks).astype(np.float32, copy=False), self.names

    def fit_transform_inner(self, df: pl.DataFrame, y: NDArray[np.integer]) -> Matrix:
        """Encode these rows with an inner K-fold, so no row is encoded by its own label."""
        y_arr = np.asarray(y, dtype=np.int64)
        out = np.empty((df.height, len(self.columns) * len(io.CLASSES)), dtype=np.float32)
        inner = np.random.default_rng(self.seed).permutation(df.height) % self.inner_splits
        for k in range(self.inner_splits):
            held_out = inner == k
            fold_encoder = ExactValueTargetEncoder(
                columns=self.columns,
                smoothing=self.smoothing,
                inner_splits=self.inner_splits,
                seed=self.seed,
            )
            fold_encoder.fit(df.filter(~held_out), y_arr[~held_out])
            out[held_out], _ = fold_encoder.transform(df.filter(held_out))
        # Leave self fitted on ALL of these rows — that is the map val/test rows use.
        self.fit(df, y_arr)
        return out, self.names

    def _encode_column(self, df: pl.DataFrame, col: str) -> NDArray[np.float64]:
        known, table, null_te = self._maps[col]
        values = df[col].cast(pl.Float64).to_numpy()
        is_null = np.isnan(values)
        clipped = np.clip(np.searchsorted(known, values), 0, max(known.size - 1, 0))
        seen = (
            np.zeros(values.size, dtype=bool)
            if known.size == 0
            else (~is_null) & (known[clipped] == values)
        )
        out = np.broadcast_to(self._prior, (values.size, self._prior.size)).copy()
        out[seen] = table[clipped[seen]]
        out[is_null] = null_te
        return out


def _all_numeric_te() -> FoldEncoder:
    """Every numeric column encoded, including the four the replication test rejected.

    Exists to *test the test*: if `eda.exact_value_signal` is a working screen, adding
    twelve columns it called noise should buy nothing over the three it kept.
    """
    return ExactValueTargetEncoder(columns=io.NUMERIC_COLS)


BUILDERS: dict[str, Callable[[], FoldEncoder]] = {
    "exact_value_te": ExactValueTargetEncoder,
    "exact_value_te_all": _all_numeric_te,
}


def build(name: str) -> FoldEncoder:
    """A fresh, unfitted encoder by name — one per fold, so no state crosses folds."""
    if name not in BUILDERS:
        msg = f"unknown encoder {name!r}; registered: {sorted(BUILDERS)}"
        raise KeyError(msg)
    return BUILDERS[name]()
