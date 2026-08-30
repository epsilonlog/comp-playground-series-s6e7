"""Feature construction. Every experiment names one FEATURE SET from `MATRICES`.

Everything in this module is a *declared* transform — mappings fixed by the schema in
`io` (ORDINAL_LEVELS, NOMINAL_LEVELS, CLASSES), never fitted on data — so nothing here
can leak across folds (CLAUDE.md rule 3). Any transform that must be *fitted* (target
encoding, scaling, imputation) belongs inside the fold loop in `cv.py`, not here.

A feature set is a function `pl.DataFrame -> (float32 matrix, column names)`. Adding an
idea = one function + one `MATRICES` entry + one experiment id. Sets compose on top of
`baseline` so a comparison against the parent isolates exactly the added columns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from s6e7 import io

Matrix = tuple[NDArray[np.float32], list[str]]


def _code(col: str, levels: tuple[str, ...], *, null_code: float | None = None) -> pl.Expr:
    """Integer code per declared level, as Float32.

    Nulls pass through as null (NaN downstream) unless `null_code` is given; a level
    outside the declared vocabulary raises — silent coding of a drifted schema is
    exactly the bug this refuses to have.
    """
    mapping = {level: float(i) for i, level in enumerate(levels)}
    expr = pl.col(col).replace_strict(mapping, return_dtype=pl.Float32)
    return expr if null_code is None else expr.fill_null(null_code)


def _base_exprs(*, cat_null_code: float | None = None) -> list[pl.Expr]:
    exprs = [pl.col(c).cast(pl.Float32) for c in io.NUMERIC_COLS]
    exprs += [_code(c, io.ORDINAL_LEVELS[c], null_code=cat_null_code) for c in io.ORDINAL_COLS]
    exprs += [_code(c, io.NOMINAL_LEVELS[c], null_code=cat_null_code) for c in io.NOMINAL_COLS]
    return exprs


BASE_NAMES: Final[list[str]] = list(io.NUMERIC_COLS + io.ORDINAL_COLS + io.NOMINAL_COLS)

#: Column indices of the 6 categorical codes in every set that starts from the base
#: layout — what LightGBM's `categorical_feature` parameter needs.
CATEGORICAL_IDX: Final[list[int]] = [BASE_NAMES.index(c) for c in io.ORDINAL_COLS + io.NOMINAL_COLS]


def _to_matrix(df: pl.DataFrame, exprs: list[pl.Expr], names: list[str]) -> Matrix:
    matrix = df.select(exprs).to_numpy().astype(np.float32, copy=False)
    return matrix, names


def baseline_matrix(df: pl.DataFrame) -> Matrix:
    """The floor: raw features, declared codes, nulls as NaN (GBDTs route natively)."""
    return _to_matrix(df, _base_exprs(), BASE_NAMES)


def native_cats_matrix(df: pl.DataFrame) -> Matrix:
    """Baseline layout with categorical nulls coded -1 instead of NaN.

    LightGBM's native categorical handling casts the column to int and treats negative
    values as missing; NaN in a declared-categorical numpy column is undefined. Only
    meaningful together with ``params: {categorical_feature: CATEGORICAL_IDX}`` — the
    two lines are one experimental variable ("categorical handling").
    """
    return _to_matrix(df, _base_exprs(cat_null_code=-1.0), BASE_NAMES)


def indicators_matrix(df: pl.DataFrame) -> Matrix:
    """Baseline + an explicit 0/1 is-null flag per feature column.

    Tests whether explicit indicators beat the NaN routing the trees already do. The
    known signal is `bmi_is_null` (unhealthy 2.79% vs 8.47%), but NaN routing may
    already express it — a clean candidate for an honest negative result.
    """
    flags = [pl.col(c).is_null().cast(pl.Float32).alias(f"{c}_isnull") for c in io.FEATURE_COLS]
    names = BASE_NAMES + [f"{c}_isnull" for c in io.FEATURE_COLS]
    return _to_matrix(df, _base_exprs() + flags, names)


def _ratio(a: str, b: str, name: str) -> pl.Expr:
    """a / b with a zero denominator giving null (NaN downstream), never inf."""
    return (pl.col(a) / pl.when(pl.col(b) != 0).then(pl.col(b))).cast(pl.Float32).alias(name)


RATIO_NAMES: Final[list[str]] = ["cal_per_step", "cal_per_exmin", "steps_per_exmin"]


def ratios_matrix(df: pl.DataFrame) -> Matrix:
    """Baseline + three activity-intensity ratios.

    Motivated by the exp_0001 error profile (notebook 06): `fit` recall dies in the
    low-step and low-exercise bins, and the activity trio is internally correlated
    (0.37-0.44) — a *ratio* is exactly the combination a tree cannot build from
    axis-aligned splits (LEARNING.md, interactions that are never free).
    """
    ratios = [
        _ratio("calorie_expenditure", "step_count", "cal_per_step"),
        _ratio("calorie_expenditure", "exercise_duration", "cal_per_exmin"),
        _ratio("step_count", "exercise_duration", "steps_per_exmin"),
    ]
    return _to_matrix(df, _base_exprs() + ratios, BASE_NAMES + RATIO_NAMES)


MATRICES: dict[str, Callable[[pl.DataFrame], Matrix]] = {
    "baseline": baseline_matrix,
    "native_cats": native_cats_matrix,
    "indicators": indicators_matrix,
    "ratios": ratios_matrix,
}


def build_matrix(name: str, df: pl.DataFrame) -> Matrix:
    """The named feature set. Raises on an unknown name rather than guessing."""
    if name not in MATRICES:
        msg = f"unknown feature set {name!r}; available: {sorted(MATRICES)}"
        raise KeyError(msg)
    return MATRICES[name](df)


def encode_target(y: pl.Series) -> NDArray[np.int8]:
    """Labels as int8 codes in `io.CLASSES` order. Raises on a label outside the schema."""
    mapping = {cls: i for i, cls in enumerate(io.CLASSES)}
    return y.replace_strict(mapping, return_dtype=pl.Int8).to_numpy().astype(np.int8, copy=False)


def decode_target(codes: NDArray[np.integer]) -> pl.Series:
    """Class codes back to label strings, for submissions."""
    return pl.Series(io.TARGET, np.asarray(io.CLASSES, dtype=object)[codes], dtype=pl.String)
