"""Feature construction. Operator's feature ideas land here; the baseline is the floor.

Everything in this module is a *declared* transform — mappings fixed by the schema in
`io` (ORDINAL_LEVELS, NOMINAL_LEVELS, CLASSES), never fitted on data — so nothing here
can leak across folds (CLAUDE.md rule 3). Any transform that must be *fitted* (target
encoding, scaling, imputation) belongs inside the fold loop in `cv.py`, not here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from s6e7 import io


def _code(col: str, levels: tuple[str, ...]) -> pl.Expr:
    """Integer code per declared level, as Float32 so a null can become NaN downstream.

    Nulls pass through; a level outside the declared vocabulary raises — silent coding
    of a drifted schema is exactly the bug this refuses to have.
    """
    mapping = {level: float(i) for i, level in enumerate(levels)}
    return pl.col(col).replace_strict(mapping, return_dtype=pl.Float32)


def baseline_matrix(df: pl.DataFrame) -> tuple[NDArray[np.float32], list[str]]:
    """The raw features as one float32 matrix; nulls become NaN (GBDTs handle natively).

    Numerics pass through; ordinal columns take their declared 0..2 codes (a contiguity
    constraint, priced in LEARNING.md); nominals take their vocabulary index. Integer
    codes for the nominals are a baseline simplification — native categorical handling
    is a later, separate experiment.
    """
    names = list(io.NUMERIC_COLS + io.ORDINAL_COLS + io.NOMINAL_COLS)
    exprs = [pl.col(c).cast(pl.Float32) for c in io.NUMERIC_COLS]
    exprs += [_code(c, io.ORDINAL_LEVELS[c]) for c in io.ORDINAL_COLS]
    exprs += [_code(c, io.NOMINAL_LEVELS[c]) for c in io.NOMINAL_COLS]
    matrix = df.select(exprs).to_numpy().astype(np.float32, copy=False)
    return matrix, names


def encode_target(y: pl.Series) -> NDArray[np.int8]:
    """Labels as int8 codes in `io.CLASSES` order. Raises on a label outside the schema."""
    mapping = {cls: i for i, cls in enumerate(io.CLASSES)}
    return y.replace_strict(mapping, return_dtype=pl.Int8).to_numpy().astype(np.int8, copy=False)


def decode_target(codes: NDArray[np.integer]) -> pl.Series:
    """Class codes back to label strings, for submissions."""
    return pl.Series(io.TARGET, np.asarray(io.CLASSES, dtype=object)[codes], dtype=pl.String)
