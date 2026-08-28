"""EDA plotting — OPERATOR-OWNED FILE.

Claude may review and suggest, never implement. Bodies are written by hand.

Contract (CLAUDE.md):
    * every function takes a Polars frame
    * every function returns a ``matplotlib.figure.Figure``
    * never call ``plt.show()``
    * never save to disk — the caller decides
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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


def _grid(
    n: int, ncols: int = 3, panel: tuple[float, float] = (4.0, 3.0)
) -> tuple[Figure, list[Axes]]:
    """Allocate a figure and a flat list of exactly ``n`` axes; hide the remainder."""
    raise NotImplementedError


# --- the five used on day one -------------------------------------------------


def target_overview(df: pl.DataFrame, target: str) -> Figure:
    """Distribution, class balance, and summary statistics of the target."""
    raise NotImplementedError


def missingness(df: pl.DataFrame) -> Figure:
    """Null count / fraction per column, and co-missingness structure."""
    raise NotImplementedError


def numeric_grid(df: pl.DataFrame, cols: Sequence[str], target: str | None = None) -> Figure:
    """One panel per numeric column; conditioned on ``target`` when given."""
    raise NotImplementedError


def categorical_grid(df: pl.DataFrame, cols: Sequence[str], target: str) -> Figure:
    """One panel per categorical column: level frequency and target rate per level."""
    raise NotImplementedError


def correlation(df: pl.DataFrame, cols: Sequence[str], method: str = "spearman") -> Figure:
    """Correlation heatmap over ``cols``."""
    raise NotImplementedError


# --- the four that actually win competitions ----------------------------------


def train_test_shift(train: pl.DataFrame, test: pl.DataFrame, cols: Sequence[str]) -> Figure:
    """Per-feature train/test distribution overlay."""
    raise NotImplementedError


def fold_distribution(df: pl.DataFrame, folds: np.ndarray, target: str) -> Figure:
    """Fold size and target balance per fold."""
    raise NotImplementedError


def oof_diagnostics(y: np.ndarray, oof: np.ndarray) -> Figure:
    """Residuals, calibration, and error by segment for out-of-fold predictions."""
    raise NotImplementedError


def importance(model: Any, names: Sequence[str], top: int = 30) -> Figure:
    """Top-``top`` feature importances from a fitted model."""
    raise NotImplementedError
