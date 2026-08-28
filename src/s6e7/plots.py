"""EDA plotting — OPERATOR-OWNED FILE.

Claude may review and critique, never write it. Bodies are written by hand.

Contract (PLOTS_SPEC.md). Every function:
    * takes a Polars DataFrame (convert internally if a library needs pandas)
    * returns a ``matplotlib.figure.Figure``
    * never calls ``plt.show()``, ``plt.savefig()``, or global ``plt.tight_layout()``
    * never mutates the input frame
    * has type hints and a docstring naming the decision it informs

The point isn't pretty charts. Each function answers a specific question that
changes a decision. If a plot doesn't change a decision, don't build it.

Build order:
    1. target_overview, missingness          — before touching S6E7's features
    2. numeric_grid, categorical_grid, correlation  — during first EDA
    3. fold_distribution                     — the moment folds are generated
    4. train_test_shift                      — alongside adversarial validation
    5. oof_diagnostics                       — after the first baseline
    6. importance                            — after the first tuned model
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl
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


def _grid(n: int, ncols: int = 3) -> tuple[Figure, np.ndarray]:
    """Create a figure with n subplots in a grid, hiding unused axes."""
    raise NotImplementedError


# --- Tier 1: build first, used on day one -------------------------------------


def target_overview(df: pl.DataFrame, target: str) -> Figure:
    """Decision: which metric behaviour to expect, and whether stratification is needed.

    Distribution of the target; class counts and percentages for classification
    (minority below ~10% forces StratifiedKFold); skew and kurtosis in the title
    for regression; mark impossible or sentinel values (-999, 0-as-missing).
    """
    raise NotImplementedError


def missingness(df: pl.DataFrame) -> Figure:
    """Decision: whether missingness is itself a feature, and what to impute.

    Null fraction per column, sorted descending, zero-null columns omitted; plus a
    co-occurrence panel — blocks of correlated missingness imply a structural cause
    and are often predictive on their own. The correlation panel is the part that earns medals.
    """
    raise NotImplementedError


def numeric_grid(df: pl.DataFrame, cols: Sequence[str], target: str | None = None) -> Figure:
    """Decision: which features need transformation, and which are already informative.

    One subplot per column: histogram/KDE, split by target class or hexbin against a
    continuous target when given. Flag high skew, near-zero variance, or a suspicious
    count of exact-duplicate values (a hint of synthetic data or capping).
    """
    raise NotImplementedError


def categorical_grid(df: pl.DataFrame, cols: Sequence[str], target: str) -> Figure:
    """Decision: encoding strategy per column.

    One subplot per column: top ~15 value counts with the rest collapsed to "other";
    target rate per category against a global-mean line; cardinality in the title
    (<~10 one-hot, >~50 target encoding or CatBoost native, between → test both).
    Flag categories present in test but absent from train — they break naive encoders.
    """
    raise NotImplementedError


def correlation(df: pl.DataFrame, cols: Sequence[str], method: str = "spearman") -> Figure:
    """Decision: which redundant features to drop, and where multicollinearity bites.

    Lower-triangle-masked heatmap, Spearman by default (rank-based, tolerates
    non-linearity and outliers). Optionally a sorted bar of each feature's
    correlation with the target.
    """
    raise NotImplementedError


# --- Tier 2: the ones that actually win competitions --------------------------


def train_test_shift(train: pl.DataFrame, test: pl.DataFrame, cols: Sequence[str]) -> Figure:
    """Decision: whether your CV can be trusted at all.

    The visual companion to adversarial validation — that gives one AUC, this shows
    which columns caused it. Overlaid normalised train/test distributions per column,
    a distance in the title (KS for numeric, PSI or total-variation for categorical),
    subplots sorted worst-first.
    """
    raise NotImplementedError


def fold_distribution(df: pl.DataFrame, folds: np.ndarray, target: str) -> Figure:
    """Decision: whether your fold assignment is actually valid.

    Row count per fold; target rate per fold against the global line; for grouped
    folds confirm zero group overlap and show group count per fold; for time-based
    show each fold's date range. Run immediately after generating folds — a silently
    broken split is invisible in the CV number, which will just look suspiciously good.
    """
    raise NotImplementedError


def oof_diagnostics(y_true: np.ndarray, oof: np.ndarray) -> Figure:
    """Decision: what to fix next.

    A single CV number says how much you're wrong; this says where. Predicted vs
    actual (or a calibration curve), residual distribution and residuals vs
    prediction for heteroscedasticity, error by prediction quantile and by key
    categorical levels, and the worst-N rows by absolute error as a table.
    """
    raise NotImplementedError


def importance(model: Any, feature_names: Sequence[str], top: int = 30) -> Figure:
    """Decision: what to prune, and whether a leak exists.

    Sorted top-N importance with mean and std across folds, not a single fold — a
    feature important in one fold only is noise. Prefer permutation importance or
    SHAP over split-count gain, which is biased toward high-cardinality features.
    One feature dominating overwhelmingly is a leak signal.
    """
    raise NotImplementedError
