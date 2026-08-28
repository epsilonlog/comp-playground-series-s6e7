"""Tabular EDA summaries. Frames in, frames out — never figures.

`plots.py` owns anything that returns a `Figure`. This module owns the checks whose
answer is a number or a small table, which is most of them: a 7-row table beats a
7-panel grid whenever you only need to compare magnitudes.

Everything here is read-only and fits no state, so it is safe to run on full data
outside a fold (CLAUDE.md rule 3 governs *learned* transforms, not description).
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def overview(df: pl.DataFrame) -> pl.DataFrame:
    """One row per column: dtype, null count and rate, and distinct value count.

    Decision: which columns are numeric vs categorical, which need imputation, and
    which are effectively constant.

    ``n_unique`` excludes nulls. Polars' own ``n_unique`` counts null as a distinct
    value, which silently reports a 3-level categorical with missing data as having 4
    levels.
    """
    height = df.height
    return pl.DataFrame(
        [
            {
                "column": name,
                "dtype": str(series.dtype),
                "nulls": series.null_count(),
                "null_pct": round(100.0 * series.null_count() / height, 2) if height else 0.0,
                "n_unique": series.drop_nulls().n_unique(),
            }
            for name, series in df.to_dict().items()
        ]
    )


def category_levels(
    train: pl.DataFrame, cols: Sequence[str], test: pl.DataFrame | None = None
) -> pl.DataFrame:
    """The distinct levels of each categorical column, and any train/test disagreement.

    Decision: encoding strategy, and whether naive encoders are safe. A level that
    appears only in test has no encoding learned for it and will break a fitted encoder
    at predict time; a level only in train is dead weight.

    Levels come back in **alphabetical** order, which is rarely the meaningful one —
    ``high, low, medium`` sorts nothing like ``low < medium < high``. Deciding whether a
    column is ordinal, and in what order, is a modelling judgement this function
    deliberately does not make for you.
    """
    rows = []
    for col in cols:
        train_levels = train[col].drop_nulls().unique().sort().to_list()
        test_only: str | None = None
        train_only: str | None = None
        if test is not None:
            test_levels = set(test[col].drop_nulls().unique().to_list())
            test_only = ", ".join(sorted(test_levels - set(train_levels)))
            train_only = ", ".join(sorted(set(train_levels) - test_levels))
        rows.append(
            {
                "column": col,
                "n_levels": len(train_levels),
                "levels": ", ".join(str(level) for level in train_levels),
                "test_only": test_only,
                "train_only": train_only,
            }
        )
    return pl.DataFrame(rows)
