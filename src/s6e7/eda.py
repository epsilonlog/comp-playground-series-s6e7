"""Tabular EDA summaries. Frames in, frames out — never figures.

`plots.py` owns anything that returns a `Figure`. This module owns the checks whose
answer is a number or a small table, which is most of them: a 7-row table beats a
7-panel grid whenever you only need to compare magnitudes.

Everything here is read-only and fits no state, so it is safe to run on full data
outside a fold (CLAUDE.md rule 3 governs *learned* transforms, not description).
"""

from __future__ import annotations

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
