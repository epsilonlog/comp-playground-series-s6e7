from __future__ import annotations

import polars as pl

from s6e7.eda import overview


def test_overview_shape_and_columns() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = overview(df)
    assert out.height == 2
    assert out.columns == ["column", "dtype", "nulls", "null_pct", "n_unique"]
    assert out["column"].to_list() == ["a", "b"]


def test_overview_counts_nulls_and_rate() -> None:
    df = pl.DataFrame({"a": [1, None, None, 4]})
    row = overview(df).row(0, named=True)
    assert row["nulls"] == 2
    assert row["null_pct"] == 50.0


def test_n_unique_excludes_nulls() -> None:
    """The trap: Polars' own n_unique would report 4 here, not 3."""
    df = pl.DataFrame({"level": ["low", "medium", "high", None, "low"]})
    assert overview(df).row(0, named=True)["n_unique"] == 3


def test_empty_frame_does_not_divide_by_zero() -> None:
    df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Int64)})
    row = overview(df).row(0, named=True)
    assert row["nulls"] == 0
    assert row["null_pct"] == 0.0
    assert row["n_unique"] == 0
