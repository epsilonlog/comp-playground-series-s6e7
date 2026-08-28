from __future__ import annotations

import polars as pl

from s6e7.eda import category_levels, overview


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


def test_category_levels_lists_sorted_levels_ignoring_nulls() -> None:
    train = pl.DataFrame({"c": ["medium", "low", None, "high", "low"]})
    row = category_levels(train, ["c"]).row(0, named=True)
    assert row["n_levels"] == 3
    assert row["levels"] == "high, low, medium"


def test_category_levels_without_test_leaves_comparison_null() -> None:
    train = pl.DataFrame({"c": ["a", "b"]})
    row = category_levels(train, ["c"]).row(0, named=True)
    assert row["test_only"] is None
    assert row["train_only"] is None


def test_category_levels_flags_test_only_and_train_only() -> None:
    train = pl.DataFrame({"c": ["a", "b", "gone"]})
    test = pl.DataFrame({"c": ["a", "b", "new"]})
    row = category_levels(train, ["c"], test).row(0, named=True)
    assert row["test_only"] == "new"
    assert row["train_only"] == "gone"


def test_category_levels_matching_sets_report_empty_strings() -> None:
    train = pl.DataFrame({"c": ["a", "b"]})
    test = pl.DataFrame({"c": ["b", "a", None]})
    row = category_levels(train, ["c"], test).row(0, named=True)
    assert row["test_only"] == ""
    assert row["train_only"] == ""


def test_category_levels_handles_multiple_columns() -> None:
    train = pl.DataFrame({"a": ["x", "y"], "b": ["p", "p"]})
    out = category_levels(train, ["a", "b"])
    assert out["column"].to_list() == ["a", "b"]
    assert out["n_levels"].to_list() == [2, 1]
