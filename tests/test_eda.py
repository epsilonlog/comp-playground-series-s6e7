from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from s6e7.eda import (
    category_levels,
    class_profile,
    level_target_rates,
    missing_cooccurrence,
    missing_vs_target,
    null_count_vs_target,
    numeric_correlation,
    numeric_shift,
    numeric_summary,
    overview,
)


def test_numeric_shift_max_bin_dev_ignores_a_pure_null_gap() -> None:
    """Identical non-null distributions + very different null rates must read ~0 dev."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=20_000)
    with_nulls = [None] * 8_000 + [float(v) for v in values[8_000:]]
    out = numeric_shift(
        pl.DataFrame({"x": values}),
        pl.DataFrame({"x": pl.Series(with_nulls, dtype=pl.Float64)}),
        ["x"],
    )
    assert out["max_bin_dev"][0] < 0.05
    assert abs(out["null_gap"][0] - 40.0) < 1.0


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


# --- level_target_rates -------------------------------------------------------


def test_level_target_rates_respects_declared_order() -> None:
    df = pl.DataFrame(
        {
            "lvl": ["high", "low", "medium", "low"],
            "y": ["a", "b", "a", "b"],
        }
    )
    out = level_target_rates(df, ["lvl"], "y", {"lvl": ("low", "medium", "high")})
    assert out["level"].to_list() == ["low", "medium", "high"]


def test_level_target_rates_falls_back_to_alphabetical() -> None:
    df = pl.DataFrame({"lvl": ["b", "a", "c"], "y": ["x", "x", "y"]})
    assert level_target_rates(df, ["lvl"], "y")["level"].to_list() == ["a", "b", "c"]


def test_level_target_rates_puts_nulls_last_as_their_own_level() -> None:
    df = pl.DataFrame({"lvl": ["a", None, "b"], "y": ["x", "y", "x"]})
    out = level_target_rates(df, ["lvl"], "y")
    assert out["level"].to_list() == ["a", "b", "<null>"]


def test_level_target_rates_computes_class_rate_per_level() -> None:
    df = pl.DataFrame(
        {
            "lvl": ["low", "low", "low", "low", "high", "high"],
            "y": ["fit", "fit", "fit", "ill", "fit", "ill"],
        }
    )
    out = level_target_rates(df, ["lvl"], "y", {"lvl": ("low", "high")})
    assert out["p_fit"].to_list() == [0.75, 0.5]
    assert out["rows"].to_list() == [4, 2]


# --- numeric_summary ----------------------------------------------------------


def test_numeric_summary_finds_grid_and_bounds() -> None:
    df = pl.DataFrame({"x": [1.0, 1.5, 2.0, 2.5, 3.0]})
    row = numeric_summary(df, ["x"]).row(0, named=True)
    assert row["min"] == 1.0
    assert row["max"] == 3.0
    assert row["grid"] == 0.5
    assert row["n_unique"] == 5


def test_grid_is_the_median_gap_not_the_minimum() -> None:
    """One off-grid value must not make a clean 1.0 grid look like a 0.2 grid."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 5.2, 6.0, 7.0, 8.0]})
    row = numeric_summary(df, ["x"]).row(0, named=True)
    assert row["grid"] == 1.0
    assert row["min_gap"] == pytest.approx(0.2)


def test_mode_pct_catches_a_point_mass_away_from_the_bounds() -> None:
    """Zero-inflation style spike sitting in the middle of the range."""
    df = pl.DataFrame({"x": [1.0, 9.0, *([5.0] * 8)]})
    row = numeric_summary(df, ["x"]).row(0, named=True)
    assert row["mode"] == 5.0
    assert row["mode_pct"] == 80.0
    assert row["pct_at_min"] == 10.0
    assert row["pct_at_max"] == 10.0


def test_numeric_summary_detects_clipping_mass() -> None:
    """Four of ten rows pinned at the maximum — a clip, not a taper."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 10.0, 10.0, 10.0]})
    row = numeric_summary(df, ["x"]).row(0, named=True)
    assert row["pct_at_max"] == 40.0
    assert row["pct_at_min"] == 10.0


def test_numeric_summary_ignores_nulls() -> None:
    df = pl.DataFrame({"x": [1.0, None, 3.0]})
    row = numeric_summary(df, ["x"]).row(0, named=True)
    assert row["n_unique"] == 2
    assert row["mean"] == 2.0


def test_numeric_summary_handles_all_null_column() -> None:
    df = pl.DataFrame({"x": pl.Series("x", [None, None], dtype=pl.Float64)})
    assert numeric_summary(df, ["x"]).row(0, named=True)["n_unique"] == 0


# --- missing_vs_target --------------------------------------------------------


def test_missing_vs_target_detects_no_relationship() -> None:
    """Same class rate whether present or missing -> abs_diff 0."""
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, None, None],
            "y": ["a", "b", "a", "b"],
        }
    )
    out = missing_vs_target(df, ["x"], "y").filter(pl.col("target_class") == "a")
    assert out.row(0, named=True)["abs_diff"] == 0.0


def test_missing_vs_target_detects_relationship() -> None:
    df = pl.DataFrame(
        {
            "x": [1.0, 2.0, None, None],
            "y": ["a", "a", "b", "b"],
        }
    )
    row = missing_vs_target(df, ["x"], "y").filter(pl.col("target_class") == "a").row(0, named=True)
    assert row["p_when_present"] == 1.0
    assert row["p_when_missing"] == 0.0
    assert row["abs_diff"] == 1.0


def test_missing_vs_target_skips_columns_with_no_nulls() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"]})
    assert missing_vs_target(df, ["x"], "y").height == 0


# --- missing_cooccurrence -----------------------------------------------------


def test_missing_cooccurrence_flags_a_shared_mask() -> None:
    """a and b always go missing together; c is independent."""
    a = [None, None, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    b = [None, None, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    c = [3.0, 3.0, None, None, 3.0, 3.0, 3.0, 3.0]
    out = missing_cooccurrence(pl.DataFrame({"a": a, "b": b, "c": c}), ["a", "b", "c"])
    top = out.row(0, named=True)
    assert {top["col_a"], top["col_b"]} == {"a", "b"}
    assert top["both_missing"] == 2
    assert top["ratio"] > 1.0


def test_missing_cooccurrence_returns_empty_when_nothing_is_null() -> None:
    df = pl.DataFrame({"a": [1.0], "b": [2.0]})
    assert missing_cooccurrence(df, ["a", "b"]).height == 0


# --- class_profile ------------------------------------------------------------


def test_class_profile_ranks_the_separating_feature_first() -> None:
    df = pl.DataFrame(
        {
            "signal": [0.0, 0.0, 10.0, 10.0],
            "noise": [5.0, 5.0, 5.0, 5.0],
            "y": ["a", "a", "b", "b"],
        }
    )
    out = class_profile(df, ["noise", "signal"], "y")
    assert out["column"].to_list()[0] == "signal"
    assert out.row(0, named=True)["mean_a"] == 0.0
    assert out.row(0, named=True)["mean_b"] == 10.0


def test_class_profile_gives_constant_feature_zero_spread() -> None:
    df = pl.DataFrame({"flat": [5.0, 5.0, 5.0, 5.0], "y": ["a", "a", "b", "b"]})
    row = class_profile(df, ["flat"], "y").row(0, named=True)
    assert row["spread_sd"] in (0.0, None)


def test_identical_class_means_give_total_overlap_and_chance_accuracy() -> None:
    """d = 0: the two curves are the same curve."""
    df = pl.DataFrame({"x": [1.0, 2.0, 1.0, 2.0], "y": ["a", "a", "b", "b"]})
    row = class_profile(df, ["x"], "y").row(0, named=True)
    assert row["spread_sd"] == 0.0
    assert row["overlap_pct"] == 100.0
    assert row["best_split_acc"] == 0.5


def test_overlap_matches_the_closed_form() -> None:
    """OVL = 2*phi(-d/2); best single-split accuracy = phi(d/2)."""
    df = pl.DataFrame({"x": [0.0, 2.0, 2.0, 4.0], "y": ["a", "a", "b", "b"]})
    row = class_profile(df, ["x"], "y").row(0, named=True)
    d = row["spread_sd"]
    expected_acc = 0.5 * (1.0 + math.erf((d / 2.0) / math.sqrt(2.0)))
    assert row["best_split_acc"] == pytest.approx(expected_acc, abs=5e-4)
    assert row["overlap_pct"] == pytest.approx(100.0 * 2.0 * (1.0 - expected_acc), abs=0.1)


# --- numeric_correlation ------------------------------------------------------


def test_correlation_finds_a_perfect_monotone_pair() -> None:
    """Spearman is rank-based, so a curved but monotone relation still reads 1.0."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 4.0, 9.0, 16.0]})
    row = numeric_correlation(df, ["a", "b"]).row(0, named=True)
    assert row["spearman"] == pytest.approx(1.0)


def test_correlation_sorts_by_absolute_value_so_negatives_rank() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 2.0, 2.0],
        }
    )
    out = numeric_correlation(df, ["a", "b", "c"])
    top = out.row(0, named=True)
    assert {top["col_a"], top["col_b"]} == {"a", "b"}
    assert top["spearman"] == pytest.approx(-1.0)


def test_correlation_drops_nulls_pairwise() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, None, 4.0], "b": [1.0, 2.0, 3.0, None]})
    row = numeric_correlation(df, ["a", "b"]).row(0, named=True)
    assert row["n_pairs"] == 2


def test_correlation_returns_all_pairs() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [2.0, 1.0], "c": [1.0, 3.0], "d": [4.0, 1.0]})
    assert numeric_correlation(df, ["a", "b", "c", "d"]).height == 6


def _null_frame() -> pl.DataFrame:
    """20 rows: `a` null on the first 8, `b` on the first 4, so null counts run 2,2,2,2,1,1,1,1,0..."""
    return pl.DataFrame(
        {
            "a": [None] * 8 + list(range(12)),
            "b": [None] * 4 + list(range(16)),
            "y": ["hit", "miss"] * 10,
        }
    )


def test_null_count_vs_target_buckets_by_row_null_count() -> None:
    out = null_count_vs_target(_null_frame(), ["a", "b"], "y", min_rows=1)
    assert out["split"].unique().to_list() == ["all"]
    assert out["n_nulls"].to_list() == [0, 1, 2]
    assert out["n_rows"].to_list() == [12, 4, 4]


def test_null_count_vs_target_reports_rate_and_binomial_se() -> None:
    row = null_count_vs_target(_null_frame(), ["a", "b"], "y", min_rows=1).row(1, named=True)
    assert row["pct_hit"] == 50.0
    # 100 * sqrt(0.5 * 0.5 / 4) = 25.0
    assert row["se_hit"] == 25.0


def test_null_count_vs_target_drops_thin_buckets() -> None:
    """Below min_rows the error bar is wider than any effect worth reading."""
    out = null_count_vs_target(_null_frame(), ["a", "b"], "y", min_rows=5)
    assert out["n_nulls"].to_list() == [0]


def test_null_count_vs_target_splits_on_one_columns_nullity() -> None:
    """The trap this guards: one informative indicator bending the unsplit rates on its own."""
    out = null_count_vs_target(_null_frame(), ["a", "b"], "y", split_on="a", min_rows=1)
    assert out["split"].unique(maintain_order=True).to_list() == ["a present", "a null"]
    present = out.filter(pl.col("split") == "a present")
    assert present["n_nulls"].to_list() == [0]
    assert present["n_rows"].to_list() == [12]
    absent = out.filter(pl.col("split") == "a null")
    assert absent["n_nulls"].to_list() == [1, 2]
    assert absent["n_rows"].to_list() == [4, 4]
