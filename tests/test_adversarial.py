from __future__ import annotations

import polars as pl

from s6e7.adversarial import complete_cases


def test_complete_cases_keeps_only_fully_observed_rows() -> None:
    df = pl.DataFrame({"a": [1, None, 3, 4], "b": [1, 2, None, 4], "c": [1, 2, 3, 4]})
    assert complete_cases(df, ["a", "b"])["c"].to_list() == [1, 4]


def test_complete_cases_ignores_nulls_outside_the_named_columns() -> None:
    """The restriction is over `features` only — a null elsewhere must not drop the row."""
    df = pl.DataFrame({"a": [1, 2], "spare": [None, 2]})
    assert complete_cases(df, ["a"]).height == 2


def test_complete_cases_can_return_nothing() -> None:
    df = pl.DataFrame({"a": [None, None]})
    out = complete_cases(df, ["a"])
    assert out.height == 0
    assert out.columns == ["a"]
