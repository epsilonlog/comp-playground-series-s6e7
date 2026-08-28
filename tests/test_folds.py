from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from s6e7 import io
from s6e7.folds import (
    FOLD,
    NULL_BUCKET,
    FoldConfig,
    assign,
    build,
    composition,
    fold_vector,
    iter_folds,
    null_bucket,
    verify,
)

CLASSES = ("at-risk", "unhealthy", "fit")


def make_train(n: int = 600, *, nulls: dict[str, int] | None = None) -> pl.DataFrame:
    """Synthetic frame with the real column names — 80/13/7 target, optional planted nulls.

    Tests never touch ``data/``: it is gitignored, so a test that needs it cannot run in CI.
    """
    labels = ["at-risk"] * (n * 80 // 100) + ["unhealthy"] * (n * 13 // 100)
    labels += ["fit"] * (n - len(labels))
    frame = pl.DataFrame(
        {
            io.ID: pl.Series(range(n), dtype=pl.UInt32),
            io.TARGET: labels,
            **{c: pl.Series([1.0] * n, dtype=pl.Float32) for c in io.NUMERIC_COLS},
            **{c: ["a"] * n for c in io.CATEGORICAL_COLS},
        }
    )
    for col, count in (nulls or {}).items():
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) < count).then(None).otherwise(pl.col(col)).alias(col)
        )
    return frame


def test_assign_partitions_every_row_exactly_once() -> None:
    out = assign(make_train())
    assert out.height == 600
    assert sorted(out[FOLD].unique().to_list()) == [0, 1, 2, 3, 4]
    assert out[io.ID].to_list() == list(range(600))


def test_fold_sizes_differ_by_at_most_one() -> None:
    counts = assign(make_train(n=601))[FOLD].value_counts()[FOLD.replace(FOLD, "count")]
    assert max(counts) - min(counts) <= 1


def test_stratification_holds_the_minority_class_flat() -> None:
    """The property the design is bought with: `fit` is 5.8% of real data, so it must not drift."""
    out = assign(make_train(n=1000)).with_columns(make_train(n=1000)[io.TARGET])
    per_fold = [
        int((part[io.TARGET] == "fit").sum())
        for k in range(5)
        for part in [out.filter(pl.col(FOLD) == k)]
    ]
    assert max(per_fold) - min(per_fold) <= 1


def test_assignment_is_deterministic_and_seed_dependent() -> None:
    train = make_train()
    assert assign(train).equals(assign(train))
    other = assign(train, config=FoldConfig(seed=7))
    assert not assign(train).equals(other)


def test_null_bucket_counts_and_clips() -> None:
    # Nested prefixes: row 0 carries all four nulls, row 1 three, rows 2-3 two, rows 4-5 one.
    train = make_train(n=10, nulls={"bmi": 6, "heart_rate": 4, "step_count": 2, "gender": 1})
    assert null_bucket(train, cap=99).to_list() == [4, 3, 2, 2, 1, 1, 0, 0, 0, 0]
    assert null_bucket(train).to_list() == [3, 3, 2, 2, 1, 1, 0, 0, 0, 0]
    assert null_bucket(train, cap=2).to_list() == [2, 2, 2, 2, 1, 1, 0, 0, 0, 0]


def test_null_bucket_rides_along_in_the_assignment() -> None:
    out = assign(make_train(n=100, nulls={"bmi": 30}))
    assert out[NULL_BUCKET].sum() == 30
    assert out.schema[NULL_BUCKET] == pl.UInt8


def test_iter_folds_yields_disjoint_covering_splits() -> None:
    fold = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
    splits = list(iter_folds(fold))
    assert len(splits) == 3
    for fit_idx, val_idx in splits:
        assert not set(fit_idx) & set(val_idx)
        assert sorted([*fit_idx, *val_idx]) == list(range(6))
    assert sorted(np.concatenate([val for _, val in splits]).tolist()) == list(range(6))


def test_fold_vector_follows_row_order_not_position(tmp_path: Path) -> None:
    """The bug this guards: correct labels attached to the wrong rows after a reorder."""
    train = make_train()
    path = build(train, path=tmp_path / "folds.parquet")
    shuffled = train.sample(fraction=1.0, shuffle=True, seed=1)

    got = fold_vector(shuffled, path=path)
    expected = (
        assign(train)
        .join(shuffled.select(io.ID).with_row_index("pos"), on=io.ID)
        .sort("pos")[FOLD]
        .to_numpy()
    )
    assert np.array_equal(got, expected)


def test_fold_vector_rejects_unknown_ids(tmp_path: Path) -> None:
    path = build(make_train(), path=tmp_path / "folds.parquet")
    stranger = make_train(n=1).with_columns(pl.lit(999_999, dtype=pl.UInt32).alias(io.ID))
    with pytest.raises(ValueError, match="no frozen fold"):
        fold_vector(stranger, path=path)


def test_build_is_idempotent_and_refuses_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "folds.parquet"
    build(make_train(), path=path)
    original = pl.read_parquet(path)

    build(make_train(), config=FoldConfig(seed=7), path=path)
    assert pl.read_parquet(path).equals(original), "a re-run silently re-froze the partition"

    build(make_train(), config=FoldConfig(seed=7), path=path, force=True)
    assert not pl.read_parquet(path).equals(original)


def test_verify_passes_on_a_clean_freeze_and_raises_on_a_tampered_one(tmp_path: Path) -> None:
    train = make_train()
    path = build(train, path=tmp_path / "folds.parquet")
    verify(train, path=path)

    tampered = pl.read_parquet(path)
    tampered[0, FOLD] = (tampered[0, FOLD] + 1) % 5
    tampered.write_parquet(path)
    with pytest.raises(ValueError, match="rule 6"):
        verify(train, path=path)


def test_verify_catches_a_changed_seed(tmp_path: Path) -> None:
    train = make_train()
    path = build(train, path=tmp_path / "folds.parquet")
    with pytest.raises(ValueError, match="rule 6"):
        verify(train, config=FoldConfig(seed=7), path=path)


def test_composition_reports_folds_train_and_test(tmp_path: Path) -> None:
    train = make_train(n=1000, nulls={"bmi": 100})
    path = build(train, path=tmp_path / "folds.parquet")
    out = composition(train, test=make_train(n=200).drop(io.TARGET), path=path)

    assert out["source"].to_list() == [
        "fold_0",
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "train",
        "test",
    ]
    assert out.filter(pl.col("source") == "train")["n_rows"].item() == 1000
    assert out.filter(pl.col("source") == "train")["pct_k1"].item() == 10.0
    # Test carries no target, so its class cells are null rather than zero.
    assert out.filter(pl.col("source") == "test")["n_fit"].item() is None
    assert out.filter(pl.col("source") == "test")["pct_k0"].item() == 100.0
