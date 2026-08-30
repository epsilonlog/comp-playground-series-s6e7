from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from s6e7 import io
from s6e7.features import baseline_matrix, decode_target, encode_target


def frame(**overrides: list[object]) -> pl.DataFrame:
    base: dict[str, list[object]] = {
        io.ID: [0, 1, 2],
        io.TARGET: ["at-risk", "fit", "unhealthy"],
        **{c: [1.5, None, 3.0] for c in io.NUMERIC_COLS},
        "stress_level": ["low", "medium", "high"],
        "sleep_quality": ["poor", "average", "good"],
        "physical_activity_level": ["sedentary", "moderate", "active"],
        "smoking_alcohol": ["no", "occasional", "yes"],
        "diet_type": ["balanced", "non-veg", "veg"],
        "gender": ["female", "male", None],
    }
    base.update(overrides)
    return pl.DataFrame(base, schema_overrides=dict.fromkeys(io.NUMERIC_COLS, pl.Float32))


def test_baseline_matrix_shape_names_and_dtype() -> None:
    matrix, names = baseline_matrix(frame())
    assert matrix.shape == (3, 13)
    assert matrix.dtype == np.float32
    assert names == list(io.NUMERIC_COLS + io.ORDINAL_COLS + io.NOMINAL_COLS)


def test_declared_codes_follow_the_schema_order() -> None:
    matrix, names = baseline_matrix(frame())
    for col in io.ORDINAL_COLS + io.NOMINAL_COLS:
        assert matrix[:, names.index(col)].tolist()[:2] == [0.0, 1.0], col


def test_nulls_become_nan_never_a_code() -> None:
    matrix, names = baseline_matrix(frame())
    assert np.isnan(matrix[1, names.index("bmi")])
    assert np.isnan(matrix[2, names.index("gender")])


def test_unknown_level_raises_rather_than_silently_coding() -> None:
    bad = frame(diet_type=["balanced", "KETO", "veg"])
    with pytest.raises(pl.exceptions.InvalidOperationError):
        baseline_matrix(bad)


def test_target_roundtrip_and_unknown_label() -> None:
    y = encode_target(frame()[io.TARGET])
    assert y.tolist() == [0, 1, 2]
    assert y.dtype == np.int8
    assert decode_target(y).to_list() == ["at-risk", "fit", "unhealthy"]
    with pytest.raises(pl.exceptions.InvalidOperationError):
        encode_target(pl.Series([io.CLASSES[0], "zombie"]))
