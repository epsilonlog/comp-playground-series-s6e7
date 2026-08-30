from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from s6e7 import io
from s6e7.features import (
    CATEGORICAL_IDX,
    baseline_matrix,
    build_matrix,
    decode_target,
    encode_target,
)


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


def test_native_cats_codes_null_as_negative() -> None:
    matrix, names = build_matrix("native_cats", frame())
    assert matrix[2, names.index("gender")] == -1.0
    assert np.isnan(matrix[1, names.index("bmi")])  # numerics keep NaN
    assert [names[i] for i in CATEGORICAL_IDX] == list(io.ORDINAL_COLS + io.NOMINAL_COLS)


def test_indicators_append_null_flags() -> None:
    matrix, names = build_matrix("indicators", frame())
    assert matrix.shape == (3, 13 + 13)
    assert matrix[1, names.index("bmi_isnull")] == 1.0
    assert matrix[0, names.index("bmi_isnull")] == 0.0
    assert matrix[2, names.index("gender_isnull")] == 1.0


def test_ratios_guard_zero_denominators() -> None:
    base = frame(
        calorie_expenditure=[2000.0, 1500.0, 1000.0],
        step_count=[8000.0, 0.0, 4000.0],
        exercise_duration=[40.0, 20.0, 0.0],
    )
    matrix, names = build_matrix("ratios", base)
    assert matrix.shape == (3, 16)
    assert matrix[0, names.index("cal_per_step")] == pytest.approx(0.25)
    assert np.isnan(matrix[1, names.index("cal_per_step")])  # 0 steps -> NaN, not inf
    assert np.isnan(matrix[2, names.index("steps_per_exmin")])
    assert not np.isinf(matrix).any()


def test_unknown_feature_set_raises() -> None:
    with pytest.raises(KeyError, match="unknown feature set"):
        build_matrix("kitchen_sink_9000", frame())


def test_target_roundtrip_and_unknown_label() -> None:
    y = encode_target(frame()[io.TARGET])
    assert y.tolist() == [0, 1, 2]
    assert y.dtype == np.int8
    assert decode_target(y).to_list() == ["at-risk", "fit", "unhealthy"]
    with pytest.raises(pl.exceptions.InvalidOperationError):
        encode_target(pl.Series([io.CLASSES[0], "zombie"]))
