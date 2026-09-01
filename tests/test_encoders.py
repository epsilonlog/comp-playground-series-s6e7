from __future__ import annotations

import numpy as np
import polars as pl

from s6e7 import encoders, io


def planted(n_per_value: int = 40, n_values: int = 20) -> tuple[pl.DataFrame, np.ndarray]:
    """`sleep_duration` value parity decides the class; nothing else carries signal."""
    values = np.repeat(np.arange(n_values, dtype=np.float32), n_per_value)
    y = np.where(values % 2 == 0, 2, 0).astype(np.int64)
    frame = pl.DataFrame(
        {c: pl.Series(values, dtype=pl.Float32) for c in encoders.EXACT_VALUE_COLUMNS}
    )
    return frame, y


def test_encoding_recovers_a_planted_per_value_rate() -> None:
    frame, y = planted()
    enc = encoders.ExactValueTargetEncoder().fit(frame, y)
    matrix, names = enc.transform(frame)

    assert matrix.shape == (frame.height, len(encoders.EXACT_VALUE_COLUMNS) * len(io.CLASSES))
    assert names[:3] == [
        "te_sleep_duration__at-risk",
        "te_sleep_duration__fit",
        "te_sleep_duration__unhealthy",
    ]
    even = frame["sleep_duration"].to_numpy() % 2 == 0
    unhealthy_col = matrix[:, io.CLASSES.index("unhealthy")]
    assert unhealthy_col[even].min() > 0.6  # smoothed toward the prior, still high
    assert unhealthy_col[~even].max() < 0.4


def test_unseen_value_and_null_do_not_crash_and_fall_back_sanely() -> None:
    frame, y = planted()
    enc = encoders.ExactValueTargetEncoder().fit(frame, y)
    probe = pl.DataFrame(
        {c: pl.Series([999.0, None], dtype=pl.Float32) for c in encoders.EXACT_VALUE_COLUMNS}
    )
    matrix, _ = enc.transform(probe)
    assert np.isfinite(matrix).all()
    # Unseen value gets the global prior; the fit frame had no nulls, so null does too.
    assert abs(float(matrix[0, io.CLASSES.index("unhealthy")]) - float((y == 2).mean())) < 1e-6


def test_inner_cross_fitting_denies_a_row_its_own_label() -> None:
    """One row per value: a naive encoding would hand the model the label verbatim."""
    frame, y = planted(n_per_value=1, n_values=200)
    enc = encoders.ExactValueTargetEncoder()
    inner, _ = enc.fit_transform_inner(frame, y)
    naive, _ = encoders.ExactValueTargetEncoder().fit(frame, y).transform(frame)

    col = io.CLASSES.index("unhealthy")
    truth = (y == 2).astype(float)
    # Naive: each singleton value's own label leaks through; inner cross-fitting kills it.
    # What is left is the wobble of the per-split prior (SE ~ 1/sqrt(200) = 0.07), not leak.
    assert abs(float(np.corrcoef(naive[:, col], truth)[0, 1])) > 0.5
    assert abs(float(np.corrcoef(inner[:, col], truth)[0, 1])) < 0.25


def test_encoder_is_left_fitted_on_all_rows_after_inner_fit() -> None:
    """val/test rows must be encoded by the map built from ALL of the fold's fit rows."""
    frame, y = planted()
    enc = encoders.ExactValueTargetEncoder()
    enc.fit_transform_inner(frame, y)
    after_inner, _ = enc.transform(frame)
    direct, _ = encoders.ExactValueTargetEncoder().fit(frame, y).transform(frame)
    assert np.allclose(after_inner, direct)


def test_build_rejects_an_unknown_name() -> None:
    try:
        encoders.build("nope")
    except KeyError as exc:
        assert "exact_value_te" in str(exc)
    else:
        raise AssertionError("unknown encoder name must raise")
