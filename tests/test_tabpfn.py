from __future__ import annotations

import numpy as np
import pytest

from s6e7 import registry
from s6e7.models.tabpfn import TabPFNContext, stratified_context


def _labels(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).choice(3, size=n, p=[0.86, 0.06, 0.08]).astype(np.int64)


def test_small_data_is_used_whole() -> None:
    y = _labels(500)
    np.testing.assert_array_equal(stratified_context(y, 1000), np.arange(500))


def test_subsample_keeps_the_class_proportions_and_is_deterministic() -> None:
    y = _labels(50_000)
    idx = stratified_context(y, 5_000)
    assert idx.size == 5_000
    assert np.all(np.diff(idx) > 0), "indices must be sorted so rows stay positional"
    full = np.bincount(y, minlength=3) / y.size
    sub = np.bincount(y[idx], minlength=3) / idx.size
    np.testing.assert_allclose(sub, full, atol=0.002)
    np.testing.assert_array_equal(idx, stratified_context(y, 5_000))
    assert not np.array_equal(idx, stratified_context(y, 5_000, seed=1))


def test_builder_is_registered_and_defers_the_import() -> None:
    model = registry.build("tabpfn", {"max_fit_rows": 10})
    assert isinstance(model, TabPFNContext)
    assert list(model.classes_) == [0, 1, 2]
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_proba(np.zeros((2, 13), dtype=np.float32))


def test_fit_predict_when_the_library_is_available() -> None:
    pytest.importorskip("tabpfn")  # Kaggle only: weights are gated and GPU-sized
    from s6e7 import features
    from test_cv import make_train

    train = make_train(300)
    X, _ = features.build_matrix("baseline", train)
    y = features.encode_target(train["health_condition"])
    model = TabPFNContext({"max_fit_rows": 200, "n_estimators": 1, "device": "cpu"}).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (300, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)
