from __future__ import annotations

import numpy as np
import pytest

from s6e7 import features, folds, registry
from s6e7.cv import ExperimentConfig, run
from s6e7.models.tabm import TabMClassifier
from test_cv import make_train

# Tiny everything: the tests prove the plumbing, not the model.
FAST = {
    "k": 4,
    "d_block": 16,
    "n_bins": 4,
    "d_embedding": 4,
    "batch_size": 64,
    "max_epochs": 2,
    "patience": 1,
    "device": "cpu",
    "amp": False,
    "verbose": False,
}


def _fit(n: int = 300, **extra: object) -> tuple[TabMClassifier, np.ndarray, np.ndarray]:
    train = make_train(n)
    X, _ = features.build_matrix("baseline", train)
    y = features.encode_target(train["health_condition"])
    model = registry.build("tabm", {**FAST, **extra})
    assert isinstance(model, TabMClassifier)
    return model.fit(X, y), X, y


def test_predict_proba_is_a_probability_matrix_in_class_order() -> None:
    model, X, _ = _fit()
    proba = model.predict_proba(X)
    assert proba.shape == (300, 3)
    assert not np.isnan(proba).any()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert list(model.classes_) == [0, 1, 2]


def test_nulls_are_handled_and_flagged() -> None:
    model, X, _ = _fit()
    assert np.isnan(X).any(), "the fixture must contain nulls for this test to mean anything"
    cats = model._categorical(X)
    # 6 categorical codes + 7 null flags; codes inside their cardinalities; flags 0/1.
    assert cats.shape == (300, 13)
    for j, card in enumerate(model._cardinalities()):
        assert cats[:, j].min() >= 0 and cats[:, j].max() < card
    num = model._numeric(X)
    assert np.isfinite(num).all()


def test_extra_numeric_columns_are_accepted() -> None:
    train = make_train(300)
    X, _ = features.build_matrix("baseline", train)
    y = features.encode_target(train["health_condition"])
    extra = np.random.default_rng(0).random((300, 9), dtype=np.float32)
    model = TabMClassifier(FAST).fit(np.hstack([X, extra]), y)
    assert model.predict_proba(np.hstack([X, extra])).shape == (300, 3)
    with pytest.raises(ValueError, match="columns"):
        model.predict_proba(X)


def test_early_stopping_keeps_the_best_epoch() -> None:
    model, _, _ = _fit(max_epochs=3, patience=5)
    assert len(model.history_) == 3
    best = min(model.history_, key=lambda h: h["holdout_nll"])
    assert model.best_epoch_ == best["epoch"]


def test_plain_likelihood_is_a_valid_option_and_others_are_not() -> None:
    _fit(class_weight=None)
    with pytest.raises(ValueError, match="class_weight"):
        _fit(class_weight="sqrt")


def test_runs_through_the_harness_with_the_exact_value_encoder(tmp_path) -> None:  # type: ignore[no-untyped-def]
    train = make_train(400)
    result = run(
        ExperimentConfig(exp_id="exp_9101", model="tabm", params=FAST, encoder="exact_value_te"),
        train=train,
        folds_path=folds.build(train, path=tmp_path / "folds.parquet"),
        oof_dir=tmp_path / "oof",
        ledger=tmp_path / "experiments.csv",
    )
    oof = np.load(result.oof_path)
    assert oof.shape == (400, 3) and not np.isnan(oof).any()
    assert len(result.fold_scores) == 5
