from __future__ import annotations

import numpy as np

from s6e7 import decision, metric


def planted() -> tuple[np.ndarray, np.ndarray]:
    """Class 2 is always second-best under argmax; a multiplier ~1.3 fixes every row."""
    y = np.array([0] * 600 + [1] * 200 + [2] * 100)
    proba = np.zeros((900, 3))
    proba[:600] = [0.8, 0.1, 0.1]
    proba[600:800] = [0.1, 0.8, 0.1]
    proba[800:] = [0.5, 0.1, 0.4]
    return proba, y


def test_apply_reweights_the_argmax() -> None:
    proba = np.array([[0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])
    assert decision.apply(proba, np.array([1.0, 1.0, 1.0])).tolist() == [0, 2]
    assert decision.apply(proba, np.array([1.0, 2.0, 1.0])).tolist() == [1, 1]


def test_search_never_loses_to_argmax() -> None:
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=600)
    y = rng.integers(0, 3, size=600)
    _, score = decision.search(proba, y)
    assert score >= metric.balanced_accuracy(y, proba.argmax(axis=1)) - 1e-12


def test_search_finds_a_planted_multiplier() -> None:
    proba, y = planted()
    assert metric.balanced_accuracy(y, proba.argmax(axis=1)) < 0.7
    multipliers, score = decision.search(proba, y)
    assert score > 0.999
    assert multipliers[0] == 1.0
    assert 0.5 / 0.4 < multipliers[2] < 0.8 / 0.1


def test_cross_fit_is_scored_out_of_fold() -> None:
    proba, y = planted()
    fold = np.arange(len(y)) % 5
    result = decision.cross_fit(proba, y, fold)
    assert result.labels.shape == y.shape
    assert len(result.fold_scores) == 5
    assert len(result.fold_multipliers) == 5
    assert all(len(m) == 3 and m[0] == 1.0 for m in result.fold_multipliers)
    assert float(np.mean(result.fold_scores)) > 0.999
