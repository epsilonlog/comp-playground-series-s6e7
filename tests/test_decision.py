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


def test_prior_multipliers_are_the_inverse_prior() -> None:
    _, y = planted()  # 600 / 200 / 100
    assert decision.prior_multipliers(y).tolist() == [1.0, 3.0, 6.0]


def test_landscape_matches_the_slow_exact_path_everywhere() -> None:
    """The log-threshold shortcut must be exact, not approximate — every grid cell."""
    proba, y = planted()
    m1, m2, ba = decision.landscape(proba, y, size=6)
    for i, a in enumerate(m1):
        for j, b in enumerate(m2):
            exact = metric.balanced_accuracy(y, decision.apply(proba, np.array([1.0, a, b])))
            assert ba[i, j] == exact


def test_landscape_peak_agrees_with_search() -> None:
    proba, y = planted()
    _, best = decision.search(proba, y)
    _, _, ba = decision.landscape(proba, y, size=25)
    assert ba.max() == best


def test_rule_effect_accounts_for_the_score_change() -> None:
    """Recall deltas weighted by 1/(K*n_k) must reproduce the balanced-accuracy gain."""
    proba, y = planted()
    m, best = decision.search(proba, y)
    effect = decision.rule_effect(proba, y, m)
    gain = float(effect["delta"].sum()) / 3.0
    argmax_score = metric.balanced_accuracy(y, proba.argmax(axis=1))
    assert abs(gain - (best - argmax_score)) < 1e-12
