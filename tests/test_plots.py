from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from s6e7 import features, folds, io, plots, registry
from test_cv import FAST_LGBM, make_train


def test_every_plot_returns_a_figure() -> None:
    train = make_train(n=300)
    test = make_train(n=120, seed=2).drop(io.TARGET)
    fold = folds.assign(train)[folds.FOLD].to_numpy()
    rng = np.random.default_rng(0)
    oof = rng.dirichlet(np.ones(3), size=300)

    figs = [
        plots.target_overview(train, io.TARGET),
        plots.missingness(train),
        plots.numeric_grid(train, io.NUMERIC_COLS[:3], io.TARGET),
        plots.categorical_grid(train, io.CATEGORICAL_COLS[:2], io.TARGET),
        plots.correlation(train, list(io.NUMERIC_COLS[:4])),
        plots.train_test_shift(train, test, ["bmi", "gender"]),
        plots.fold_distribution(train, fold, io.TARGET, test=test),
        plots.oof_diagnostics(train[io.TARGET].to_numpy(), oof),
    ]
    assert all(isinstance(f, Figure) for f in figs)


def test_importance_accepts_one_or_many_models() -> None:
    train = make_train(n=200)
    matrix, names = features.baseline_matrix(train)
    y = features.encode_target(train[io.TARGET])
    models = [registry.build("lgbm", FAST_LGBM).fit(matrix, y) for _ in range(2)]

    assert isinstance(plots.importance(models[0], names), Figure)
    assert isinstance(plots.importance(models, names, top=5), Figure)
