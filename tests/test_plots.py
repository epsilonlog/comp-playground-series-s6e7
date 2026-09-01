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
        plots.rule_landscape(oof, features.encode_target(train[io.TARGET]), size=7),
    ]
    assert all(isinstance(f, Figure) for f in figs)


def test_train_test_shift_shows_test_only_levels() -> None:
    import polars as pl

    train = pl.DataFrame({"c": ["A"] * 90 + ["B"] * 10})
    test = pl.DataFrame({"c": ["A"] * 80 + ["B"] * 10 + ["NEW"] * 10})
    fig = plots.train_test_shift(train, test, ["c"])
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert "NEW" in labels


def test_experiment_compare_reads_a_ledger_frame() -> None:
    import polars as pl

    ledger = pl.DataFrame(
        {
            "exp_id": ["exp_0001", "exp_0002", "exp_0003"],
            "cv_mean": [0.8729, 0.8741, 0.8738],
            "cv_std": [0.0021, 0.0019, 0.0022],
            "lb_public": [0.8721, None, None],
        }
    )
    assert isinstance(plots.experiment_compare(ledger), Figure)


def test_recall_by_bin_first_row_is_global() -> None:
    from s6e7 import eda

    train = make_train(n=600)
    rng = np.random.default_rng(1)
    proba = rng.dirichlet(np.ones(3), size=600)
    table = eda.recall_by_bin(train, proba, "bmi", io.TARGET, n_bins=4, min_rows=10)
    assert table["bin"][0] == "all"
    assert table["n_rows"][0] == 600
    assert table["n_rows"].to_list()[1:] and sum(table["n_rows"].to_list()[1:]) == 600

    cats = eda.recall_by_bin(train, proba, "gender", io.TARGET, min_rows=10)
    assert set(cats["bin"].to_list()) <= {"all", *io.NOMINAL_LEVELS["gender"]}


def test_resolution_demo_is_data_free() -> None:
    assert isinstance(plots.resolution_demo(), Figure)
    assert isinstance(plots.resolution_demo(0.002, effects=(0.001, 0.01)), Figure)


def test_importance_accepts_one_or_many_models() -> None:
    train = make_train(n=200)
    matrix, names = features.baseline_matrix(train)
    y = features.encode_target(train[io.TARGET])
    models = [registry.build("lgbm", FAST_LGBM).fit(matrix, y) for _ in range(2)]

    assert isinstance(plots.importance(models[0], names), Figure)
    assert isinstance(plots.importance(models, names, top=5), Figure)
