from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from s6e7 import folds, io, registry
from s6e7.cv import ExperimentConfig, run, submission_frame

FAST_LGBM = {"n_estimators": 5, "n_jobs": 1}


def make_train(n: int = 400, seed: int = 0) -> pl.DataFrame:
    """Synthetic frame with the real schema and enough class/level variety to fit on."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(io.CLASSES, size=n, p=[0.7, 0.15, 0.15])
    cols: dict[str, object] = {
        io.ID: pl.Series(range(n), dtype=pl.UInt32),
        io.TARGET: pl.Series(labels),
    }
    for c in io.NUMERIC_COLS:
        vals = rng.normal(size=n).astype(np.float32)
        vals[rng.random(n) < 0.05] = np.nan
        cols[c] = pl.Series(vals, dtype=pl.Float32)
    for c in io.ORDINAL_COLS:
        cols[c] = pl.Series(rng.choice(io.ORDINAL_LEVELS[c], size=n))
    for c in io.NOMINAL_COLS:
        cols[c] = pl.Series(rng.choice(io.NOMINAL_LEVELS[c], size=n))
    return pl.DataFrame(cols)


@pytest.fixture
def harness(tmp_path: Path) -> dict[str, object]:
    train = make_train()
    return {
        "train": train,
        "folds_path": folds.build(train, path=tmp_path / "folds.parquet"),
        "oof_dir": tmp_path / "oof",
        "ledger": tmp_path / "experiments.csv",
    }


def _run(harness: dict[str, object], exp_id: str = "exp_9001", **kwargs: object) -> object:
    return run(
        ExperimentConfig(exp_id=exp_id, model="lgbm", params=FAST_LGBM),
        train=harness["train"],  # type: ignore[arg-type]
        folds_path=harness["folds_path"],  # type: ignore[arg-type]
        oof_dir=harness["oof_dir"],  # type: ignore[arg-type]
        ledger=harness["ledger"],  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_run_produces_complete_probability_oof(harness: dict[str, object]) -> None:
    result = _run(harness)
    oof = np.load(result.oof_path)
    assert oof.shape == (400, 3)
    assert not np.isnan(oof).any()
    np.testing.assert_allclose(oof.sum(axis=1), 1.0, atol=1e-5)
    assert len(result.fold_scores) == 5
    assert result.cv_std == pytest.approx(float(np.std(result.fold_scores, ddof=1)))


def test_run_appends_one_ledger_row_and_refuses_a_duplicate(harness: dict[str, object]) -> None:
    _run(harness)
    ledger: Path = harness["ledger"]  # type: ignore[assignment]
    rows = ledger.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 2 and rows[1].startswith("exp_9001,")
    with pytest.raises(ValueError, match="append-only"):
        _run(harness)


def test_log_false_leaves_no_ledger(harness: dict[str, object]) -> None:
    _run(harness, log=False)
    assert not Path(str(harness["ledger"])).exists()


def test_test_predictions_average_the_folds(harness: dict[str, object]) -> None:
    test = make_train(n=50, seed=1).drop(io.TARGET)
    result = _run(harness, test=test)
    preds = np.load(result.test_pred_path)
    assert preds.shape == (50, 3)
    np.testing.assert_allclose(preds.sum(axis=1), 1.0, atol=1e-5)

    sub = submission_frame(test, preds)
    assert sub.columns == [io.ID, io.TARGET]
    assert set(sub[io.TARGET].unique().to_list()) <= set(io.CLASSES)


def test_if_logged_skip_reloads_instead_of_raising(harness: dict[str, object]) -> None:
    first = _run(harness)
    again = _run(harness, if_logged="skip")
    assert again.fold_scores == pytest.approx(first.fold_scores)
    assert again.fit_scores == ()  # reconstructed results carry no fit scores
    assert len(first.fit_scores) == 5
    assert all(0.0 <= s <= 1.0 for s in first.fit_scores)


def test_blend_of_identical_parents_scores_like_the_parent(harness: dict[str, object]) -> None:
    from s6e7.cv import run_blend

    parent = _run(harness)
    blend = run_blend(
        "exp_9010",
        ["exp_9001", "exp_9001"],
        train=harness["train"],  # type: ignore[arg-type]
        folds_path=harness["folds_path"],  # type: ignore[arg-type]
        oof_dir=harness["oof_dir"],  # type: ignore[arg-type]
        ledger=harness["ledger"],  # type: ignore[arg-type]
    )
    assert blend.cv_mean == pytest.approx(parent.cv_mean)
    ledger: Path = harness["ledger"]  # type: ignore[assignment]
    assert "blend(exp_9001+exp_9001)" in ledger.read_text(encoding="utf-8")


def test_blend_refuses_a_conflicting_reuse_of_an_exp_id(harness: dict[str, object]) -> None:
    from s6e7.cv import run_blend

    _run(harness)
    _run(harness, exp_id="exp_9002")
    kwargs = {
        "train": harness["train"],
        "folds_path": harness["folds_path"],
        "oof_dir": harness["oof_dir"],
        "ledger": harness["ledger"],
    }
    run_blend("exp_9010", ["exp_9001", "exp_9002"], **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pick a new exp_id"):
        run_blend("exp_9010", ["exp_9001"], **kwargs)  # type: ignore[arg-type]


def test_oof_functions_refuse_a_reordered_frame(harness: dict[str, object]) -> None:
    from s6e7.cv import slice_report

    result = _run(harness)
    shuffled = harness["train"].sample(fraction=1.0, shuffle=True, seed=3)  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="canonical"):
        slice_report("exp_9001", train=shuffled, oof_dir=result.oof_path.parent)


def test_run_rule_cross_fits_and_is_idempotent(harness: dict[str, object]) -> None:
    from s6e7.cv import run_rule

    _run(harness)
    kwargs = {
        "train": harness["train"],
        "folds_path": harness["folds_path"],
        "oof_dir": harness["oof_dir"],
        "ledger": harness["ledger"],
    }
    result, multipliers = run_rule("exp_9020", "exp_9001", **kwargs)  # type: ignore[arg-type]
    assert len(result.fold_scores) == 5
    assert multipliers.shape == (3,) and multipliers[0] == 1.0

    ledger: Path = harness["ledger"]  # type: ignore[assignment]
    n_rows = len(ledger.read_text(encoding="utf-8").strip().splitlines())
    run_rule("exp_9020", "exp_9001", **kwargs)  # type: ignore[arg-type]
    assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == n_rows


def test_paired_diff_of_an_experiment_with_itself_is_zero(harness: dict[str, object]) -> None:
    from s6e7.cv import paired_diff

    result = _run(harness)
    table = paired_diff(
        "exp_9001",
        "exp_9001",
        train=harness["train"],  # type: ignore[arg-type]
        oof_dir=result.oof_path.parent,
        folds_path=harness["folds_path"],  # type: ignore[arg-type]
    )
    assert table.height == 6  # 5 folds + mean row
    assert table["diff"].to_list() == [0.0] * 6
    assert table["fold"].to_list()[-1] == "mean"


def test_slice_report_covers_all_rows_once(harness: dict[str, object]) -> None:
    from s6e7.cv import slice_report

    result = _run(harness)
    report = slice_report("exp_9001", train=harness["train"], oof_dir=result.oof_path.parent)  # type: ignore[arg-type]
    assert report["slice"].to_list()[0] == "all"
    assert report["n_rows"][0] == 400
    assert sum(report["n_rows"].to_list()[1:]) == 400
    assert 0.0 <= report["balanced_acc"][0] <= 1.0


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        registry.build("gpt7")
