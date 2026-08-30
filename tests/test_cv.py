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


def test_registry_rejects_unknown_model() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        registry.build("gpt7")
