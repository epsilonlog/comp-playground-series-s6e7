"""The experiment harness. Frozen folds in; OOF probabilities, a score, a ledger row out.

One call site for the loop every experiment repeats: build a fresh model per fold, fit on
the other four, predict *probabilities* on the held-out fold, score the argmax with the
real metric. Probabilities are what get saved — the decision-rule search and blending
need them, and labels cannot be recovered into probabilities (LEARNING.md).

The ledger (`experiments.csv`) is append-only and `run` refuses an exp_id it has already
seen: a re-run under the same id would silently overwrite the OOF file that past
comparisons point at.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from s6e7 import features, folds, io, metric, registry

OOF_DIR: Final[Path] = io.ROOT / "oof"
LEDGER: Final[Path] = io.ROOT / "experiments.csv"

#: What the `folds` ledger column records. Changes only if the frozen file does — never.
FOLD_SPEC: Final[str] = "stratified5_seed42"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One experiment. `changed` states the single thing that differs from `parent`."""

    exp_id: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    parent: str = ""
    changed: str = ""


@dataclass(frozen=True, slots=True)
class CVResult:
    exp_id: str
    fold_scores: tuple[float, ...]
    cv_mean: float
    cv_std: float
    oof_path: Path
    test_pred_path: Path | None
    runtime_s: float


def run(
    config: ExperimentConfig,
    *,
    train: pl.DataFrame | None = None,
    test: pl.DataFrame | None = None,
    folds_path: Path = folds.FOLDS_PATH,
    oof_dir: Path = OOF_DIR,
    ledger: Path = LEDGER,
    log: bool = True,
) -> CVResult:
    """Run one experiment on the frozen folds and (by default) log it to the ledger.

    Pass `test` to also produce test predictions: the five fold-models' probabilities
    averaged, saved beside the OOF as ``<exp_id>_test.npy``. Pass ``log=False`` for smoke
    runs that must not enter the ledger.
    """
    if log and _already_logged(ledger, config.exp_id):
        msg = f"{config.exp_id} is already in {ledger.name}; the ledger is append-only"
        raise ValueError(msg)

    train = io.load_train() if train is None else train
    started = time.perf_counter()

    matrix, _ = features.baseline_matrix(train)
    y = features.encode_target(train[io.TARGET])
    fold = folds.fold_vector(train, path=folds_path)
    n_classes = len(io.CLASSES)

    test_matrix: NDArray[np.float32] | None = None
    test_sum: NDArray[np.float64] | None = None
    if test is not None:
        test_matrix, _ = features.baseline_matrix(test)
        test_sum = np.zeros((test.height, n_classes), dtype=np.float64)

    oof = np.full((train.height, n_classes), np.nan, dtype=np.float64)
    scores: list[float] = []
    for fit_idx, val_idx in folds.iter_folds(fold):
        model = registry.build(config.model, config.params)
        model.fit(matrix[fit_idx], y[fit_idx])
        _assert_class_order(model, n_classes)
        proba = np.asarray(model.predict_proba(matrix[val_idx]))
        oof[val_idx] = proba
        scores.append(metric.balanced_accuracy(y[val_idx], proba.argmax(axis=1)))
        if test_matrix is not None and test_sum is not None:
            test_sum += np.asarray(model.predict_proba(test_matrix))

    if np.isnan(oof).any():
        msg = "OOF has unpredicted rows — the fold vector does not cover the frame"
        raise RuntimeError(msg)

    runtime_s = time.perf_counter() - started
    cv_mean = float(np.mean(scores))
    cv_std = float(np.std(scores, ddof=1))

    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{config.exp_id}.npy"
    np.save(oof_path, oof.astype(np.float32))

    test_pred_path: Path | None = None
    if test_sum is not None:
        test_pred_path = oof_dir / f"{config.exp_id}_test.npy"
        np.save(test_pred_path, (test_sum / len(scores)).astype(np.float32))

    if log:
        _append_row(ledger, config, cv_mean, cv_std, oof_path, runtime_s)

    return CVResult(
        exp_id=config.exp_id,
        fold_scores=tuple(scores),
        cv_mean=cv_mean,
        cv_std=cv_std,
        oof_path=oof_path,
        test_pred_path=test_pred_path,
        runtime_s=runtime_s,
    )


def submission_frame(test: pl.DataFrame, test_proba: NDArray[np.floating]) -> pl.DataFrame:
    """id + predicted label, in the sample-submission format. Plain argmax — the decision
    rule is a separate, later experiment."""
    return pl.DataFrame(
        {
            io.ID: test[io.ID],
            io.TARGET: features.decode_target(test_proba.argmax(axis=1)),
        }
    )


def slice_report(
    exp_id: str,
    *,
    train: pl.DataFrame | None = None,
    oof_dir: Path = OOF_DIR,
) -> pl.DataFrame:
    """Balanced accuracy per null-count slice for one logged experiment's OOF.

    The watchdog the fold decision left behind (folds.py): the one way the shift could
    bite is a missing-data-handling change that behaves differently on null-heavy rows,
    and this table is where that would show first. `k>=3` is the shifted bucket —
    2.2% of train, 4.6% of test.
    """
    train = io.load_train() if train is None else train
    proba = np.load(oof_dir / f"{exp_id}.npy")
    y = features.encode_target(train[io.TARGET])
    pred = proba.argmax(axis=1)
    buckets = folds.null_bucket(train).to_numpy()

    def describe(label: str, sel: NDArray[np.bool_]) -> dict[str, object]:
        row: dict[str, object] = {
            "slice": label,
            "n_rows": int(sel.sum()),
            "balanced_acc": round(metric.balanced_accuracy(y[sel], pred[sel]), 5),
        }
        matrix, _ = metric.confusion(y[sel], pred[sel], labels=range(len(io.CLASSES)))
        support = matrix.sum(axis=1)
        for i, cls in enumerate(io.CLASSES):
            row[f"recall_{cls}"] = (
                round(float(matrix[i, i] / support[i]), 4) if support[i] else None
            )
        return row

    rows = [describe("all", np.ones(len(y), dtype=bool))]
    for b in range(folds.NULL_BUCKET_CAP + 1):
        label = f"k>={b}" if b == folds.NULL_BUCKET_CAP else f"k={b}"
        rows.append(describe(label, buckets == b))
    return pl.DataFrame(rows)


def _assert_class_order(model: object, n_classes: int) -> None:
    """predict_proba columns must be class codes 0..K-1 or every saved OOF lies."""
    classes = getattr(model, "classes_", None)
    if classes is not None and list(classes) != list(range(n_classes)):
        msg = f"model orders classes as {list(classes)}, expected 0..{n_classes - 1}"
        raise RuntimeError(msg)


def _already_logged(ledger: Path, exp_id: str) -> bool:
    if not ledger.exists():
        return False
    with ledger.open(newline="", encoding="utf-8") as f:
        return any(row and row[0] == exp_id for row in csv.reader(f))


def _append_row(
    ledger: Path,
    config: ExperimentConfig,
    cv_mean: float,
    cv_std: float,
    oof_path: Path,
    runtime_s: float,
) -> None:
    header = (
        "exp_id,date,config,model,folds,cv_mean,cv_std,lb_public,oof_path,runtime_s,parent,changed"
    )
    if not ledger.exists():
        ledger.write_text(header + "\n", encoding="utf-8")
    with ledger.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                config.exp_id,
                date.today().isoformat(),
                f"configs/{config.exp_id}.yaml",
                config.model,
                FOLD_SPEC,
                f"{cv_mean:.5f}",
                f"{cv_std:.5f}",
                "",
                oof_path.relative_to(io.ROOT).as_posix()
                if oof_path.is_relative_to(io.ROOT)
                else str(oof_path),
                f"{runtime_s:.0f}",
                config.parent,
                config.changed,
            ]
        )
