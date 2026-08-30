"""The experiment harness. Frozen folds in; OOF probabilities, a score, a ledger row out.

One call site for the loop every experiment repeats: build a fresh model per fold, fit on
the other four, predict *probabilities* on the held-out fold, score the argmax with the
real metric. Probabilities are what get saved — the decision-rule search and blending
need them, and labels cannot be recovered into probabilities (LEARNING.md).

Three kinds of experiment, all landing in the same append-only ledger:

- ``run``       — train a model on a named feature set (the expensive kind)
- ``run_blend`` — average already-saved OOF probabilities (arithmetic, no training)
- ``run_rule``  — search the decision rule on a parent's OOF (the last step)

``run`` refuses an exp_id it has already seen unless told to *reuse* it: a re-run under
the same id would silently overwrite the OOF file that past comparisons point at. With
``if_logged="skip"`` a notebook cell containing a training run is idempotent — first
execution trains, every later execution reloads the logged result.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from s6e7 import decision, features, folds, io, metric, registry

OOF_DIR: Final[Path] = io.ROOT / "oof"
LEDGER: Final[Path] = io.ROOT / "experiments.csv"

#: What the `folds` ledger column records. Changes only if the frozen file does — never.
FOLD_SPEC: Final[str] = "stratified5_seed42"

_HEADER: Final[str] = (
    "exp_id,date,config,model,folds,cv_mean,cv_std,lb_public,oof_path,runtime_s,parent,changed"
)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One experiment. `changed` states the single thing that differs from `parent`."""

    exp_id: str
    model: str
    params: dict[str, Any] = field(default_factory=dict)
    features: str = "baseline"
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
    #: Score of each fold model on its own 552k training rows. The fit-vs-val gap is the
    #: under/overfitting dial: a small gap with a low score says underfit (add capacity),
    #: a large gap says the capacity went into memorising (regularise). Empty when the
    #: result was reloaded from the ledger rather than trained.
    fit_scores: tuple[float, ...] = ()


def run(
    config: ExperimentConfig,
    *,
    train: pl.DataFrame | None = None,
    test: pl.DataFrame | None = None,
    folds_path: Path = folds.FOLDS_PATH,
    oof_dir: Path = OOF_DIR,
    ledger: Path = LEDGER,
    log: bool = True,
    if_logged: Literal["raise", "skip"] = "raise",
) -> CVResult:
    """Run one experiment on the frozen folds and (by default) log it to the ledger.

    Pass `test` to also produce test predictions: the five fold-models' probabilities
    averaged, saved beside the OOF as ``<exp_id>_test.npy``. Pass ``log=False`` for smoke
    runs that must not enter the ledger. ``if_logged="skip"`` makes the call idempotent:
    an exp_id already in the ledger is reloaded instead of retrained.
    """
    if log and _already_logged(ledger, config.exp_id):
        if if_logged == "skip":
            return result_from_oof(
                config.exp_id, train=train, oof_dir=oof_dir, ledger=ledger, folds_path=folds_path
            )
        msg = f"{config.exp_id} is already in {ledger.name}; the ledger is append-only"
        raise ValueError(msg)

    train = io.load_train() if train is None else train
    started = time.perf_counter()

    matrix, _ = features.build_matrix(config.features, train)
    y = features.encode_target(train[io.TARGET])
    fold = folds.fold_vector(train, path=folds_path)
    n_classes = len(io.CLASSES)

    test_matrix: NDArray[np.float32] | None = None
    test_sum: NDArray[np.float64] | None = None
    if test is not None:
        test_matrix, _ = features.build_matrix(config.features, test)
        test_sum = np.zeros((test.height, n_classes), dtype=np.float64)

    oof = np.full((train.height, n_classes), np.nan, dtype=np.float64)
    scores: list[float] = []
    fit_scores: list[float] = []
    for fit_idx, val_idx in folds.iter_folds(fold):
        model = registry.build(config.model, config.params)
        model.fit(matrix[fit_idx], y[fit_idx])
        _assert_class_order(model, n_classes)
        proba = np.asarray(model.predict_proba(matrix[val_idx]))
        oof[val_idx] = proba
        scores.append(metric.balanced_accuracy(y[val_idx], proba.argmax(axis=1)))
        fit_proba = np.asarray(model.predict_proba(matrix[fit_idx]))
        fit_scores.append(metric.balanced_accuracy(y[fit_idx], fit_proba.argmax(axis=1)))
        if test_matrix is not None and test_sum is not None:
            test_sum += np.asarray(model.predict_proba(test_matrix))

    if np.isnan(oof).any():
        msg = "OOF has unpredicted rows — the fold vector does not cover the frame"
        raise RuntimeError(msg)

    runtime_s = time.perf_counter() - started

    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / f"{config.exp_id}.npy"
    np.save(oof_path, oof.astype(np.float32))

    test_pred_path: Path | None = None
    if test_sum is not None:
        test_pred_path = oof_dir / f"{config.exp_id}_test.npy"
        np.save(test_pred_path, (test_sum / len(scores)).astype(np.float32))

    result = CVResult(
        exp_id=config.exp_id,
        fold_scores=tuple(scores),
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores, ddof=1)),
        oof_path=oof_path,
        test_pred_path=test_pred_path,
        runtime_s=runtime_s,
        fit_scores=tuple(fit_scores),
    )
    if log:
        _append_row(ledger, config, result)
    return result


def run_blend(
    exp_id: str,
    parents: list[str],
    *,
    weights: list[float] | None = None,
    train: pl.DataFrame | None = None,
    folds_path: Path = folds.FOLDS_PATH,
    oof_dir: Path = OOF_DIR,
    ledger: Path = LEDGER,
    log: bool = True,
) -> CVResult:
    """Average the parents' saved OOF probabilities and score the result. No training.

    This is why every run saves probabilities: a blend is arithmetic over ``oof/*.npy``.
    Test predictions blend the same way when every parent saved them. Idempotent — an
    already-logged exp_id recomputes the (cheap) scores without appending a second row.
    """
    train = io.load_train() if train is None else train
    started = time.perf_counter()
    w = np.array([1.0] * len(parents) if weights is None else weights, dtype=np.float64)
    w = w / w.sum()

    oof = sum(w_i * np.load(oof_dir / f"{p}.npy") for w_i, p in zip(w, parents, strict=True))
    oof = np.asarray(oof, dtype=np.float64)
    y = features.encode_target(train[io.TARGET])
    fold = folds.fold_vector(train, path=folds_path)
    scores = [
        metric.balanced_accuracy(y[val_idx], oof[val_idx].argmax(axis=1))
        for _, val_idx in folds.iter_folds(fold)
    ]

    oof_path = oof_dir / f"{exp_id}.npy"
    np.save(oof_path, oof.astype(np.float32))
    test_pred_path: Path | None = None
    parent_tests = [oof_dir / f"{p}_test.npy" for p in parents]
    if all(p.exists() for p in parent_tests):
        blended = sum(w_i * np.load(p) for w_i, p in zip(w, parent_tests, strict=True))
        test_pred_path = oof_dir / f"{exp_id}_test.npy"
        np.save(test_pred_path, np.asarray(blended, dtype=np.float32))

    result = CVResult(
        exp_id=exp_id,
        fold_scores=tuple(scores),
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores, ddof=1)),
        oof_path=oof_path,
        test_pred_path=test_pred_path,
        runtime_s=time.perf_counter() - started,
    )
    if log and not _already_logged(ledger, config := _blend_config(exp_id, parents, w)):
        _append_row(ledger, config, result)
    return result


def run_rule(
    exp_id: str,
    parent: str,
    *,
    train: pl.DataFrame | None = None,
    folds_path: Path = folds.FOLDS_PATH,
    oof_dir: Path = OOF_DIR,
    ledger: Path = LEDGER,
    log: bool = True,
) -> tuple[CVResult, NDArray[np.float64]]:
    """The decision rule as an experiment: cross-fitted on the parent's OOF.

    Scored the only honest way — the multipliers each fold is judged with were searched
    without that fold (decision.cross_fit). Also returns the multipliers searched on the
    *full* OOF, which are what the submission applies to the parent's test probabilities.
    The parent's probabilities are unchanged, so the ledger row points at the parent's
    OOF file. Idempotent like `run_blend`.
    """
    train = io.load_train() if train is None else train
    started = time.perf_counter()
    proba = np.load(oof_dir / f"{parent}.npy")
    y = features.encode_target(train[io.TARGET])
    fold = folds.fold_vector(train, path=folds_path)

    fitted = decision.cross_fit(proba, y, fold)
    final_multipliers, _ = decision.search(proba, y)

    result = CVResult(
        exp_id=exp_id,
        fold_scores=fitted.fold_scores,
        cv_mean=float(np.mean(fitted.fold_scores)),
        cv_std=float(np.std(fitted.fold_scores, ddof=1)),
        oof_path=oof_dir / f"{parent}.npy",
        test_pred_path=(path if (path := oof_dir / f"{parent}_test.npy").exists() else None),
        runtime_s=time.perf_counter() - started,
    )
    config = ExperimentConfig(
        exp_id=exp_id,
        model=f"rule({parent})",
        parent=parent,
        changed="decision rule: per-class multipliers, cross-fitted on OOF",
    )
    if log and not _already_logged(ledger, exp_id):
        _append_row(ledger, config, result)
    return result, final_multipliers


def result_from_oof(
    exp_id: str,
    *,
    train: pl.DataFrame | None = None,
    oof_dir: Path = OOF_DIR,
    ledger: Path = LEDGER,
    folds_path: Path = folds.FOLDS_PATH,
) -> CVResult:
    """Reconstruct a logged experiment's result from its saved OOF. Argmax scoring only.

    Rule experiments (`run_rule`) score with searched multipliers, not argmax — reload
    those through `run_rule` itself, which is idempotent.
    """
    row = _ledger_row(ledger, exp_id)
    train = io.load_train() if train is None else train
    oof = np.load(oof_dir / f"{exp_id}.npy")
    y = features.encode_target(train[io.TARGET])
    fold = folds.fold_vector(train, path=folds_path)
    scores = [
        metric.balanced_accuracy(y[val_idx], oof[val_idx].argmax(axis=1))
        for _, val_idx in folds.iter_folds(fold)
    ]
    test_path = oof_dir / f"{exp_id}_test.npy"
    return CVResult(
        exp_id=exp_id,
        fold_scores=tuple(scores),
        cv_mean=float(np.mean(scores)),
        cv_std=float(np.std(scores, ddof=1)),
        oof_path=oof_dir / f"{exp_id}.npy",
        test_pred_path=test_path if test_path.exists() else None,
        runtime_s=float(row["runtime_s"]),
    )


def submission_frame(test: pl.DataFrame, test_proba: NDArray[np.floating]) -> pl.DataFrame:
    """id + predicted label, in the sample-submission format. Plain argmax — apply
    `decision.apply` to the probabilities first when a rule was chosen."""
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


def _blend_config(exp_id: str, parents: list[str], w: NDArray[np.float64]) -> ExperimentConfig:
    label = "+".join(parents)
    weight_note = "" if np.allclose(w, w[0]) else f", weights {np.round(w, 3).tolist()}"
    return ExperimentConfig(
        exp_id=exp_id,
        model=f"blend({label})",
        parent=parents[0],
        changed=f"blend of {label}{weight_note}",
    )


def _assert_class_order(model: object, n_classes: int) -> None:
    """predict_proba columns must be class codes 0..K-1 or every saved OOF lies."""
    classes = getattr(model, "classes_", None)
    if classes is not None and list(classes) != list(range(n_classes)):
        msg = f"model orders classes as {list(classes)}, expected 0..{n_classes - 1}"
        raise RuntimeError(msg)


def _already_logged(ledger: Path, exp_id: str | ExperimentConfig) -> bool:
    key = exp_id if isinstance(exp_id, str) else exp_id.exp_id
    if not ledger.exists():
        return False
    with ledger.open(newline="", encoding="utf-8") as f:
        return any(row and row[0] == key for row in csv.reader(f))


def _ledger_row(ledger: Path, exp_id: str) -> dict[str, str]:
    with ledger.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["exp_id"] == exp_id:
                return row
    msg = f"{exp_id} not found in {ledger}"
    raise KeyError(msg)


def _append_row(ledger: Path, config: ExperimentConfig, result: CVResult) -> None:
    if not ledger.exists():
        ledger.write_text(_HEADER + "\n", encoding="utf-8")
    with ledger.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                config.exp_id,
                date.today().isoformat(),
                f"configs/{config.exp_id}.yaml",
                config.model,
                FOLD_SPEC,
                f"{result.cv_mean:.5f}",
                f"{result.cv_std:.5f}",
                "",
                result.oof_path.relative_to(io.ROOT).as_posix()
                if result.oof_path.is_relative_to(io.ROOT)
                else str(result.oof_path),
                f"{result.runtime_s:.0f}",
                config.parent,
                config.changed,
            ]
        )
