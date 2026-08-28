"""The frozen cross-validation partition. Workflow step 3, and the last word on it.

Every experiment scores against *these* folds. That is not bookkeeping fussiness — two
models evaluated on the identical validation rows share their luck ("these particular
7,961 `fit` people were easy") and it cancels out of the difference between them. A
paired comparison on frozen folds resolves a smaller gap than either absolute score can.
Regenerating the partition throws that cancellation away and leaves two independently
wobbling numbers.

So `build` refuses to overwrite an existing file unless forced, and `verify` re-derives
the assignment from the raw data and checks the frozen file still matches it.

Design: ``StratifiedKFold(5)``, stratified on the target alone, seed 42.

    `id` is unique across all 690,088 train and 295,753 test rows, so there is no
    repeating entity and `GroupKFold` has nothing to group by. Nothing is time-ordered.
    At 5.8% minority, stratification is required regardless: 5 folds leaves ~7,960 `fit`
    rows in each held-out set.

Chosen *after* adversarial validation returned AUC 0.6518 — a real train/test shift, and
the reason this decision was reopened. Two follow-ups closed it again::

    complete-case rows only (no nulls at all)   AUC 0.5304   the shift is the null pattern
    p(target | n_nulls, bmi_is_null)            flat in k    and it is target-orthogonal

All of missingness's target signal is `bmi_is_null`, whose marginal rate is identical in
both files (2.014%). What shifted is the *co-occurrence* of nulls, which predicts nothing.
The one measurable consequence is that test carries more null-heavy rows than train
(k >= 3: 4.55% vs 2.17%), so test rows are slightly information-poorer and **LB should
land a little below CV** — a level effect that cancels out of every model comparison.

`null_bucket` rides along in the frozen file for exactly that reason. It is a **diagnostic
slice key, not a stratification key**: stratifying on it was priced at 0.00004 against a
0.002 fold-level standard error. Carrying it costs one column and lets every experiment
report per-slice recall, which is what would catch this analysis being wrong — most
plausibly on a comparison of missing-data handling, the one design axis the shift could
mis-rank.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold

from s6e7 import io
from s6e7.config import SEED

FOLDS_PATH: Final[Path] = io.PROCESSED / "folds.parquet"

FOLD: Final[str] = "fold"
NULL_BUCKET: Final[str] = "null_bucket"

#: Rows with this many nulls or more collapse into the top diagnostic bucket. Above 3 the
#: counts are too thin to read a recall from: train holds 1,479 rows at k=4 and 96 beyond.
NULL_BUCKET_CAP: Final[int] = 3


@dataclass(frozen=True, slots=True)
class FoldConfig:
    """The partition's full specification. Changing any field invalidates every prior run."""

    n_splits: int = 5
    seed: int = SEED
    stratify_on: str = io.TARGET


DEFAULT_CONFIG: Final[FoldConfig] = FoldConfig()


def null_bucket(
    df: pl.DataFrame,
    cols: Sequence[str] = io.FEATURE_COLS,
    *,
    cap: int = NULL_BUCKET_CAP,
) -> pl.Series:
    """Per-row null count, clipped at `cap`. The diagnostic slice key, one value per row.

    This is the quantity adversarial validation caught shifting: train draws its nulls
    independently per column, test clumps them onto the same rows. Same per-column rates,
    same mean of 0.6514 nulls per row, 32% more variance.
    """
    expr = pl.sum_horizontal(pl.col(c).is_null() for c in cols).clip(upper_bound=cap)
    return df.select(expr.cast(pl.UInt8).alias(NULL_BUCKET))[NULL_BUCKET]


def assign(
    train: pl.DataFrame | None = None,
    *,
    config: FoldConfig = DEFAULT_CONFIG,
) -> pl.DataFrame:
    """Compute the partition: one row per training row, `id` / `fold` / `null_bucket`.

    Pure and deterministic — same input and config, same output. `build` is what makes it
    permanent; this is what `verify` re-runs to check the permanent copy.

    The result carries `id` rather than relying on row position, so a downstream frame
    that has been filtered or reordered still joins correctly.
    """
    train = io.load_train() if train is None else train
    y = train[config.stratify_on].to_numpy()

    fold = np.empty(train.height, dtype=np.int8)
    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    for k, (_, val_idx) in enumerate(splitter.split(np.zeros(train.height), y)):
        fold[val_idx] = k

    return pl.DataFrame(
        {
            io.ID: train[io.ID],
            FOLD: pl.Series(fold, dtype=pl.Int8),
            NULL_BUCKET: null_bucket(train),
        }
    )


def build(
    train: pl.DataFrame | None = None,
    *,
    config: FoldConfig = DEFAULT_CONFIG,
    path: Path = FOLDS_PATH,
    force: bool = False,
) -> Path:
    """Freeze the partition to parquet. Idempotent, and refuses to overwrite.

    `force=True` exists so the file can be rebuilt from scratch after a data refresh; it
    is not an escape hatch for mid-competition second thoughts. If the fold design is
    genuinely wrong, every logged `cv_mean` becomes incomparable and the ledger has to
    say so out loud.
    """
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    assign(train, config=config).write_parquet(path)
    return path


def load(path: Path = FOLDS_PATH) -> pl.DataFrame:
    """Read the frozen partition. Raises if it was never built."""
    if not path.exists():
        msg = f"{path} not found. Build it once with s6e7.folds.build()."
        raise FileNotFoundError(msg)
    return pl.read_parquet(path)


def verify(
    train: pl.DataFrame | None = None,
    *,
    config: FoldConfig = DEFAULT_CONFIG,
    path: Path = FOLDS_PATH,
) -> None:
    """Re-derive the partition and assert the frozen file still matches it. Raises if not.

    Stronger than recording the config in the file's metadata, which would only confirm
    what the writer *claimed*. This checks the assignment itself, so it catches a changed
    seed, a changed splitter, a reordered or re-downloaded source file, and a partially
    written parquet alike.
    """
    frozen = load(path)
    expected = assign(train, config=config)
    if frozen.height != expected.height:
        msg = f"{path} has {frozen.height:,} rows, source data implies {expected.height:,}"
        raise ValueError(msg)
    if not frozen.equals(expected):
        msg = (
            f"{path} does not match the assignment {config} produces. The folds are frozen "
            f"(CLAUDE.md rule 6) — do not rebuild. Find out what changed underneath them."
        )
        raise ValueError(msg)


def fold_vector(df: pl.DataFrame, *, path: Path = FOLDS_PATH) -> NDArray[np.int8]:
    """Fold index per row of `df`, in `df`'s own row order. Joined by `id`, never position.

    ``validate="1:1"`` makes a duplicated or missing `id` an error rather than a silently
    longer frame, and ``maintain_order="left"`` pins the output to `df`'s order — without
    it the returned vector could be a correct set of labels attached to the wrong rows.
    """
    joined = df.select(io.ID).join(
        load(path).select(io.ID, FOLD),
        on=io.ID,
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    missing = joined[FOLD].null_count()
    if missing:
        msg = f"{missing:,} of {df.height:,} rows have no frozen fold — is this the training set?"
        raise ValueError(msg)
    return joined[FOLD].to_numpy().astype(np.int8)


def iter_folds(fold: NDArray[np.integer]) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    """Yield ``(fit_idx, val_idx)`` per fold, in fold order. Positions into `fold`, not ids.

    A generator rather than a list: the index arrays for 690k rows run to a few MB a pair,
    and nothing ever needs two folds at once.
    """
    for k in np.unique(fold):
        yield np.flatnonzero(fold != k), np.flatnonzero(fold == k)


def composition(
    train: pl.DataFrame | None = None,
    *,
    test: pl.DataFrame | None = None,
    path: Path = FOLDS_PATH,
) -> pl.DataFrame:
    """Per-fold composition beside the pooled train row and, optionally, the test row.

    Decision: did the freeze produce what the design promised, and how far do the folds
    sit from the thing they stand in for? Three readings in one table:

    - **the stratified dimension** — per-class rates, flat across folds to within a row or
      two. Boring by construction; if it is not boring, the splitter is wrong.
    - **the unstratified dimension** — `null_bucket` shares, free to drift. The binomial
      says +-0.04pp on the k>=3 bucket, so anything larger is a bug, not variance.
    - **the reference** — the `test` row, whose null-bucket shares *are* the shift. No fold
      resembles it, deliberately; that gap is what the LB-below-CV prediction is made of.
      Test carries no target, so its class-rate cells stay null.
    """
    train = io.load_train() if train is None else train
    assignment = load(path)
    joined = train.join(assignment, on=io.ID, how="inner", validate="1:1")
    classes = sorted(str(v) for v in train[io.TARGET].drop_nulls().unique().to_list())

    def describe(source: str, frame: pl.DataFrame, buckets: pl.Series) -> dict[str, object]:
        labelled = io.TARGET in frame.columns
        out: dict[str, object] = {"source": source, "n_rows": frame.height}
        for cls in classes:
            count = int((frame[io.TARGET] == cls).sum()) if labelled else 0
            out[f"n_{cls}"] = count if labelled else None
            out[f"pct_{cls}"] = round(100.0 * count / frame.height, 3) if labelled else None
        shares = buckets.to_numpy()
        for b in range(NULL_BUCKET_CAP + 1):
            label = f"pct_k{b}plus" if b == NULL_BUCKET_CAP else f"pct_k{b}"
            out[label] = round(100.0 * float((shares == b).mean()), 3)
        return out

    rows = [
        describe(f"fold_{k}", part, part[NULL_BUCKET])
        for k, part in (
            (k, joined.filter(pl.col(FOLD) == k)) for k in sorted(joined[FOLD].unique())
        )
    ]
    rows.append(describe("train", joined, joined[NULL_BUCKET]))
    if test is not None:
        rows.append(describe("test", test, null_bucket(test)))
    return pl.DataFrame(rows)
