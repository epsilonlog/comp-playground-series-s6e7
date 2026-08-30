"""Raw data loading and the frozen dtype schema.

CSV is read exactly once and cached as parquet in ``data/processed``; everything
downstream reads parquet. Loaders build the cache on first call.

Numerics are Float32 — half the memory of Float64 and far more precision than any
GBDT split needs. Categoricals stay as ``String`` deliberately: encoding is a learned
transform and must fit inside the fold (CLAUDE.md rule 3), and keeping the raw levels
is what lets `train_test_shift` detect categories present in test but not train.

Note: this module shadows the stdlib ``io`` on its own name only. Absolute imports mean
``import io`` elsewhere still resolves to the standard library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import polars as pl

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW: Final[Path] = ROOT / "data" / "raw"
PROCESSED: Final[Path] = ROOT / "data" / "processed"

ID: Final[str] = "id"
TARGET: Final[str] = "health_condition"

NUMERIC_COLS: Final[tuple[str, ...]] = (
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
)

CATEGORICAL_COLS: Final[tuple[str, ...]] = (
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
)

FEATURE_COLS: Final[tuple[str, ...]] = NUMERIC_COLS + CATEGORICAL_COLS

# Declared semantic order for the columns judged ordinal. Alphabetical order is wrong
# for every one of them ("high, low, medium"), so the order has to be stated by hand.
#
# This is a MODELLING JUDGEMENT, not a fact about the data — edit it freely. Whether the
# target actually respects these orderings is what `eda.level_target_rates` measures.
ORDINAL_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "stress_level": ("low", "medium", "high"),
    "sleep_quality": ("poor", "average", "good"),
    "physical_activity_level": ("sedentary", "moderate", "active"),
    "smoking_alcohol": ("no", "occasional", "yes"),
}

ORDINAL_COLS: Final[tuple[str, ...]] = tuple(ORDINAL_LEVELS)
NOMINAL_COLS: Final[tuple[str, ...]] = ("diet_type", "gender")

# Declared vocabulary for the nominal columns — verified identical in train and test
# (2026-08-30). A fixed vocabulary makes integer coding a schema fact rather than a
# learned transform, so it cannot leak across folds.
NOMINAL_LEVELS: Final[dict[str, tuple[str, ...]]] = {
    "diet_type": ("balanced", "non-veg", "veg"),
    "gender": ("female", "male", "other"),
}

#: Target classes in sorted order. Every OOF matrix and `predict_proba` output orders
#: its columns this way; class index i everywhere means CLASSES[i].
CLASSES: Final[tuple[str, ...]] = ("at-risk", "fit", "unhealthy")

SCHEMA: Final[dict[str, pl.DataType]] = {
    ID: pl.UInt32(),
    TARGET: pl.String(),
    **{col: pl.Float32() for col in NUMERIC_COLS},
    **{col: pl.String() for col in CATEGORICAL_COLS},
}


def _read_csv(name: str) -> pl.DataFrame:
    """Read one raw CSV under the frozen schema, ignoring columns it doesn't contain."""
    path = RAW / name
    if not path.exists():
        msg = (
            f"{path} not found. Run:\n"
            f"  uv run kaggle competitions download -c playground-series-s6e7 -p data/raw"
        )
        raise FileNotFoundError(msg)
    return pl.read_csv(path, schema_overrides=SCHEMA)


def build_parquet(*, force: bool = False) -> None:
    """Convert the raw CSVs to parquet in ``data/processed``. Idempotent."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for stem in ("train", "test"):
        target = PROCESSED / f"{stem}.parquet"
        if force or not target.exists():
            _read_csv(f"{stem}.csv").write_parquet(target)


def _load(stem: str) -> pl.DataFrame:
    path = PROCESSED / f"{stem}.parquet"
    if not path.exists():
        build_parquet()
    return pl.read_parquet(path)


def load_train() -> pl.DataFrame:
    """Training rows: id, 13 features, and the target."""
    return _load("train")


def load_test() -> pl.DataFrame:
    """Test rows: id and 13 features."""
    return _load("test")


def load_sample_submission() -> pl.DataFrame:
    """The host's submission template — defines the required output format."""
    return _read_csv("sample_submission.csv")
