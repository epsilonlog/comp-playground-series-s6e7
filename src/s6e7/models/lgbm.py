"""LightGBM builder: library defaults plus the three house params (seed, threads, quiet).

Anything in `params` wins over the house defaults, so an experiment can override even the
seed — deliberately, in a config, where the ledger records it.
"""

from __future__ import annotations

from typing import Any

from s6e7.config import N_JOBS, SEED
from s6e7.protocols import Classifier
from s6e7.registry import register


@register("lgbm")
def build_lgbm(params: dict[str, Any]) -> Classifier:
    import lightgbm as lgb

    merged: dict[str, Any] = {"random_state": SEED, "n_jobs": N_JOBS, "verbosity": -1, **params}
    model: Classifier = lgb.LGBMClassifier(**merged)
    return model
