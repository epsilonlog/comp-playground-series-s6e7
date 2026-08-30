"""XGBoost builder. Joins the pool for diversity, not to replace LightGBM.

`tree_method="hist"` matches LightGBM's histogram binning; NaN is handled natively
(split directions for missing are learned during training).
"""

from __future__ import annotations

from typing import Any

from s6e7.config import N_JOBS, SEED
from s6e7.protocols import Classifier
from s6e7.registry import register


@register("xgb")
def build_xgb(params: dict[str, Any]) -> Classifier:
    from xgboost import XGBClassifier

    merged: dict[str, Any] = {
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "tree_method": "hist",
        "verbosity": 0,
        **params,
    }
    model: Classifier = XGBClassifier(**merged)
    return model
