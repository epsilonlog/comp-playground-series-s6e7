"""Structural typing for what the harness drives. Protocols, no ABC hierarchy.

Anything with sklearn's fit/predict_proba shape satisfies `Classifier` — nothing has to
inherit from anything, which is what lets the registry hold LightGBM, XGBoost, CatBoost
and hand-rolled models behind one name.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class Classifier(Protocol):
    """The two calls cv.run makes. Column order of predict_proba follows `classes_`."""

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]) -> Classifier: ...

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]: ...
