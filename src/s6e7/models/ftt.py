"""FT-Transformer builders via masamlp — the step-8 reproduction of Kawamata's recipe.

Source: "S6E7 | FT-Transformer-v2 [CV: 0.95063]" (masayakawamata), private LB 0.95084,
the model both the 2nd- and 4th-place solutions built on. Two registered names, one
experimental variable apart:

- ``ftt``    — the network on the 13 raw baseline columns (exp_0013: the model family)
- ``ftt_te`` — the same network plus 39 per-value target-encoding features, 13 columns
               x 3 classes (exp_0014: the notebook's measured lever, +0.0012 on 5/5 folds)

``masamlp`` and ``catstat`` are the source notebook's own libraries, pinned to its exact
versions (masamlp==0.3.0, catstat==0.4.0) so the recipe is reproduced against the code
that produced its numbers. Neither is installed locally — training happens on Kaggle GPU
(CLAUDE.md: >10 min belongs on Kaggle) — so both imports are deferred to fit time and
this module imports cleanly everywhere.

Deliberate deviation from the source, priced by its own paired measurement: ``n_ens=1``
instead of 4 — the in-model ensemble buys +0.00017 balanced accuracy (4/7 folds, inside
the metric's discreteness) for 4x the wall clock. We keep the wall clock.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from s6e7 import features, io
from s6e7.config import SEED
from s6e7.protocols import Classifier
from s6e7.registry import register

#: The source notebook's MODEL_PARAMS, verbatim except ``n_ens`` (module docstring) and
#: ``categorical_features`` (ours are the baseline layout's 6 integer-coded columns).
KAWAMATA_PARAMS: dict[str, Any] = {
    "model": "ft_transformer",
    "n_epochs": 16,
    "batch_size": 4096,
    "learning_rate": 0.001,
    "weight_decay": 1e-05,
    "optimizer": "adamw",
    "optimizer_betas": None,
    "lr_scheduler": "cosine",
    "weight_decay_schedule": "none",
    "grad_clip": None,
    "num_embedding": "plr-lite",
    "numeric_scaler": "quantile",
    "cat_encoding": "embedding",
    "class_weight": None,
    "label_smoothing": 0.0,
    "early_stopping_rounds": None,  # fixed budget: no checkpoint picked by the held-out fold
    "eval_metric": "multi_logloss",
    "n_ens": 1,
    "device": "auto",
    "amp": "auto",
    "verbose": 0,
    "ens_mode": "loop",
    "eval_batch_size": 2048,
    "model_params": {
        "d_block": 128,
        "n_blocks": 2,
        "attention_n_heads": 8,
        "n_frequencies": 24,
        "sigma": 0.1,
    },
    "random_state": SEED,
}

#: The 6 categorical columns by their baseline-layout names. masamlp embeds each distinct
#: value per column (missing = a reserved index), so integer codes work the same as the
#: raw strings the source notebook passed.
_CAT_NAMES: list[str] = list(io.ORDINAL_COLS + io.NOMINAL_COLS)


class FTTransformer:
    """masamlp behind the harness protocol: float32 matrix in, probabilities out.

    The harness hands the baseline-layout numpy matrix; masamlp wants a frame plus the
    categorical column names. With ``with_te`` the catstat encoder — internally
    out-of-fold, so a row never encodes itself — is fitted on the fit rows only and its
    39 columns appended at both fit and predict time. The encoder is a *fitted*
    transform, which is exactly why it lives inside the estimator, inside the fold
    (CLAUDE.md rule 3), and not in features.py (declared transforms only).
    """

    def __init__(self, params: dict[str, Any], *, with_te: bool) -> None:
        self._params = params
        self._with_te = with_te
        self._enc: Any = None
        self._model: Any = None

    @property
    def classes_(self) -> NDArray[np.int64]:
        """masamlp orders proba columns by sorted unique labels — our 0..K-1 codes."""
        return np.asarray(self._model.classes_)

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]) -> FTTransformer:
        from masamlp import MasaClassifier

        frame = self._frame(X)
        if self._with_te:
            from catstat import TargetEncoder

            # The source notebook's encoder, verbatim: exact-value ("direct") encoding
            # of all 13 columns, one mean per class, empirical-Bayes smoothing.
            self._enc = TargetEncoder(
                random_state=SEED,
                cols=features.BASE_NAMES,
                stats=("mean",),
                target_type="multiclass",
                smooth="auto",
                numeric="direct",
            )
            encoded = self._enc.fit_transform(frame[features.BASE_NAMES], np.asarray(y))
            frame = self._augment(frame, encoded)
        self._model = MasaClassifier(**self._params, categorical_features=_CAT_NAMES)
        self._model.fit(frame, np.asarray(y))
        return self

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        frame = self._frame(X)
        if self._with_te:
            frame = self._augment(frame, self._enc.transform(frame[features.BASE_NAMES]))
        return np.asarray(self._model.predict_proba(frame), dtype=np.float64)

    @staticmethod
    def _frame(X: NDArray[np.floating[Any]]) -> Any:
        import pandas as pd

        return pd.DataFrame(np.asarray(X, dtype=np.float32), columns=features.BASE_NAMES)

    @staticmethod
    def _augment(frame: Any, encoded: Any) -> Any:
        import pandas as pd

        te = np.asarray(encoded, dtype=np.float32)
        te_frame = pd.DataFrame(te, columns=[f"te_{i}" for i in range(te.shape[1])])
        return pd.concat([frame.reset_index(drop=True), te_frame], axis=1)


@register("ftt")
def build_ftt(params: dict[str, Any]) -> Classifier:
    return FTTransformer({**KAWAMATA_PARAMS, **params}, with_te=False)


@register("ftt_te")
def build_ftt_te(params: dict[str, Any]) -> Classifier:
    return FTTransformer({**KAWAMATA_PARAMS, **params}, with_te=True)
