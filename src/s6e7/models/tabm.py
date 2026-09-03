"""TabM builder — the second neural family, on the harness's own preprocessing.

TabM (Gorishniy et al., ICLR 2025, ``pip install tabm``) is one MLP that efficiently
represents an ensemble of *k* MLPs through weight sharing: all k members train in
parallel on the same batches, the loss is the mean over members, and inference averages
the k probability vectors. Used here as a **baseline**: ``TabM.make`` defaults with
piecewise-linear embeddings, the README's recommended optimizer, early stopping on an
inner holdout. Nothing is tuned.

Why this family: notebook 07 measured the two GBDT families wrong together 27x more
often than independence allows (error correlation 0.885), and the only decorrelation
anyone found in this competition came from a neural net. TabM is the cheapest strong
neural baseline, and its OOF is what a stack would consume.

The wrapper owns every *fitted* preprocessing step, inside ``fit`` and therefore inside
the fold (CLAUDE.md rule 3):

- **numeric** — median imputation, then ``QuantileTransformer`` to a normal, both fitted
  on the fit rows. A 0/1 null flag per base numeric column (as a categorical of
  cardinality 2) keeps the one informative null, ``bmi_is_null``, that imputation would
  otherwise erase.
- **categorical** — the baseline layout's integer codes with null as a reserved extra
  level; TabM one-hot encodes them.
- **piecewise-linear bins** — quantile bins computed on the transformed fit rows.
- **class weights** — ``balanced`` in the cross-entropy: the training-route prior
  correction exp_0004 established (+0.077). The OOF argmax is then the metric-correct
  decision, and the ledger row is comparable with exp_0004/exp_0017 without a rule run.
- **early stopping** — on a stratified holdout carved from the *fit* rows, never on the
  fold's held-out rows: the fold's score must not pick the checkpoint. The stopping
  criterion is the class-weighted NLL of the ensemble-mean probability, i.e. the
  ensemble's objective, which is the point of training the k members in parallel.

Columns beyond the 13-column base layout (the exact-value TE columns ``cv.run`` appends
when a config names an encoder) are treated as numeric.

``tabm`` and ``rtdl_num_embeddings`` are project dependencies (CPU torch locally, GPU on
Kaggle); the imports are deferred to fit time so the registry imports without torch.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from s6e7 import features, io
from s6e7.config import SEED
from s6e7.protocols import Classifier
from s6e7.registry import register

#: The baseline recipe. Architecture = ``TabM.make`` defaults when embeddings are given;
#: embeddings = the example notebook's PLE setting; optimizer = the README's AdamW.
TABM_PARAMS: dict[str, Any] = {
    "k": 32,
    "n_blocks": 2,
    "d_block": 512,
    "dropout": 0.1,
    "n_bins": 48,
    "d_embedding": 16,
    "lr": 2e-3,
    "weight_decay": 3e-4,
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 8,
    "holdout": 0.1,
    "class_weight": "balanced",  # or None for the plain likelihood (needs a rule run)
    "eval_batch_size": 8192,
    "device": "auto",
    "amp": True,
    "seed": SEED,
    "verbose": True,
}

_N_BASE: int = len(features.BASE_NAMES)
_CAT_IDX: list[int] = list(features.CATEGORICAL_IDX)
_BASE_NUM_IDX: list[int] = [i for i in range(_N_BASE) if i not in _CAT_IDX]
#: Declared levels plus one reserved level for null, in CATEGORICAL_IDX order.
_CAT_CARDINALITIES: list[int] = [len(io.ORDINAL_LEVELS[c]) + 1 for c in io.ORDINAL_COLS] + [
    len(io.NOMINAL_LEVELS[c]) + 1 for c in io.NOMINAL_COLS
]


def _pick_device(spec: str) -> Any:
    import torch

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class TabMClassifier:
    """TabM behind the harness protocol: float32 matrix in, probabilities out."""

    def __init__(self, params: dict[str, Any]) -> None:
        self._p: dict[str, Any] = {**TABM_PARAMS, **params}
        self._n_features: int = 0
        self._medians: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._scaler: Any = None
        self._model: Any = None
        self._device: Any = None
        self._amp_dtype: Any = None
        self._use_amp: bool = False
        self.best_epoch_: int = -1
        self.history_: list[dict[str, float]] = []

    @property
    def classes_(self) -> NDArray[np.int64]:
        return np.arange(len(io.CLASSES), dtype=np.int64)

    # -- preprocessing ------------------------------------------------------------------

    def _num_idx(self) -> list[int]:
        return _BASE_NUM_IDX + list(range(_N_BASE, self._n_features))

    def _cardinalities(self) -> list[int]:
        return _CAT_CARDINALITIES + [2] * len(_BASE_NUM_IDX)

    def _numeric(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        num = X[:, self._num_idx()]
        num = np.where(np.isnan(num), self._medians, num)
        return np.asarray(self._scaler.transform(num), dtype=np.float32)

    @staticmethod
    def _categorical(X: NDArray[np.float32]) -> NDArray[np.int64]:
        codes = X[:, _CAT_IDX]
        null_level = np.asarray(_CAT_CARDINALITIES, dtype=np.float32) - 1.0
        codes = np.where(np.isnan(codes), null_level, codes)
        flags = np.isnan(X[:, _BASE_NUM_IDX])
        return np.hstack([codes, flags]).astype(np.int64)

    # -- protocol -------------------------------------------------------------------------

    def fit(self, X: NDArray[np.floating[Any]], y: NDArray[np.integer[Any]]) -> TabMClassifier:
        import tabm
        import torch
        from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import QuantileTransformer
        from torch.nn import functional as F

        p = self._p
        torch.manual_seed(p["seed"])
        rng = np.random.default_rng(p["seed"])
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)
        n_classes = len(io.CLASSES)
        self._n_features = X_arr.shape[1]
        self._device = dev = _pick_device(p["device"])

        # Fitted numeric preprocessing: medians and quantiles from the fit rows only.
        num_raw = X_arr[:, self._num_idx()]
        medians = np.nanmedian(num_raw, axis=0).astype(np.float32)
        self._medians = np.where(np.isnan(medians), np.float32(0.0), medians)
        num = np.where(np.isnan(num_raw), self._medians, num_raw)
        noise = rng.normal(0.0, 1e-5, num.shape).astype(np.float32)  # the example's trick
        self._scaler = QuantileTransformer(
            n_quantiles=max(min(len(num) // 30, 1000), 10),
            output_distribution="normal",
            subsample=10**9,
            random_state=p["seed"],
        ).fit(num + noise)
        x_num = self._numeric(X_arr)
        x_cat = self._categorical(X_arr)

        # Early-stopping holdout carved from the fit rows, stratified on the target.
        train_idx, hold_idx = train_test_split(
            np.arange(len(y_arr)), test_size=p["holdout"], stratify=y_arr, random_state=p["seed"]
        )

        weight = None
        if p["class_weight"] == "balanced":
            counts = np.maximum(np.bincount(y_arr, minlength=n_classes), 1).astype(np.float64)
            weight = torch.tensor(
                len(y_arr) / (n_classes * counts), dtype=torch.float32, device=dev
            )
        elif p["class_weight"] is not None:
            msg = f"class_weight must be 'balanced' or None, got {p['class_weight']!r}"
            raise ValueError(msg)

        t_num = torch.as_tensor(x_num, device=dev)
        t_cat = torch.as_tensor(x_cat, device=dev)
        t_y = torch.as_tensor(y_arr, device=dev)
        tr = torch.as_tensor(train_idx, device=dev)
        ho = torch.as_tensor(hold_idx, device=dev)

        # Bins on CPU: deterministic, and torch.quantile on CUDA has an element ceiling.
        bins = compute_bins(torch.as_tensor(x_num[train_idx]), n_bins=p["n_bins"])
        embeddings = PiecewiseLinearEmbeddings(
            bins, d_embedding=p["d_embedding"], activation=False, version="B"
        )
        model = tabm.TabM.make(
            n_num_features=x_num.shape[1],
            cat_cardinalities=self._cardinalities(),
            d_out=n_classes,
            num_embeddings=embeddings,
            k=p["k"],
            n_blocks=p["n_blocks"],
            d_block=p["d_block"],
            dropout=p["dropout"],
        ).to(dev)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"]
        )

        self._use_amp = bool(p["amp"]) and dev.type == "cuda"
        self._amp_dtype = (
            torch.bfloat16 if self._use_amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        scaler = (
            torch.amp.GradScaler("cuda")
            if self._use_amp and self._amp_dtype == torch.float16
            else None
        )
        self._model = model

        def loss_of(logits: Any, target: Any) -> Any:
            # (B, k, C) -> (B*k, C): every member is trained on its own prediction —
            # the mean LOSS, never the loss of the mean prediction (TabM README).
            return F.cross_entropy(
                logits.flatten(0, 1), target.repeat_interleave(model.k), weight=weight
            )

        def holdout_metrics() -> tuple[float, float]:
            proba = self._proba_tensor(t_num[ho], t_cat[ho])
            nll = F.nll_loss(torch.log(proba.clamp_min(1e-12)), t_y[ho], weight=weight)
            pred = proba.argmax(1)
            recalls = [
                (pred[t_y[ho] == c] == c).float().mean()
                for c in range(n_classes)
                if bool((t_y[ho] == c).any())
            ]
            return float(nll), float(torch.stack(recalls).mean())

        gen = torch.Generator(device=dev).manual_seed(p["seed"])
        best_state: dict[str, Any] | None = None
        best_nll, remaining = math.inf, int(p["patience"])
        self.history_ = []
        for epoch in range(int(p["max_epochs"])):
            model.train()
            perm: Any = tr[torch.randperm(len(tr), device=dev, generator=gen)]
            for batch in perm.split(int(p["batch_size"])):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_of(self._forward(t_num[batch], t_cat[batch]), t_y[batch])
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            nll, ba = holdout_metrics()
            improved = nll < best_nll
            self.history_.append({"epoch": epoch, "holdout_nll": nll, "holdout_ba": ba})
            if p["verbose"]:
                print(
                    f"  epoch {epoch:3d}  holdout nll {nll:.4f}  ba {ba:.5f}"
                    f"{'  *' if improved else ''}",
                    flush=True,
                )
            if improved:
                best_nll, self.best_epoch_ = nll, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                remaining = int(p["patience"])
            else:
                remaining -= 1
                if remaining < 0:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        import torch

        if self._model is None:
            msg = "model is not fitted"
            raise RuntimeError(msg)
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.shape[1] != self._n_features:
            msg = f"expected {self._n_features} columns, got {X_arr.shape[1]}"
            raise ValueError(msg)
        t_num = torch.as_tensor(self._numeric(X_arr), device=self._device)
        t_cat = torch.as_tensor(self._categorical(X_arr), device=self._device)
        proba: NDArray[np.floating[Any]] = self._proba_tensor(t_num, t_cat).cpu().numpy()
        return proba.astype(np.float64)

    # -- torch plumbing ---------------------------------------------------------------------

    def _forward(self, x_num: Any, x_cat: Any) -> Any:
        import torch

        with torch.autocast(self._device.type, dtype=self._amp_dtype, enabled=self._use_amp):
            out = self._model(x_num, x_cat)
        return out.float()

    def _proba_tensor(self, x_num: Any, x_cat: Any) -> Any:
        """Ensemble-mean probabilities, (n, C): softmax per member, then mean over k."""
        import torch

        self._model.eval()
        chunks = []
        with torch.inference_mode():
            rows: Any = torch.arange(len(x_num), device=self._device)
            for idx in rows.split(int(self._p["eval_batch_size"])):
                logits = self._forward(x_num[idx], x_cat[idx])
                chunks.append(torch.softmax(logits, dim=-1).mean(1))
        return torch.cat(chunks)


@register("tabm")
def build_tabm(params: dict[str, Any]) -> Classifier:
    return TabMClassifier(params)
