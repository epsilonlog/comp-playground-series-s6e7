# CLAUDE.md

Instructions for Claude Code / Codex in this repository. Read fully before any task.

---

## Division of labour

**The operator is learning. Do not write code they haven't asked for.**

| Operator writes | Claude writes | Claude never writes |
|---|---|---|
| `plots.py` (all EDA plotting) | Boilerplate: `pyproject.toml`, `.gitignore`, CI | Anything in `plots.py` |
| Feature ideas in `features.py` | `io.py`, dtype schemas | Whole solutions unprompted |
| The CV design decision | The harness once the design is chosen | Model code before the harness exists |
| Model/hyperparameter choices | Registry plumbing, type stubs | New experiments without being asked |
| `SOLUTION.md` | Git commits, branch hygiene | Config values |

When the operator asks "how do I X", **explain first, then offer code**. Do not
answer a conceptual question with a patch.

When asked to implement something whose design is undetermined, ask one
question rather than assume.

---

## Hard rules

1. **Never invent an API.** Any non-trivial call → Context7 (`resolve-library-id`
   then `query-docs`) before writing. Wrong kwargs silently produce a wrong model.
2. **No model before the CV harness exists.** Refuse and build the harness.
3. **Every transform fits inside the fold.** Encoders, scalers, imputers, target
   statistics, feature selection, vectorisers. Flag any full-data `.fit()` as a leak.
4. **One variable per experiment.** Two changes = zero information.
5. **No logic in notebooks.** Notebooks import from `src/` and display only.
6. **Folds are frozen** after step 3 of the workflow. Never regenerate them.
7. **Latest stable libraries.** Check the installed version before using an API;
   don't code against remembered older signatures.
8. **CPU-only torch locally.** 2 GB VRAM + CUDA 11.6 driver ceiling. Settled.

---

## MCP protocol

| Need | Tool |
|---|---|
| API signature, library behaviour | **Context7** |
| Pretrained model, dataset, HF docs | **Hugging Face** (`hf_fs search`, `hub_repo_details`) |
| Competition data, leaderboard, submission | **Kaggle** (`https://www.kaggle.com/mcp`) |

Code competitions submit a **notebook version**, never a file upload.

---

## Environment

| | Local (Windows) | Kaggle Notebooks |
|---|---|---|
| Python | 3.12 (matches Kaggle exactly) | 3.12 |
| Package manager | `uv` | `uv` (Kaggle's own image uses it) |
| CPU | 4C/8T → `n_jobs=6` | varies |
| RAM | 16 GB, ~8.5 GB free | 30 GB |
| GPU | none usable | P100 16 GB / 2×T4 32 GB |
| Role | author code, 10% subsample runs | full training, GPU work, final submission |

**If a local run exceeds ~10 minutes, it belongs on Kaggle.**

Quotas: ~30 GPU-h/week, 9 h max GPU session, 12 h CPU, 20 GB storage.
Checkpoint every long run.

GBDT on GPU is worth it on Kaggle for large data (`device="cuda"`,
`tree_method="hist"`). Note GPU/CPU binning differs slightly — never mix devices
within one CV comparison.

---

## Repo layout

`src/` layout, per Python Packaging Authority — prevents accidental imports from
the working directory.

```
comp-<slug>/
├─ CLAUDE.md
├─ README.md
├─ pyproject.toml
├─ configs/
│  ├─ base.yaml
│  └─ exp_0001.yaml          # immutable once run
├─ data/                     # gitignored
│  ├─ raw/
│  └─ processed/             # parquet only
├─ src/<pkg>/
│  ├─ config.py              # frozen dataclasses
│  ├─ protocols.py           # Protocol definitions
│  ├─ registry.py            # dict-based factories
│  ├─ io.py
│  ├─ features.py
│  ├─ folds.py
│  ├─ metric.py
│  ├─ cv.py
│  ├─ plots.py               # OPERATOR OWNS THIS FILE
│  └─ models/
├─ tests/
├─ notebooks/
│  ├─ 01_eda.ipynb
│  └─ kaggle_submit.ipynb
├─ oof/                      # exp_XXXX.npy
├─ experiments.csv
└─ SOLUTION.md
```

---

## Python style

Idioms follow *Fluent Python* (Ramalho, 2nd ed.).

**Protocols over inheritance** — structural typing, no ABC hierarchy:

```python
from typing import Protocol
import numpy as np

class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Estimator": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

**Registry dict over Factory class** — in Python, a dict of callables replaces a
factory hierarchy:

```python
BUILDERS: dict[str, Callable[[dict], Estimator]] = {}

def register(name: str):
    def deco(fn):
        BUILDERS[name] = fn
        return fn
    return deco

@register("lgbm")
def _lgbm(params: dict) -> Estimator:
    import lightgbm as lgb
    return lgb.LGBMRegressor(**params)

def build(name: str, params: dict) -> Estimator:
    return BUILDERS[name](params)
```

**Frozen dataclasses for config:**

```python
@dataclass(frozen=True, slots=True)
class CVConfig:
    n_splits: int = 5
    seed: int = 42
    group_col: str | None = None
```

**Generators for folds** — yield, don't materialise:

```python
def iter_folds(folds: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for k in np.unique(folds):
        yield np.where(folds != k)[0], np.where(folds == k)[0]
```

Also: `functools.singledispatch` for type-based branching, `cached_property` for
expensive derived attributes, `pathlib` never `os.path`, f-strings, `|` unions.

Ruff (format + lint, line length 100). Type hints on every public function.
`SEED = 42` in `config.py`, seeded into numpy, `random`, and every model.

---

## Data layer

**Polars default.** Lower memory than pandas, and `scan_*` filters/projects
before materialising — decisive at 8.5 GB free.

`.to_numpy()` is zero-copy only when the frame is contiguous, numeric,
null-free, `order="fortran"`, `writable=False`. Break any condition and you
silently pay a full copy. Convert once, at the model boundary.

pandas only when a library demands it: `df.to_pandas(use_pyarrow_extension_array=True)`.

Parquet for all intermediates, never CSV.

---

## plots.py — operator-owned interface

Claude may review and suggest, **never implement**.

Contract: every function takes a Polars frame, returns a `matplotlib.figure.Figure`,
never calls `plt.show()`, never saves. The caller decides.

```python
def target_overview(df, target) -> Figure
def missingness(df) -> Figure
def numeric_grid(df, cols, target=None) -> Figure
def categorical_grid(df, cols, target) -> Figure
def correlation(df, cols, method="spearman") -> Figure

# The four that actually win competitions:
def train_test_shift(train, test, cols) -> Figure    # per-feature distribution overlay
def fold_distribution(df, folds, target) -> Figure   # target/size balance per fold
def oof_diagnostics(y, oof) -> Figure                # residuals, calibration, error by segment
def importance(model, names, top=30) -> Figure
```

The bottom four are absent from every tutorial plotting library and are where the
signal is. Build the top five first (they get used on day one), then the bottom four.

---

## CV protocol

The core of the project. Everything else is secondary.

1. **Reimplement the competition metric** in `metric.py`. Unit-test against any
   host-provided example.
2. **Choose the fold structure** by asking: *how does test differ from train?*
   - i.i.d. rows → `StratifiedKFold`
   - repeated entity (user, well, store, patient) → `GroupKFold`
   - both → `StratifiedGroupKFold`
   - time-ordered → forward-chaining, never random
3. **Adversarial validation.** Classify train vs test.
   - AUC ≈ 0.5 → distributions match
   - AUC high → shift exists; read feature importance to find the leaking columns,
     then design folds or drop features
4. **Freeze folds** to `data/processed/folds.parquet`. Every subsequent experiment
   uses these exact folds. Changing them invalidates all prior comparisons.
5. **Baseline:** LightGBM defaults, raw features. Record OOF. This is your floor.
6. **Submit once, early.** Record CV and LB together to measure their correlation.

Rules:

- Save OOF to `oof/exp_XXXX.npy` for every run. Without OOF you cannot blend.
- Report `cv_mean` **and** `cv_std`. A 0.001 gain with 0.004 fold spread is noise.
- Repeated CV (multiple seeds) when the metric is noisy; more folds when data is small.
- **Select finals by CV.** Write the rule in `SOLUTION.md` before looking at the LB.

**Benchmark to internalise** — ROGII 3rd place: 5-fold GroupKFold by well under
five split patterns → 25 checkpoints per model family. Local CV RMSE 5.2884,
public 6.043, private 5.836. Pessimistic CV, private beat public. That is what a
trustworthy harness produces.

---

## Experiment tracking

Append-only `experiments.csv`. Never edit a past row.

```csv
exp_id,date,config,model,folds,cv_mean,cv_std,lb_public,oof_path,runtime_s,parent,changed
```

- `exp_id` monotonic, appears in the config filename and the OOF filename.
- `changed` states the **one** thing that differs from `parent`.
- Log failed experiments. Otherwise you retry them in week 3.
- Optuna studies persist to SQLite in `data/processed/`, not memory.

Plain CSV over wandb: zero setup, works offline, survives a killed Kaggle session,
diffs in git. Graduate later if the ledger outgrows it.

---

## Git

Repo per competition, named `comp-<slug>`. Claude handles commits and branches.

- `main` holds working code only.
- Branch per experiment family: `exp/target-encoding`, `exp/lgbm-tuning`.
- Conventional commits: `feat:`, `fix:`, `exp:`, `docs:`, `refactor:`.
- Commit `configs/`, `experiments.csv`, and `SOLUTION.md`. Never commit `data/`
  or `oof/`.
- Tag the final submission: `git tag final-sub-1`.

---

## Workflow

| # | Step | Done when |
|---|---|---|
| 1 | Read overview, data, rules. Reimplement metric | `pytest` green on `metric.py` |
| 2 | EDA using `plots.py` | `01_eda.ipynb` committed |
| 3 | **CV harness.** Adversarial validation. Freeze folds | `folds.parquet` exists |
| 4 | Baseline + one submission | CV and LB in `experiments.csv` |
| 5 | Iterate, one variable at a time | ≥15 logged experiments |
| 6 | Blend strong singles | Blend beats best single on OOF |
| 7 | Select finals by CV | Two chosen, rule documented |
| 8 | Write `SOLUTION.md`, read top 5 writeups, reproduce one technique | Published |

Step 8 is where most of the learning is. Do not skip it.

---

## Anti-patterns

| Anti-pattern | Rule |
|---|---|
| Chasing public LB | Selection rule written before looking |
| Full-data preprocessing | Fit inside the fold |
| Regenerating folds mid-competition | Frozen at step 3 |
| Logic in notebooks | `src/` only |
| Multiple simultaneous changes | One variable per `exp_id` |
| Claude writing `plots.py` | Operator owns it |
| Claude writing unrequested code | Explain first |
| Installing CUDA locally | Settled. Don't revisit |
| Two competitions at once | One, through step 8 |
| Joining late | First two weeks or skip |
