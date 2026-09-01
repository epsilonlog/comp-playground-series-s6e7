# CLAUDE.md

Instructions for Claude Code / Codex in this repository. Read fully before any task.

---

## Division of labour

**The operator is learning. Do not write code they haven't asked for.**

| Operator writes | Claude writes | Claude never writes |
|---|---|---|
| Feature ideas in `features.py` | `plots.py` and all plotting (transferred from operator, 2026-08-30) | Whole solutions unprompted |
| The CV design decision | The harness once the design is chosen | Model code before the harness exists |
| Model/hyperparameter choices | Boilerplate: `pyproject.toml`, `.gitignore`, CI; `io.py`, dtype schemas | New experiments without being asked |
| `SOLUTION.md` | Registry plumbing, type stubs; git commits, branch hygiene | Config values |

When the operator asks "how do I X", **explain first, then offer code**. Do not
answer a conceptual question with a patch.

When asked to implement something whose design is undetermined, ask one
question rather than assume.

**Keep explanations short** (operator feedback, 2026-08-30): one worked number per
idea, no parallel derivations, no exhaustive option surveys. If it takes three
screens, it's too long.

---

## Project state — settled decisions (updated 2026-08-30)

Facts a fresh session needs; do not re-derive or reopen without new evidence.

- **Task:** 3-class classification of `health_condition` (`at-risk` 85.9% / `unhealthy`
  8.4% / `fit` 5.8%). 690,088 train / 295,753 test rows, 13 features (7 numeric,
  6 categorical). Metric: balanced accuracy, reimplemented + tested in `metric.py`.
- **Resolution:** SE(`cv_mean`) ≈ 0.001, expected `cv_std` ≈ 0.002. Effects below
  ~0.001 are unmeasurable — don't chase them.
- **Folds frozen:** `StratifiedKFold(5)` on target alone, seed 42 →
  `data/processed/folds.parquet`, with `null_bucket` as a diagnostic slice key (not a
  stratification key). Run `folds.verify()` when in doubt; never rebuild.
- **Shift (closed):** adversarial AUC 0.6518 all rows / 0.5304 complete cases. The shift
  is null co-occurrence (test clumps nulls; k≥3 rows: 4.55% test vs 2.17% train) and is
  target-orthogonal. `bmi_is_null` carries the only missingness signal (unhealthy 2.79%
  when null vs 8.47% present) and its rate is identical in both files. Decision: no fold
  change; watch per-`null_bucket` recall on missing-data-handling experiments.
- **Standing prediction — confirmed 2026-08-30:** predicted LB 0.001–0.002 below CV;
  exp_0001 landed public 0.87210 (CV − 0.0008) and private 0.87171 (CV − 0.0012).
  CV↔LB correlation is trusted; select by CV. (Private scores are visible because the
  competition has ended — this is a late-submission practice run; LB rank is moot.)
- **Ladder complete (exp_0002–exp_0012, 2026-08-31):** encoding +0.0009 (paired t=4.8),
  capacity +0.0039, features ±0.0000 (two honest negatives), **prior correction +0.077**
  (dominates everything), routes converge (weights 0.94945 / searched rule 0.94931),
  XGBoost 0.88042 at argmax, blend dead (co-error 0.0293 vs an independence line of
  0.0011 — 27×, error correlation 0.885).
  Post-correction: four-way tie 0.94931–0.94956 inside one resolution.
- **The rule is arithmetic, not tuning (measured 2026-09-01):** zero-parameter `1/π`
  scores 0.94932, the 2-parameter grid search 0.94941 — the search buys +0.00009. The
  within-one-resolution plateau covers 9% of the multiplier box (`m_fit` 6–40,
  `m_unhealthy` 4–20), which is also why the five per-fold searches disagree 36% on
  `m_unhealthy` while the score moves 0.00006. Judge tuned post-processing on score
  stability, never parameter agreement. Derivation + landscape figure in notebook 07.
- **Selection rule (declared before further LB looks):** highest `cv_mean`; ties within
  0.001 break toward fewer moving parts. **Final = exp_0004** (LGBM defaults +
  `class_weight=balanced`); second pick = exp_0012 (blend + rule).
- **Stopped, deliberately:** remaining *knob* ideas (rule-on-xgb, CatBoost, Optuna,
  stacking) have best cases inside the tie band — unreadable even if they win.
- **New information source found (2026-09-01) — the stopping clause has fired.**
  `eda.exact_value_signal` (split-half replication test) measures genuine *per-value*
  target structure that a 255-bin histogram cannot reach: `sleep_duration` r=0.94 at
  SE 0.041 with a replicating residual sd of 0.094 against a 0.084 base rate;
  `water_intake` 0.0435; `heart_rate` 0.0232. `step_count`/`bmi`/`exercise_duration`/
  `calorie_expenditure` read r≈0. Concentrated in the bands where the minorities live,
  **not** in the mid-band crater (base rate 0.0013 there). Test coverage 89–99%.
  Evidence in notebook 06 §4. The fold-fitted per-value target-encoding experiment is
  licensed but **not yet run** — as is one FT-Transformer for the decorrelation the
  two-GBDT blend measurably lacks (exp_0013/0014 harness exists).
- **Carry to the next competition:** fix the metric-correct decision rule *before*
  laddering — argmax gains here largely vanished once the prior was corrected.

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
│  ├─ plots.py
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

## plots.py — the plotting interface

Claude-owned since 2026-08-30 (previously operator-owned). Every notebook shows its
figures — an analysis without the plot is half-delivered.

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

## LEARNING.md — the operator's notebook

The operator is here to learn by doing. `LEARNING.md` is the record of that.

**Claude maintains it without being asked.** After explaining any concept the operator
engaged with — asked a follow-up about, corrected, or worked through — append a section.
Do it in the same turn, don't batch it, and mention it in one line rather than reprinting
the content.

Rules:

- **Concepts and reasoning only.** No code, no config, no API signatures. If it lives in
  the repo, it doesn't belong here. This is the part that transfers to the next
  competition.
- **Append new sections at the end.** Dated `## YYYY-MM-DD — <concept>` headings.
- **Consolidate when asked, or when entries start overlapping.** Merge related sections,
  keep every worked number, drop the scaffolding. A short file gets reread; a long one
  doesn't.
- **Keep the worked numbers.** The toy example that made it click is the most valuable
  part of an entry. Don't compress it into an abstract statement.
- **Write what was actually understood**, including corrections of the operator's
  first reading — those are the entries that stick.
- Do not log routine build steps, tool output, or things the operator already knew.

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
| Notebooks without figures | `plots.py` exists to be used |
| Claude writing unrequested code | Explain first |
| Installing CUDA locally | Settled. Don't revisit |
| Two competitions at once | One, through step 8 |
| Joining late | First two weeks or skip |
