# comp-playground-series-s6e7

Predict student health risk (`at-risk` / `unhealthy` / `fit`) from tabular lifestyle features.

| | |
|---|---|
| Competition | [Playground Series S6E7](https://www.kaggle.com/competitions/playground-series-s6e7) |
| Type | Playground · medals: no |
| Metric | **balanced accuracy** (macro-average recall, 3 classes) |
| Deadline | closed — late submissions only |
| Teams | 3,355 |
| Entry | solo |
| Result | *<rank / n>* |

Training competition. The deliverable is a reusable framework, not a rank.

## Status

`workflow step 3 in progress` — adversarial validation done and it found a real shift.
Fold design is **reopened**; folds are NOT frozen.

| | |
|---|---|
| Best CV | — |
| Best public LB | — |
| Private LB | — |
| Experiments run | 0 |

## Data

| | rows | cols |
|---|---|---|
| train | 690,088 | 15 |
| test | 295,753 | 14 |

7 numeric, 6 categorical (3 levels each), `id`, and the target. Target is
**86 / 8 / 6**: `at-risk` 592,561 · `unhealthy` 57,724 · `fit` 39,803.

**The floor is 0.333.** Always-predict-`at-risk` scores 85.9% plain accuracy and 0.333
balanced accuracy.

## Validation

**Fold structure:** *reopened — see below.* `StratifiedKFold(5)`, stratified on the
target alone, was chosen on the evidence that train and test are one distribution. That
premise did not survive adversarial validation.

**Splitter family:** `id` is unique across all 690k train and 295k test rows, so no
repeating entity — `GroupKFold` is out. Nothing time-ordered. At 5.8% minority,
stratification is required regardless: 5 folds gives ~7,960 `fit` rows each.

**Measurement resolution** (computed before the first run, so it can be falsified):
per-fold `SE(balanced accuracy) ≈ 0.0023`, `SE(cv_mean) ≈ 0.001`. Expect `cv_std ≈ 0.002`.
Treat +0.0008 as noise. `fit` supplies 55% of the variance — the effective sample size is
~40k rows, not 690k.

**Adversarial validation AUC: 0.6518** (per-fold sd 0.0012, `id` excluded).
Shuffled-label control 0.5002 (`adversarial.shuffled_control`), so the result is not a
harness artifact. Expected ~0.5;
it is not. See the Log entry below — the shift is in the *joint* missingness structure
and every univariate check passes.

**Selection rule** (written before looking at the LB): *<step 3>*

## Setup

```bash
uv sync
uv run kaggle competitions download -c playground-series-s6e7 -p data/raw
uv run python -c "from s6e7 import io; io.build_parquet()"
```

`uv` and `gh` are installed but **not on PATH** — see the note in the Log.

## Layout

| Path | Contents |
|---|---|
| `configs/` | one YAML per experiment, immutable once run |
| `src/s6e7/io.py` | frozen dtype schema, CSV→parquet cache, column roles |
| `src/s6e7/metric.py` | balanced accuracy, tested against sklearn |
| `src/s6e7/adversarial.py` | train-vs-test classifier — the test of the i.i.d. premise |
| `src/s6e7/config.py` | `SEED`, `N_JOBS` |
| `src/s6e7/eda.py` | tabular EDA summaries — frames in, frames out |
| `src/s6e7/plots.py` | EDA plotting, **operator-owned**, stubs only so far |
| `notebooks/01_eda.ipynb` | EDA steps 1–7, display only |
| `notebooks/02_train_test_shift.ipynb` | marginals, joint missingness, adversarial validation |
| `oof/` | out-of-fold predictions, `exp_XXXX.npy` |
| `experiments.csv` | append-only ledger |
| `LEARNING.md` | concepts, not code — the transferable part |

## Log

Notable findings, dead ends, and things worth remembering next competition.

- **2026-08-28 — train and test differ, and no marginal shows it.** Adversarial AUC
  **0.6518** (sd 0.0012) while every feature's *solo* adversarial AUC sits at chance
  (0.4989–0.5216). Per-column null rates match to five decimals; means within 0.006 SD;
  largest Spearman difference 0.006. The shift is the **number of nulls per row**: train
  matches the column-independence baseline (50.76% predicted / 50.66% observed with zero
  nulls, variance 0.5967 vs 0.6012), test does not (55.82% with zero nulls, variance
  0.7916 — **32% overdispersed**, same mean 0.6514). Train's nulls are drawn
  independently per column; test's co-occur. Fold design is reopened.
- **2026-08-28 — adversarial gain importances misled here.** `water_intake` ranked first
  at 32.3% gain with a solo AUC of 0.4990; `gender` ranked eighth at 2.6% gain and is the
  most-shifted single column (solo AUC 0.5216). Gain rewards split opportunities, and a
  12,000-value continuous column has vastly more than a 3-level categorical. Read the AUC
  first and never take the importance ranking as a shift ranking.
- **2026-08-28 — `stress_level` is a gate on the minority classes.** `fit` lives almost
  only at `low` (p=0.2006 vs 0.003 elsewhere); `unhealthy` almost only at `high`
  (p=0.2787 vs 0.003). `medium` is 99.4% `at-risk`. 84.5% of all `fit` rows are at `low`;
  85.8% of all `unhealthy` at `high`. The problem decomposes into two sub-problems rather
  than one 5.8% needle.
- **2026-08-28 — `sleep_duration` is the strongest numeric by far.** spread 2.13 SD
  (`fit` 7.95 / `at-risk` 7.09 / `unhealthy` 5.37) → ~29% class overlap, and a single
  threshold would reach ~0.86 balanced accuracy on the two extreme classes.
- **2026-08-28 — `step_count` and `exercise_duration` only find `fit`.** Both score ~0.82
  spread, but `at-risk` and `unhealthy` sit on top of each other. Expect `unhealthy`
  recall to be the score bottleneck — it has fewer features pointing at it.
- **2026-08-28 — `bmi` missingness is the one informative null.** p(`unhealthy`) drops
  0.0848 → 0.0293 when `bmi` is null (~22σ). Add `bmi_is_null`. Every other column's
  nulls sit at the global rates — skip the other 12 indicators and skip imputation
  strategy entirely.
- **2026-08-28 — nulls are a fixed proportion per column, drawn independently.**
  690,088 × 1% = 6,900.88 → 6,901 exactly. Equal counts across columns are arithmetic,
  not a shared mask; joint missingness sits at the independence baseline.
- **2026-08-28 — four categoricals are ordinal, two nominal.** `stress_level`,
  `sleep_quality`, `physical_activity_level`, `smoking_alcohol` ordered in
  `io.ORDINAL_LEVELS`. `sleep_quality` and `smoking_alcohol` are cleanly monotone in both
  minority classes — monotone-constraint candidates. `diet_type` and `gender` are nominal
  and `gender` carries no signal at all.
- **2026-08-28 — no cleanup needed.** Zero duplicate feature rows, no test-only category
  levels, all skews within ±0.38, nothing clipped (pct-at-bound ≤ 0.05%). The one point
  mass is `exercise_duration` at exactly 0 — 2.41%, a genuine "does not exercise", not a
  coded missing (the column has its own explicit nulls).
- **2026-08-28 — no redundancy.** Max pairwise Spearman is 0.44
  (`step_count` × `exercise_duration`) — only 19% shared variance. But that trio invites
  *ratio* features (calories per step, steps per minute), which trees cannot form from
  splits.
- **2026-08-28 — feature grids are coarser than `max_bin`.** `step_count` has 12,807
  distinct values and `sleep_duration` 701, versus LightGBM's default 255 bins. `max_bin`
  is a real tunable here, not a throwaway.
- **2026-08-28 — tooling paths.** `uv` lives at
  `%LOCALAPPDATA%\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe` and `gh` at
  `C:\Program Files\GitHub CLI\gh.exe`; neither is on PATH in this shell. The git remote
  is HTTPS (SSH host-key verification fails). After adding a **new** top-level name to a
  module, restart the Jupyter kernel — `%autoreload 2` picks up changed function bodies
  but not newly added module-level names.
