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

`workflow step 3 complete` — adversarial validation found a real shift, the shift was
traced to a target-orthogonal direction, and the folds are **frozen** in
`data/processed/folds.parquet`. Remaining before step 4: write the selection rule.

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

**Fold structure — frozen 2026-08-28:** `StratifiedKFold(5)`, stratified on the target
alone, `seed=42`. `data/processed/folds.parquet` holds `id` / `fold` / `null_bucket`;
`folds.verify()` re-derives the assignment and asserts the file still matches.

Chosen *after* the adversarial result below, not in ignorance of it. The shift is real
but sits in a direction that carries no target signal, and correcting for it — importance
weighting, or test-like pessimistic folds — would buy a ≤0.002 correction with weights
noisier than the thing they correct. Stratifying on the null bucket was priced at
**0.00004** against a 0.002 fold-level SE and rejected; the bucket is carried as a
**diagnostic slice key** instead, so every experiment can report per-slice recall.

**Falsifiable consequence:** test carries more null-heavy rows than train (k ≥ 3: 4.55%
vs 2.17%), so its rows are information-poorer. **LB should land ~0.001–0.002 below CV.**
A gap of 0.02 means this analysis is not the explanation.

**Splitter family:** `id` is unique across all 690k train and 295k test rows, so no
repeating entity — `GroupKFold` is out. Nothing time-ordered. At 5.8% minority,
stratification is required regardless: 5 folds gives ~7,960 `fit` rows each.

**Measurement resolution** (computed before the first run, so it can be falsified):
per-fold `SE(balanced accuracy) ≈ 0.0023`, `SE(cv_mean) ≈ 0.001`. Expect `cv_std ≈ 0.002`.
Treat +0.0008 as noise. `fit` supplies 55% of the variance — the effective sample size is
~40k rows, not 690k.

**Adversarial validation AUC: 0.6518** (per-fold sd 0.0012, `id` excluded).
Shuffled-label control 0.5002 (`adversarial.shuffled_control`), so the result is not a
harness artifact. Expected ~0.5; it is not. The shift is in the *joint* missingness
structure and every univariate check passes. Two follow-ups located it:

| run | AUC | reading |
|---|---|---|
| all rows, 13 features | 0.6518 | a real shift |
| complete-case rows only (k=0) | 0.5304 | strip missingness and it mostly vanishes |
| best single feature, complete cases | 0.5186 | `smoking_alcohol`; mild categorical drift |
| shuffled control, complete cases | 0.4999 | not a harness artifact |

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
| `src/s6e7/folds.py` | the frozen partition: build, verify, iterate, describe |
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

- **2026-08-28 — folds frozen: the shift was real and still didn't change the design.**
  Two checks closed the question the 0.6518 opened. Re-running adversarial validation on
  **complete-case rows only** (349,623 train / 165,084 test, no nulls at all, so the
  classifier cannot see missingness) gives **AUC 0.5304** against a 0.4999 shuffled
  control — strip the null pattern and the shift mostly goes with it. And
  `p(target | n_nulls, bmi_is_null)` is **flat in the null count**: 0.0847 / 0.0851 /
  0.0832 / 0.0889 at k = 0…3 with `bmi` present, 0.0279 / 0.0299 / 0.0311 with it null.
  All of missingness's target signal is `bmi_is_null`, whose marginal rate is **identical
  in both files (2.014%)**. What shifted predicts nothing. Options priced before choosing:
  compound stratification on the null bucket buys 0.00004 against a 0.002 fold SE;
  importance weighting and pessimistic folds correct ≤0.002 with estimated weights. Chose
  `StratifiedKFold(5)` on the target alone and carried `null_bucket` as a diagnostic
  column. The freeze verifies: `fit` counts 7961/7961/7961/7960/7960, k≥3 share
  2.162–2.221% against a predicted ±0.039pp — and test's 4.564%, which no fold resembles.
- **2026-08-28 — a covariate shift is only a fold problem if it moves the ranking.** The
  reflex on AUC 0.65 is to redesign folds. The question that actually decides it is
  whether test reweights a region where *models differ*. Two models tied overall but
  differing by δ on the k≥3 slice have their gap moved by 0.0238·δ, so δ must exceed
  **0.042** before the reweighting outruns the 0.001 resolution. The one design axis where
  that isn't absurd is **missing-data handling** — which is exactly what the diagnostic
  bucket is there to catch.
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
