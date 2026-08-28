# comp-playground-series-s6e7

Predict student health risk from tabular survey/lifestyle features.

| | |
|---|---|
| Competition | [Playground Series S6E7 — Predicting Student Health Risk](https://www.kaggle.com/competitions/playground-series-s6e7) |
| Type | Playground · medals: no |
| Metric | *<TBD — framing question 1>* |
| Deadline | closed — late submissions only |
| Teams | *<n>* |
| Entry | solo |
| Result | *<rank / n>* |

Training competition. The deliverable is a reusable framework, not a rank.

## Status

`workflow step 1` — read overview, data, rules; reimplement metric. See `CLAUDE.md`.

| | |
|---|---|
| Best CV | — |
| Best public LB | — |
| Private LB | — |
| Experiments run | 0 |

## Validation

**Fold structure:** *<TBD — chosen at workflow step 3>*

**Why:** *<how test differs from train, and how the folds mirror it>*

**Adversarial validation AUC:** *<value>* → *<conclusion>*

**Selection rule** (written before looking at the LB): *<rule>*

## Setup

```bash
uv sync
kaggle competitions download -c playground-series-s6e7 -p data/raw && unzip data/raw/*.zip -d data/raw
uv run python -m src.s6e7.folds        # writes data/processed/folds.parquet
uv run python -m src.s6e7.cv --config configs/exp_0001.yaml
```

Local runs default to a 10% subsample; pass `--full` for the whole set.

## Layout

| Path | Contents |
|---|---|
| `configs/` | one YAML per experiment, immutable once run |
| `src/s6e7/` | all logic — notebooks import from here |
| `src/s6e7/plots.py` | EDA plotting (hand-written) |
| `notebooks/` | EDA and the Kaggle submission notebook |
| `oof/` | out-of-fold predictions, `exp_XXXX.npy` |
| `experiments.csv` | append-only ledger |
| `SOLUTION.md` | write-up, written before the deadline |

## Log

Notable findings, dead ends, and things worth remembering next competition.

- 2026-08-28 — project skeleton created; package `s6e7`, standalone repo.
