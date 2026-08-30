# LEARNING.md

Concepts worth carrying to the next competition. No code, no config. Kept deliberately
short: worked numbers stay, scaffolding goes.

| Section | One line |
|---|---|
| [The metric](#the-metric--balanced-accuracy) | macro-recall: a rare row is worth 14 ordinary ones, and the floor is 1/K |
| [Probabilities → labels](#probabilities--labels) | train on log-loss, tune the decision rule on OOF — the gap is free score |
| [A CV score is a measurement](#a-cv-score-is-a-measurement) | know the resolution (~0.001 here) before chasing gains |
| [Frozen folds](#frozen-folds) | paired comparisons cancel shared luck; deterministic is not frozen |
| [Trees](#what-trees-can-and-cannot-do) | a split sees rank order only; three kinds of interaction are never free |
| [Reading data honestly](#reading-data-honestly) | min/max statistics lie; only the joint distinguishes co-missingness |
| [Finding a shift](#finding-a-shift) | marginals cannot see a joint shift; control every positive result |
| [Acting on a shift](#acting-on-a-shift) | price level vs ranking against the resolution — usually change nothing |
| [Decision beats model](#the-decision-dominates-the-model) | matching the decision to the metric was worth 20× everything else |
| [The ladder in action](#the-paired-ladder-in-action) | +0.0009 proved real at t=4.8; a perfect zero with a mechanism |
| [When to stop](#when-to-stop) | decline experiments you couldn't read even if they won |

---

## The metric — balanced accuracy

- Macro-average of per-class recall: confusion matrix, diagonal over row sums, mean.
  Micro-averaged and weighted recall both equal plain accuracy; only macro drops size.
- At 70k/25k/5k, predicting the majority everywhere: BA 0.333 vs accuracy 0.700. The
  floor is **1/K, not 0.5** — a raw 0.50 at K=3 is an adjusted 0.25, a quarter of the way
  from chance to perfect.
- One correct row in class k is worth `1/(K·n_k)`: a rare-class row is worth **14×** a
  majority row at 70/25/5.
- Reimplemented because sklearn silently drops classes absent from `y_true` (a different
  metric), and the decision-rule search calls it hundreds of times over 690k rows.

## Probabilities → labels

- BA depends only on the argmax — a step function, gradient zero or undefined — so nobody
  trains on it. GBDTs train log-loss (per-class residual `q_k − 1[y=k]`); the real metric
  applies after: model → probabilities → **decision rule** → labels → score.
- Argmax is optimal for plain *accuracy* only. Under BA, maximise `p_k/π_k`:
  p = (0.55, 0.30, 0.15), π = (0.70, 0.25, 0.05) → (0.79, 1.20, 3.00) → predict class 3.
- Calibration breaks the derivation (a rare class under-predicted at 0.08 gives
  0.08/0.05 = 1.6 and loses, where the true 0.15/0.05 = 3.0 wins). So treat the divisors
  as free parameters and **search multipliers on OOF**: toy with counts 3/2/1 — argmax
  0.500, divide-by-prior 0.778, searched 0.889. Only 2 free parameters at K=3 (scaling all
  multipliers changes no argmax). A multiplier fixes only multiplicative distortion —
  check the reliability curve first.
- **Save OOF probabilities, never labels** — the argmax destroys what the search needs.

## A CV score is a measurement

- A 0/1 outcome has variance `p(1−p)`; an average of n of them wobbles by
  `SE = √(p(1−p)/n)`. The √n law: 100× the rows buys 10× the precision. `p(1−p)` is flat
  near its 0.25 max, so p = 0.5 gives an assumption-free bound.
- A macro metric inherits the smallest class: `fit` √(0.21/7,961) = 0.0051 per fold vs
  `at-risk` 0.0009 — 86% of the data contributes 1.5% of the wobble. Effective sample size
  ≈ 40k of 690k. Per fold SE(BA) ≈ 0.0023; on `cv_mean`, 0.0023/√5 ≈ **0.001**.
- That number converts into three things: a falsifiable prediction (`cv_std` ≈ 0.002 —
  0.006 means go digging), a decision threshold (+0.0008 is the instrument rattling,
  +0.0045 is a result), and a budget filter (a 0.0005 technique is unmeasurable here).
- It is a lower bound — it prices validation sampling only, not training variability.
  Repeated CV prices the rest; at 690k rows, not needed.
- **SD vs SE:** SD spreads individuals — for 0/1 data it is √(p(1−p)), fully determined by
  the mean (28pp at p = 8.5%, says nothing). SE is the wobble of the estimate. Compare
  group rates in combined SEs: 8.469 ± 0.047 vs 8.892 ± 0.26 → 1.6 SE → noise;
  8.47 vs 2.79 → ~28 SE → real. Within ~2 combined SEs, same rate.

## Frozen folds

- Two models scored on **identical** validation rows share their luck ("these 7,961 `fit`
  people were easy") and it cancels out of the difference — a paired comparison resolves
  smaller gaps than either ±0.001 absolute score. Regenerating folds discards the
  cancellation and voids every prior ledger row.
- **Deterministic ≠ frozen.** A seeded splitter reproduces the partition only while seed,
  fold count, library internals, and source row order all hold — none pinned by the code.
  A re-download in a different order changes the partition silently. Persist the
  assignment; verify by *re-deriving* it (a stored config only confirms what the writer
  claimed).
- Stratification is variance reduction, not correctness — and never shift correction: it
  makes folds resemble *each other*, not test. Price finer keys before freezing them:
  target × null-bucket was worth 0.00004 against a 0.002 fold SE.

## What trees can and cannot do

- A split asks a **rank-order** question, so monotone transforms (log, sqrt, Box-Cox) are
  no-ops for a GBDT. Transforms exist for models with distance or linearity assumptions.
- Clip vs truncate: a clip piles percent-at-bound in **whole numbers** (real structure —
  consider an indicator); a truncation in hundredths. A zero-spike in a column with no
  explicit nulls: suspect zero *is* the missing code.
- Ordinal encoding is a **contiguity** constraint, not monotonicity: thresholds isolate
  extremes fine (a U-shape is fine); only an interesting *middle* level is unreachable.
  `stress_level` peaked in the middle and ordinal was still right — the informative levels
  were the two ends.
- Interactions that are never free: (1) **arithmetic** (`a/b` — a diagonal becomes a
  staircase; hand over the ratio), (2) **weak×weak pairs** — greedy boosting never picks
  either (XOR: representable at depth 2, unreachable by greedy fitting),
  (3) **cross-row aggregations** — trees cannot aggregate over rows at all, and this is
  where most FE value lives. Plus a depth budget: a 4-way interaction needs depth ≥ 4.
- Unseen categorical levels fail differently in every encoder. Matching level sets remove
  the whole problem — and that safety is a property of the **data**, never the pipeline.

## Reading data honestly

- Any min/max-defined summary describes your **worst data point**: one off-grid row in
  690k moved a grid estimate tenfold (the median was exactly right — 526/536 gaps at
  precisely 0.1). Use quantiles; give the extreme its own column so contamination stays
  visible.
- Modal share catches a point mass anywhere; percent-at-min only catches it at the min.
- Equal null counts are **not** a shared mask: 690,088 × 1% = 6,901 in both columns,
  drawn independently. Only the joint count (observed co-missing vs `n·p_a·p_b`)
  distinguishes the stories.
- Standardised effect size d = gap/SD, read as overlap: d = 2.1 → 29% overlap,
  best-single-split accuracy Φ(d/2) = 0.86; d = 0.05 → coin flip. Blind spots: compares
  means only (equal-mean different-variance scores zero), and at K>2 reads only the
  extremes — two features tied at 0.82 where one separated three classes and the other
  two.
- Report r², not r: 0.7 → 49% shared variance, 0.95 → 90% — which is why the redundancy
  cut sits at 0.95, not a number that merely sounds high.
- EDA has five outputs: validation design, floor and ceiling, features, where error will
  concentrate, and **work avoided**. Most findings are negative; that is the point.

## Finding a shift

- Adversarial validation: relabel rows by *file*, classify. Here all 13 marginals were at
  chance (solo AUCs 0.499–0.522; null rates identical to five decimals) yet the joint gave
  **0.6518**. You can plot 13 marginals; you cannot plot a 13-dimensional joint — AV is
  the only search that scales past a few dimensions.
- The shift was **nulls per row**: identical mean (0.6514) in both files, test variance
  +32% — train draws nulls independently per column, test clumps them onto rows. Most
  plausibly an artifact of how the synthetic files were generated; unverifiable, and it
  doesn't matter — the design must price it either way.
- AV feature importances **lie**: gain ranked `water_intake` (solo AUC 0.499) first and
  `gender` (0.522, the most shifted) eighth — high-cardinality split bias. Plots say
  *what* differs; the AUC says *whether* anything does.
- Two controls, always: shuffled labels (returned 0.5002, so the harness reads chance as
  chance) and the independence baseline for joint counts.
- Exclude `id`: per-file contiguous ranges give AUC 1.0 that means nothing.

## Acting on a shift

- A shift is a question, not an answer. Two possible harms: **level** (CV–LB offset —
  hits every model identically, cancels from every comparison) and **ranking** (test
  reweights a region where models disagree — the only harm worth paying for).
- The mixture framing prices everything: CV = 0.978·light + 0.022·heavy,
  test = 0.954·light + 0.046·heavy. Level move ≤ 0.024 × 0.10 ≈ 0.002. A comparison moves
  by 0.024·δ, so δ > **0.042** on 2% of rows before the 0.001 resolution is threatened —
  plausible only for missing-data handling, so carry the slice key and watch that recall.
- Two checks that decompose an AUC: **strip the suspected channel and re-run** (complete
  cases: 0.6518 → 0.5304 against a 0.4999 control), and **ask whether the shifted
  quantity predicts the target**. Here the null count is flat once you condition on
  `bmi_is_null`, and that indicator's rate is identical in both files (2.014%). *The
  quantity that shifted predicts nothing; the quantity that predicts didn't shift.*
- Misread to remember: "more nulls → unhealthy" was backwards — rows *missing* `bmi` are
  **less** often unhealthy (2.79% vs 8.47%). An aggregate trend can be one indicator
  dragging the average; condition on it before believing a count. And nulls **can** be
  features — the indicator transfers precisely because its rate didn't shift.
- Write the falsifiable consequence before submitting: test is information-poorer, so
  **LB should land 0.001–0.002 below CV**. A 0.02 gap means a different problem entirely.

## The decision dominates the model

- One change — `class_weight="balanced"` on the untouched baseline — was worth **+0.077**.
  Every modelling refinement combined (encoding, capacity, features, a second family,
  blending) was worth +0.005. The confusion matrix showed it on day one: errors flowing
  into the majority class is argmax maximising the *wrong* metric.
- **Two routes, one correction, same answer.** Weights during training scored 0.94945;
  multipliers searched on OOF scored 0.94931 — 0.00014 apart, and the searched values
  landed on 1/prior because the probabilities were calibrated. Pick one route, never
  both: composing them double-corrects.
- **Argmax gains can be a mirage.** Weights-on-defaults (0.94956) tied weights-on-combo
  (0.94945): the capacity and encoding gains, real at argmax, vanished after correction —
  they were repairing the same boundary region the correction fixes wholesale. Next
  competition: fix the metric-correct decision (or evaluate under it) *before* running
  the ladder, or the ladder optimises differences that will not survive.

## The paired ladder in action

- Native cats gained +0.00087 — invisible against the ±0.002 absolute fold noise, yet
  **all five paired fold diffs were positive** (sd 0.0004, t = 4.8): real. The ratios
  experiment read t = −0.3 with alternating signs: noise. Rule of thumb at 4 degrees of
  freedom: |t| > 3 real, |t| < 2 noise, between: run nothing that depends on it.
- **A negative with a mechanism is a result.** The missingness indicators moved
  probabilities by up to 0.016 and flipped essentially zero decisions — NaN routing
  already encodes what the flags say. Crossed off with evidence; never retried.
- Winners compose in their own experiment, never by assumption. Here the two gains
  predicted +0.0048 together and delivered +0.0045 — additive, this time. Checked.
- The fit-vs-val gap is the under/overfit dial: capacity widened it 0.005 → 0.016 while
  validation still rose +0.0039 — underfitting, keep going. Validation stalling while the
  gap grows is the signal to regularise; it never fired here.
- Anything *searched* on OOF (rules, thresholds, blend weights) gets the same honesty as
  a model: report the cross-fitted score, and read in-sample minus cross-fitted as its
  overfit. Two parameters on 550k rows: gap ≈ 0.

## When to stop

- Stopping has criteria, not feelings: (1) every axis probed once and priced; (2) the
  trajectory flat inside the resolution band — the last four candidates spanned 0.00025
  against a 0.001 resolution; (3) the next experiments are **unreadable even if they
  win** — their best case lands inside the tie band, so they could not be selected;
  decline them (rule-on-xgb was deliberately never run); (4) the error profile shows the
  rest is irreducible — errors live where no feature separates the classes; (5) CV↔LB
  calibrated, so the final submission is a point on a known line, not a lottery ticket.
- Ensemble appetite is a measurement: blending pays in proportion to error
  *decorrelation*. Two GBDT families were wrong together an order of magnitude more
  often than independence predicts, so the blend landed between its parents, not above
  them. Check co-error against `p_a · p_b` before buying models.
- The tie-break is part of the selection rule, written before looking: within one
  resolution of the top, take the fewest moving parts. Twelve experiments ended at
  *baseline + one parameter* — and only the ladder makes that sentence a measurement
  instead of a guess.
