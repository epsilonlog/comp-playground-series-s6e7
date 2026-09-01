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

## 2026-08-31 — What the winners did differently (step-8 writeup review)

Read the 2nd, 4th, 11th, and 36th place writeups against our final (private 0.94979 vs
winning 0.95085 — a gap of ~0.0011, one resolution unit).

- **The decision rule was everyone's biggest lever, and we found it independently.**
  4th place: argmax(p/prior) moved OOF 0.89187 → 0.95063 (+0.059). 2nd place:
  Nelder-Mead multipliers, 0.88913 → 0.95074 (+0.062). Ours: +0.077. Everything else
  in every writeup fights over the remaining ~0.001. Our "fix the rule before
  laddering" carry-forward is exactly what the field's results confirm.
- **The missing ~0.001 had two named sources.** (1) *Exact-value target encoding*:
  the synthetic generator resamples numeric values from a finite support, so a
  repeated `step_count` value is a high-cardinality category; 13 cols × 3 classes =
  39 fold-fitted TE features moved XGBoost 0.94890 → 0.94956 (+0.0007). This is signal
  from the data-generating process itself — a genuinely new information source, the
  kind our stopping rule demanded. (2) *FT-Transformer* (public notebook, CV 0.95063):
  a neural model beat every GBDT; both 2nd and 4th built on it.
- **Screening trap, directly relevant to our 10% local subsamples:** exact-value TE
  read −0.0017 on a 70k-row screen but +0.0012 on full data. Per-value statistics need
  the repeats to exist. Match the cheap experiment to the idea — shrink folds or
  epochs, never the rows that carry the signal.
- **Ensemble size bought almost nothing.** 58 models + CatBoost stacker (36th) →
  0.95087 OOF; 18 models + SLSQP weights (2nd) → 0.95074; a single FT-Transformer
  family (4th) → 0.95063 and private #4. Diversity (NN + GBDT), not count, is what
  made their blends pay at all — consistent with our co-error measurement killing the
  two-GBDT blend.
- **The board itself validated CV-trust.** 4th place sat at public rank 414 and held;
  places 3–8 all display private 0.95084. On a 3-choice discrete metric the public LB
  is binomial noise at ±0.001–0.002 — a smoke test, not a steering wheel.

## 2026-08-31 — When a neural net beats GBDTs on tabular data (FT-Transformer)

- **Mechanism:** FT-Transformer (Gorishniy et al. 2021) embeds every feature — numeric
  via periodic (PLR) embeddings, categorical via learned embeddings — as a d-dim token,
  prepends a [CLS] token, and lets attention learn feature interactions that a tree must
  carve out split by split. Interactions are *learned*, not enumerated.
- **The regime where it wins:** big N (690k here — below ~100k rows GBDTs win almost by
  default), numeric-heavy features where embeddings beat splits, and a GPU. Here it beat
  the field's best GBDT 0.95063 vs 0.95016 — half the total gap between our final and
  the winner. It is a *competitor* to XGBoost only in this regime; on small or quick
  jobs the GBDT baseline still comes first.
- **The other reason to bring one: decorrelation.** Two GBDTs correlate ≈0.999 on OOF;
  this FT-Transformer vs an XGBoost of the same CV: 0.9985, disagreeing on 0.64% of
  rows. That is the independent error source our two-GBDT blend measurably lacked —
  NNs earn their place in the ensemble even when they lose head-to-head.
- **Kawamata's notebook is a model of honest measurement**, and it independently
  confirms three of our own ledger entries: training-time class weighting is an *exact*
  substitute for the post-hoc prior rule (ΔBA +0.00001 — our routes A/B convergence);
  explicit missingness flags are redundant once NaN gets its own encoded level (our
  exp_0005 null); and per-value TE screened at 70k reads −0.0017 but is +0.0012 at
  full scale — the trap our 10%-subsample local screens would walk straight into.
- **Also transferable:** fixed-epoch training instead of best-checkpoint restore (the
  restore is +0.0003 of optimism, not skill), and stop on logloss, never on a noisy
  discrete metric like balanced accuracy (−0.012).

## 2026-09-01 — The 2-SE bar breaks when you sort a table

`missing_vs_target` now reports `se_diff`, the standard error of the gap between
"rate when missing" and "rate when present": `sqrt(p_m(1-p_m)/n_m + p_p(1-p_p)/n_p)`.
The instinct is to call a gap real at 2 SE (the 95% single-test bar). But the table has
13 features x 3 classes = 39 rows and we sort by `abs_diff` — that is 39 draws and we
read the luckiest one. Chance that at least one clean-noise row exceeds 2 SE:
1 - 0.95^39 ≈ 86%. The 2-SE bar fires on nothing, almost every time.

Raising the bar fixes it: at 3 SE the false-alarm rate over 39 rows is ≈ 10%, at
3.5 SE ≈ 2%. So ~3.5 SE is the working threshold for a *sorted* table of this size —
the same logic as Bonferroni, without the formality.

Worked numbers from our own table: bmi's unhealthy gap is 0.0568 with SE ≈ 0.003 →
~18 SE, unmistakable (this is the `bmi_is_null` finding already in CLAUDE.md). A gap
of 0.004 at the same SE is 1.3 SE — indistinguishable from the luckiest of 39 coin
flips, and not worth an indicator column.

Transferable rule: the more rows a diagnostic table has and the harder you sort it,
the higher the SE bar. 2 SE is for one pre-registered comparison, not for scanning.

## 2026-09-01 — The two-Gaussian model is a ruler, not an assumption

`spread_sd` (call it d) is the gap between the extreme class means in units of the
feature's own SD. Converting d into "what one cut could score" uses two equal-width
normal curves d apart — and the confusion was reading that as a claim that the data *is*
normal. It is not. It is a reference world with a known answer, used as a measuring
unit: the same move as Cohen's d in classical statistics, and the model behind LDA.
Nothing competition-specific about it.

The derivation that makes best_split_acc = Phi(d/2) obvious: the best single threshold
is the crossing point of the two identical curves — the midpoint between the means
(nudge it either way and you misclassify more than you rescue). The midpoint is d/2 SDs
from each mean, so each class lands on its own side with probability Phi(d/2). The two
wrong-side tails, Phi(-d/2) each, are exactly the overlapping area: overlap = 2*Phi(-d/2).

Worked numbers from our table: sleep_duration d = 2.13 -> midpoint 1.06 SD from each
mean -> Phi(1.06) ≈ 0.857 — one cut gets 86% of the extreme classes. water_intake
d = 0.019 -> Phi(0.01) ≈ 0.504 — a coin flip; its *means* carry nothing (though equal
means with unequal variances would also score d = 0 and still be separable — the plot's
job, not the table's).

## 2026-09-01 — Paired comparison: why a gain smaller than the noise is still readable

"Is the improvement real?" = "is it too big for luck?", and the reference luck is
measured. One fold score wobbles ±0.002 (the binomial SE from notebooks 01/03), so
exp_0002''s +0.0009 is invisible on absolute scores. But both experiments were scored on
the same frozen folds, and most of a fold''s wobble is *which rows landed in it* — luck
that hits both runs identically. Subtracting per fold cancels it: the five differences
spread only 0.0004. The comparison never pays the fold noise; that is the entire reason
folds are frozen.

The t-statistic is the same move as reading abs_diff/se_diff in the missingness table —
the gain in units of its own SE: t = 0.0009 / (0.0004/sqrt(5)) ≈ 4.8. With 4 degrees of
freedom the bar is |t| > ~3 (fat tails, plus the many-experiments multiple-look
inflation), so 4.8 is real. Free cross-check: all five diffs positive has probability
(1/2)^5 = 1/32 under noise. Contrast exp_0006: alternating signs, t = −0.3 — noise.

Transferable chain, one idea three times: rate vs SE (01), max_bin_dev vs noise floor
(02), paired diff vs its SE (05). Nothing is "big"; things are big *relative to the
noise of the instrument that measured them* — and pairing is how you shrink the
instrument''s noise without more data.

## 2026-09-01 — The decision rule is arithmetic, not tuning

Balanced accuracy is `(1/K) · Σ hit_k / n_k`, so one more correct row of class k adds
exactly `1/(K·n_k)`. That is a **price list fixed by the data**, and the whole rule falls
out of it. Here: one `at-risk` row is worth 0.00000056, one `fit` row 0.0000084 (14.9×),
one `unhealthy` row 0.0000058 (10.3×).

- The metric will trade 15 correct majority rows for one correct `fit` row and call it
  even. Argmax refuses every such trade — it counts rows, which is right for plain
  accuracy and wrong here. That is the entire +0.077.
- Writing recall as `E[q_k · 1(d=k)] / π_k` makes BA a sum with exactly one indicator
  firing per row, so rows do not interact and BA is maximised row by row:
  `argmax_k q_k/π_k`. The multipliers **are** the price list. Nothing is fitted.
- Scale cannot change an argmax, so fixing `m[majority] = 1` leaves K−1 free numbers —
  two here, which is why the whole thing is one 2-D picture.
- Sanity identity to run on any post-processing: rows moved per class × the price list
  must reproduce the score change. Ours: −33,013 at-risk (−0.0186), +4,680 fit
  (+0.0392), +8,897 unhealthy (+0.0514) = **+0.072**, and measured 0.87742 → 0.94941.
  Raw accuracy got 19,436 rows *worse* in the process.
- Read the shape too: if the majority-class recall loss is not the biggest number in that
  table, the multipliers are doing something other than correcting the prior.

**Derived beats searched, and searching proves it.** Zero-parameter `1/π` scores 0.94932;
the 2-parameter grid search scores 0.94941. The search buys **+0.00009** — a tenth of the
resolution. Plot the whole objective surface and the reason is obvious: the set of
multipliers within one resolution of the best covers 9% of a 100×100-fold box —
`m_fit` anywhere in 6–40, `m_unhealthy` anywhere in 4–20.

**The plateau also settles a question that looked like instability.** The five per-fold
searches disagreed by 36% on `m_unhealthy` (10.2 vs 13.8). That reads as noise-chasing
until you check what it costs: swapping those two changes 0.43% of labels and 0.00006 of
score. **Parameter agreement is not the stability test; score agreement is.** A flat
objective has no unique argmax to be stable about.

The general answer to "is this overfitting or really helping?" is unchanged and cheap:
hold out the rows you tuned on. In-sample search 0.94941 vs cross-fitted 0.94931 — a
+0.0001 gap for 2 parameters against 550k rows.

Carry forward: fix the metric-correct decision rule *first*, from the price list, before
laddering anything. Our encoding (+0.0009) and capacity (+0.0039) gains largely
evaporated once the prior was corrected — they had been buying the same boundary region
the rule fixes wholesale.

## 2026-09-01 — Ask what resolution the model can see, not only where it fails

The error-profile loop (segment → hypothesise → one experiment) produced two feature
ideas here and both were honest zeros. Its blind spot: it asks **where** the model fails,
never **at what resolution** the model is allowed to see a column.

Do that arithmetic. LightGBM bins a numeric column into ≤255 bins *before* looking for a
split, then expresses it through a few dozen leaf thresholds. `sleep_duration` has 701
distinct values on a 0.01 grid → ~3 values per bin. **Structure living at single-value
resolution is destroyed at binning time and unreachable at any capacity.** More trees
cannot recover information the histogram threw away.

Testing for it needs care, because per-value rate scatter confounds three things:
binomial noise (900 rows → SE ≈ 0.016), the column's smooth trend, and real value-level
signal. The split-half replication test separates all three:

1. split rows at random; compute per-value class rates on each half
2. subtract a leave-one-out *local neighbour* baseline from each — that removes the trend
3. correlate the two halves' residuals across values

Noise does not replicate (r ≈ 0, SE ≈ `1/√n_values`). Leftover trend replicates *and* is
smooth (positive lag-1 autocorrelation). Exact-value signal replicates *and* is white.

Result here: `sleep_duration` r = 0.94 at SE 0.041 (≈23 SE) with lag-1 −0.14, and a
replicating residual sd of 0.094 — *larger than the class's own 8.4% base rate*. The raw
rows confirm it: 5.55 h → 55.3% unhealthy, 5.58 h → 16.5%, each with SE ≈ 0.012, so
22 SE apart between neighbours 0.03 h apart, and then it swings back. `water_intake` and
`heart_rate` too; `step_count`, `bmi`, `exercise_duration`, `calorie_expenditure` read
r ≈ 0 — a working null.

**A verdict is not an effect size.** Run the same test inside bands and the structure
replicates everywhere, but in the mid-band the `unhealthy` base rate is 0.0013 and the
replicating residual 0.0015 — real, and worth nothing, because there are no minority rows
there to win. The prize is the low band: base rate 0.394, residual 0.173, i.e. individual
values swinging 10%–70%. Always read the residual *against the base rate*, never the
verdict alone.

Two practical riders:
- **Coverage decides whether it is usable.** 89–99% of test rows sit on a train value
  seen ≥30 times, so a per-value encoding applies to nearly every row. Value-level signal
  with low coverage buys nothing.
- **Never screen this kind of idea on a row subsample.** Per-value statistics need the
  repeats to exist; the 11th-place writeup read −0.0017 on 70k rows and +0.0012 on full
  data — opposite signs. Shrink folds or epochs, never rows.

Mechanism worth remembering for any Playground competition: the generator resamples
numeric values from a finite support, so a repeated value behaves like a
high-cardinality **category**, not a point on a continuum.

## 2026-09-01 — What the reopened round actually taught (exp_0017/0018)

The exact-value idea won on the first try: +0.00059 at paired t = 7.2, five of five folds
positive, `cv_mean` 0.95014 — the first score above the four-way tie band and the largest
readable gain since the prior correction. Three lessons, none of them about the score.

**Spend an experiment testing your screen, not only your idea.** exp_0018 encoded all
seven numeric columns instead of the three the replication test kept: +0.00003 at t = 0.4.
Twelve extra columns, nothing bought. That run produced no score and was the more
valuable of the two, because it converted "the replication test looks sensible" into "the
replication test picks the right 3 of 7 columns" — a two-minute diagnostic that now
replaces four training runs, here and in every later competition.

**A target encoder has two leaks, not one, and one diagnostic for both.**

1. *Across the fold* — fit on the fold's training rows only. Everyone remembers this one.
2. *Within the training rows* — a row's own label is inside its value's mean. Worst on
   rare values, where the mean **is** the label. Closed by building the training matrix
   from an inner K-fold, so no row is encoded by a statistic it helped compute.

The check that catches either: the **fit-vs-val gap**, not the CV score. A leaking
encoder makes the training score climb away from validation. Ours: +0.00177 against the
un-encoded parent's +0.00154. Unchanged — which is what "no leak" looks like.

**The pre-registered selection rule turned out to measure with the blunter instrument.**
It reads: highest `cv_mean`, ties within 0.001 break toward fewer moving parts. But 0.001
is the resolution of a *single unpaired* `cv_mean` — the wobble when nothing changed. A
paired comparison on frozen folds cancels that fold-luck and resolves several times
finer. So a result at t = 7.2 lands *inside* the rule's tie band and gets tie-broken away.
Rewriting the rule after seeing the result is the anti-pattern the pre-registration
exists to prevent, so the defect stays visible and the fix is carried forward: **write
the next rule in paired terms** — promote a candidate when its paired diff against the
incumbent clears |t| > 3, break genuine ties by moving parts.

And the sentence to keep about stopping: **stop when you are out of information sources,
not when you are out of ideas.** exp_0017 was correctly declined in August — nobody had
measured that the signal existed, so its outcome was unreadable in advance. One
measurement later it was readable, and it won immediately. Ideas are unlimited and mostly
worthless; sources are countable.
