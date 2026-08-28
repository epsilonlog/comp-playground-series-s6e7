# LEARNING.md

Concepts understood while working this competition. Concepts and reasoning only — no
code, no config. If it's in the repo, it doesn't belong here.

Append new sections at the end. Consolidate periodically: merge overlapping entries,
keep the worked numbers, drop the scaffolding.

| Section | The idea in one line |
|---|---|
| [The metric](#2026-08-28--the-metric-balanced-accuracy) | macro-recall makes a rare row worth 14 ordinary ones, and its floor is 1/K |
| [Probabilities → labels](#2026-08-28--from-probabilities-to-labels) | you train on a surrogate and score on the real metric; the gap is free score |
| [Validation](#2026-08-28--validation-oof-and-the-precision-of-a-cv-score) | a CV score is a measurement with a resolution — know it before you chase gains |
| [Trees](#2026-08-28--what-trees-can-and-cannot-do) | what a split can express, and the three things it can never find |
| [Reading data honestly](#2026-08-28--reading-data-honestly) | summary statistics that lie, and the five outputs of EDA |
| [Technique](#2026-08-28--technique) | small mechanics worth not re-deriving |

---

## 2026-08-28 — The metric: balanced accuracy

The macro-average of per-class **recall**. Build the confusion matrix, divide each
diagonal cell by its row sum, average across classes. Each class contributes 1/K
regardless of size.

Macro is the odd one out: micro-averaged recall and weighted-average recall both *equal*
plain accuracy. Only macro discards class sizes.

    classes 70k / 25k / 5k, predict the majority for everything
    recalls  1.00, 0.00, 0.00   →  balanced accuracy 0.333
                                →  plain accuracy    0.700

**The floor is 1/K, not 0.5.** Always-one-class and random guessing both score 1/K. The
bottom third of the raw scale carries no information, which is what `adjusted=True`
rescales away: `(score − 1/K) / (1 − 1/K)`, chance → 0. Kaggle scores unadjusted, so keep
the conversion as a reality check — **a raw 0.50 at K=3 is an adjusted 0.25**, only a
quarter of the way from chance to perfect.

**Row value — the number everything else follows from.** One extra correct row in class
*k* is worth `1/(K·n_k)`. At 70k/25k/5k a rare-class row is worth **14×** a majority-class
row. Same statement as "each sample is weighted by the inverse prevalence of its true
class".

**Why reimplement it when sklearn has it.** sklearn stays the reference and the test
asserts agreement. But: writing it is how you learn what you're scored on; if a class is
absent from `y_true`, sklearn silently drops it and divides by K−1, which is a *different
metric* than the host computes and would distort one fold; the decision-rule search calls
it hundreds of times over ~690k rows; and it gives one call site for every scorer.

---

## 2026-08-28 — From probabilities to labels

### You cannot train on balanced accuracy. Nobody can.

**Gradient boosting, mechanically.** For K classes the model keeps K raw scores per row,
softmaxed into probabilities. The loss is log-loss, `L = −log q_y(x)`. Each iteration fits
one tree per class to the negative gradient, which for softmax + cross-entropy collapses
to:

    ∂L/∂F_k  =  q_k − 1[y = k]

The tree fits *(is this the true class?) − (probability we gave it)*. True class B at
`q_B = 0.30` → residual 0.70 → push `F_B` up. Nothing in that loop computes an argmax or
knows what the competition metric is.

Balanced accuracy depends only on the argmax: nudge a probability from 0.51 to 0.52 and
nothing moves; nudge it past a flip and it jumps. Derivative zero where it exists,
undefined at the jumps. So every non-differentiable metric forces the same two-stage
shape — train on a differentiable **surrogate** (log-loss), then evaluate and tune on the
**real metric**. The gap between them is free score.

    model → Q (n_rows, K) floats → decision rule → y_pred (n_rows,) → metric → score

The metric cannot distinguish `(0.51, 0.49, 0.00)` from `(0.99, 0.01, 0.00)`; both argmax
to the same class. All confidence information is discarded before it is called.

### Argmax is a choice, not a law

It is provably optimal *for plain accuracy* — the default everywhere because accuracy is
the unstated default. It's wrong here: the model learned `p(k|x)` under the training
prior, and balanced accuracy declares that prior worth nothing. Decision theory says
maximise `p(k|x)/π_k`:

    p = (0.55, 0.30, 0.15),  π = (0.70, 0.25, 0.05)
      →  (0.79,  1.20,  3.00)   → predict class 3, not class 1

Two routes, and which wins is an experiment: **weight during training**
(`sample_weight ∝ 1/n_class`, so argmax becomes correct — wins when a class is so rare an
unweighted model never learns it), or **adjust after training** (free, instant, applies to
any trained model, keeps probabilities undistorted — usually the better start).

### Calibration, and why the derived answer isn't the final answer

A model is calibrated when its stated probabilities match observed frequencies: of rows
where it said 0.70, about 70% really are that class. Long boosting runs overshoot toward
0/1; early stopping and regularisation undershoot; rare classes are systematically
under-predicted; class weighting distorts deliberately.

That breaks divide-by-prior, because the rule assumes `p` is the *true* posterior:

    true posterior:       0.15 / 0.05 = 3.00  → predict it     ✓
    model under-predicts: 0.08 / 0.05 = 1.60  → predict other  ✗

**So stop trusting the derivation and let the data pick the number.** Treat the divisor as
a free parameter: predict `argmax_k m_k · q(k|x)`, choose `m` by maximising the metric
directly on OOF. If the model is calibrated the optimiser lands near `1/π` anyway; if not,
it absorbs prior correction *and* calibration error in one step.

    6 rows, class counts 3 / 2 / 1
    plain argmax      m = (1, 1,   1  )   →  0.500
    divide by prior   m = (1, 1.5, 3  )   →  0.778
    searched on OOF   m = (1, 1.2, 2.5)   →  0.889

`1/π` captures most of the gain — the prior intuition is sound. The searched rule
*sacrifices* a correct majority row to win two minority ones: a bad trade under accuracy,
a great one here, and it finds that trade without being told.

**Who searches?** Not the model. A small loop in our own code, after CV, over 2 free
parameters (scaling all K multipliers by a constant changes no argmax, so fix one at 1).
No gradients. A multiplier can only undo a per-class *multiplicative* distortion — a model
overconfident at high probabilities but accurate at low ones needs explicit calibration
(Platt / isotonic, fitted inside the fold). Check the reliability curve first.

---

## 2026-08-28 — Validation: OOF, and the precision of a CV score

### OOF is a vector, not a number

Each fold is predicted by a model trained on the other folds. Stack the blocks back in row
order: **one prediction per training row, each from a model that never saw that row** —
same length as `y_train`, aligned index-for-index.

A CV score of 0.612 is a scalar; you can only compare it to another scalar. The OOF vector
is a prediction for every row paired with its true label — a dataset you can optimise
against: tune the decision rule, search blend weights, stack base models as meta-features,
find which slice the error concentrates in. None of that is possible from a scalar.

- **Save probabilities, not labels.** Trying a new `m` requires `Q`. From stored labels
  there is nothing left to search over — the argmax destroyed it.
- **Leakage contaminates OOF silently.** Fit a scaler, imputer or target encoder on the
  full training set before splitting and fold *k*'s "unseen" rows influenced the model
  that predicts them. The vector still looks fine; everything tuned on it is tuned against
  a lie, and it only surfaces on the leaderboard.
- **Score fold-wise, not whole-vector** — that is what gives `cv_std` alongside `cv_mean`.
- OOF is **not** your test predictions; those average the K fold-models on test.

### A CV score is a measurement, and measurements wobble

Take one row. The model gets it right or not: `X` is 1 with probability `p`, else 0. Then
`E[X] = p`, and since squaring does nothing to a 0/1 variable,
`Var(X) = E[X²] − E[X]² = p − p² = p(1−p)`. A recall is the *average* of n such rows.
Variances add for independent things, so the sum has variance `n·p(1−p)`; dividing a
variable by n divides its variance by n², so the average has `p(1−p)/n`. Root it:

    SE = √( p(1−p) / n )

Nothing in there but "variances add" and "the divisor gets squared". What it buys is the
**√n law** — precision costs quadratically:

    n =    100  →  ±0.046      n = 10,000  →  ±0.0046      100× the rows, 10× the precision

**The `p` barely matters, so it can be guessed — or dropped.** `p(1−p)` is flat across the
plausible range (0.250 at p=0.5, 0.210 at 0.7, 0.090 at 0.9) and is *maximised* at p=0.5,
so assuming p=0.5 everywhere gives a bound that needs no assumption at all.

### Macro metrics inherit the precision of the smallest class

Applied to this competition's 5-fold validation sizes — only *n* differs between the rows:

    fit         √(0.70·0.30 /   7,961)  =  0.0051      ±0.5 percentage points
    unhealthy   √(0.60·0.40 /  11,545)  =  0.0046
    at-risk     √(0.90·0.10 / 118,512)  =  0.0009      ±0.09 pp — 15× steadier

Balanced accuracy averages those, so `SE = ⅓·√(Σ SE²)`. Read the terms inside the root:

    0.0051² = 26 units      0.0046² = 21 units      0.0009² = 0.8 units

**`at-risk` is 86% of the data and contributes 1.5% of the wobble.** 690,088 rows, but the
*effective* sample size for this metric is about 40,000. Per fold, SE(BA) ≈ 0.0023 — and
the assumption-free worst case (all p = 0.5) is 0.0025, so the guessed recalls were
load-bearing for nothing.

### Resolution — and why that is not a second formula

`cv_mean` averages 5 folds which jointly cover every row exactly once, so it is near
enough the metric computed on the whole training set. Apply the *same* formula with the
full per-class counts:

    fit     √(0.21/ 39,803) = 0.00230      unhealthy √(0.24/ 57,724) = 0.00204
    at-risk √(0.09/592,561) = 0.00039   →  ⅓·√(Σ) = 0.00103

and `0.0023/√5 = 0.00103` — identical, because 5× the rows shrinks SE by √5. One formula
at two scales.

**So the scale reads to about ±0.001**, and that converts into three things:

1. A **falsifiable prediction** — `cv_std` should come back ≈ 0.002. If the first run
   reports 0.006, go digging; without the prediction, 0.006 is a number you have no
   opinion about.
2. A **decision threshold** — +0.0008 is the instrument rattling, +0.0045 is a result.
   Written down once instead of relitigated fifteen times.
3. A **budget filter** — a technique worth ~0.0005 cannot be confirmed by this harness.
   Know that before spending the week.

**What the calculation omits.** The five fold-models are different models, each trained on
a different 80%. `cv_mean` measures a **procedure**, not a fixed model, and the arithmetic
prices only *validation* sampling noise — which rows landed in the held-out set. It
ignores training variability entirely. So **0.001 is a lower bound on the total wobble**;
expect observed `cv_std` at or somewhat above 0.002, and treat two-or-three-times-higher
as the signal to investigate. Repeated CV exists to price the component this omits —
worth paying for only when partition noise is not already dominated, which at 690k rows
it is.

### Absolute error bar ≠ comparison error bar — this is why folds are frozen

±0.001 applies to one score standing alone. Two models scored on the **identical** folds
see the same rows in the same validation sets, so "these particular 7,961 people were
easy" is the same luck for both and **cancels out of the difference**. A paired comparison
on frozen folds resolves smaller differences than either absolute score can. Regenerating
folds discards that cancellation and leaves two independently wobbling numbers — the real
cost of the anti-pattern, quite apart from ledger comparability.

### Stratification is variance reduction, not correctness

Random dealing gives each fold *approximately* 20% of each class; stratifying makes it
exact, removing a noise source for free. The same logic argues for finer keys (target × a
gating feature) — and the same logic prices them. Scattering 624 oddball rows into 5 piles
varies each pile by about ±10 rows; even assuming those rows are always wrong while
ordinary rows are wrong 30% of the time, that is 7 errors in 7,961 → 0.0009 on one recall
→ **0.0003** on the macro average. Real, and an order below the resolution.

**Size an effect before paying permanent complexity for it.** A frozen compound key is
forever.

---

## 2026-08-28 — What trees can and cannot do

### Monotone transformations are invisible

A split asks "is x ≤ 7.2?" — a question about **rank order**. Log, sqrt, Box-Cox and every
other monotone transform preserve rank order, so they produce the identical split, the
identical tree, and identical predictions. Log-transforming a feature for a GBDT is a
no-op.

Transforms exist for models that care about *distance* and *linearity*: a linear model
seeing a curved relationship as straight, a squared loss being dominated by extreme
values, a distance metric hijacked by scale. GBDTs have none of those properties. So on a
GBDT-first plan the skew question is moot before you measure it; it returns only when a
linear model or a network joins the blend — and then it is usually standardisation that is
wanted, not a log.

**Clipping is a different question and does matter.** A clipped column piles values at a
wall — income capped at 100,000 with 3% of rows exactly there. That spike is real
structure a tree will happily learn, and usually deserves an indicator or an explicit
censoring treatment. Distinguish it from a *truncated* sampling range, where density at
the bound is genuinely low and almost nothing piles up: **a clip shows percent-at-bound in
whole numbers, a truncation in hundredths.**

**And distinguish a real zero from a coded missing.** A spike at exactly zero means "not
applicable" mixed into a continuous column — two populations stacked. If the column also
carries explicit nulls, the zero is genuine; if it has none, suspect the zero *is* the
missing code, and converting it changes every downstream statistic.

### Encoding ordered categories: the rule is contiguity, not monotonicity

`low < medium < high` is a fact about **words**. Encoding it `0, 1, 2` is an empirical
claim about the **target**, and the words can be ordered while the claim is false.

With three levels the only bipartitions a threshold offers are `{low} | {medium, high}`
and `{low, medium} | {high}`. **`medium` cannot be isolated in one split.** The encoding is
a *constraint* restricting the model to contiguous groupings in your chosen order.

I first read this as a monotonicity requirement and was wrong. Real data showed
`stress_level` strongly non-monotone in the majority class (0.80 → 0.99 → 0.72, peaking in
the middle), which by a monotonicity rule should favour one-hot. It doesn't: the
*informative* levels were the two extremes, and threshold splits isolate extremes
perfectly. Only the middle level was unreachable, and the middle level was the boring one.

Ask **"are the levels I need to isolate contiguous in this ordering?"** Monotonicity is
sufficient for that, not necessary. A U-shape with interesting ends is fine; an
interesting *middle* is not.

At 3 levels the stakes are small — one-hot costs 2 columns, ordinal at most one extra
split. It becomes decisive at high cardinality, and GBDTs with native categorical handling
sidestep the question entirely by finding a bipartition from gradient statistics with no
ordering at all.

**Where verifying monotonicity does pay: monotone constraints.** Forcing a feature's
effect to be monotone eliminates a class of overfitting where the model learns a wiggle
that is really noise. That needs both an ordinal encoding *and* evidence the relationship
is monotone — and there, unlike in encoding, monotonicity really is the requirement.

### Interactions: three kinds that are not free

"Don't hand-craft interactions, the trees find them" is true in one sense and false in
three. A root-to-leaf path is a conjunction of conditions, so a depth-*d* tree expresses
*d*-way **axis-aligned conjunctive** interactions. Those really are free. What must still
be engineered:

1. **Arithmetic combinations** — `a/b`, `a−b`. A tree approximates a diagonal boundary
   with a staircase: many splits, poorly, and only with enough data. Hand it the ratio and
   it becomes one split. The biggest routine win.
2. **Interactions among individually-weak features.** Boosting is *greedy* — it takes the
   best immediate gain. If A alone has no gain and B alone has no gain but A×B is strongly
   predictive, neither is ever selected and the interaction is never found. XOR is the
   extreme: representable at depth 2, unreachable by greedy fitting.
3. **Cross-row aggregations** — mean target per user, count per store, days since the
   previous event. A tree cannot aggregate across rows at all. In real competitions this
   is where most feature-engineering value lives.

Plus a depth budget: a 4-way interaction needs depth ≥ 4, and typical defaults sit near 5.
Beyond that it is structurally unavailable, not merely hard to find.

So "the trees will find it" is safe when every feature is axis-aligned, individually
informative, and there are no entities to aggregate over. **Say which of those conditions
holds before relying on it.**

### Unseen categorical levels

A level present at predict time but absent when the encoder was fitted. Every encoder
fails differently and none fail well: ordinal raises or substitutes a fill; one-hot raises
or emits an all-zeros row — a pattern the model never saw in training; target encoding has
no statistic and falls back to the global mean.

Two distinct versions: **train vs test**, and — subtler — **inside CV**, where a level
appears in the validation fold but not the training folds. Low cardinality makes both
impossible; thousands of rare categories make the second one routine. Matching level sets
means none of that handling has to be written, tested and kept consistent across every
fold and every encoder — it removes a category of code and a category of bug. **The safety
is a property of the data, never of the pipeline.**

---

## 2026-08-28 — Reading data honestly

### Summary statistics that lie

**Any summary defined as a min or max describes your worst data point, not your data.** I
measured a quantisation grid as the minimum gap between adjacent distinct values. Two
clean columns came back ragged — a 0.1 grid reported as 0.03, a 0.01 grid as 0.001. Each
had exactly *one* off-grid value, splitting a normal step into two smaller ones. One row
in 690,000 moved the statistic tenfold. The median was exactly right (526 of 536 gaps were
precisely 0.1), and the tell was a `0.07 + 0.03` pair summing to one normal step.

Use a quantile to describe the bulk. If the extreme is interesting, give it its own column
so contamination stays *visible* instead of corrupting the number you meant to read.

**Bound checks miss a spike in the middle.** Percent-at-min only catches zero-inflation if
zero happens to be the minimum. Share-of-the-modal-value catches a point mass — sentinel,
default, or inflated zero — wherever it sits.

**Equal counts are not evidence of a shared mask.** Two columns showed exactly 6,901 nulls
in train and 2,958 in test; I inferred a shared missingness mask. Wrong — it was
arithmetic: 690,088 × 1% = 6,900.88 → 6,901. A fixed proportion per column, rounded, drawn
independently. Only the *joint* count distinguishes the two stories: compare observed
co-missingness against `n · p_a · p_b`, never counts against each other.

### Standardised effect size, and its three blind spots

Raw gaps between class means aren't comparable across columns — one is in steps, another
in hours. Divide each gap by its own column's standard deviation and both become rankable.
Read as overlap between two bell curves:

    2.1  →  ~29% overlap    nearly separable on that feature alone
    0.9  →  ~65% overlap
    0.05 →  ~98% overlap    the two curves are the same curve

The same quantity read as an **overlapping coefficient**: model each class as a normal
with equal σ, means `d·σ` apart. Put them at ±d/2 so the curves cross at 0; by symmetry
each contributes its own tail beyond the crossing:

    OVL            = 2 · Φ(−d/2)
    best_split_acc = Φ(d/2) = 1 − OVL/2        with  Φ(z) = ½(1 + erf(z/√2))

d = 2.1 gives 29% overlap and 0.86; d = 0.05 gives 98% and 0.51, a coin flip. Assumes
normality, equal variance, one threshold, two classes — and **the equal-priors assumption
is a feature here, not a flaw**: accuracy under equal priors *is* balanced accuracy for
two classes, so `best_split_acc` already reads in the competition metric's units, no
correction needed.

The blind spots, all three of which bit:

- Dividing by the *overall* std understates separation (the pooled within-class std is
  correct, so the estimate is conservative).
- It compares means only — equal means with different variances score zero and are still
  separable.
- With more than two classes it reads only the extremes and is blind to the middle. Two
  features scored equally at ~0.82; one separated all three classes, the other put two on
  top of each other and only found the third. Under a macro metric those are not equally
  useful.

**The ranking says where to look; the raw per-class means say what you found.**

### Correlation vs shared variance

`r` reads far larger than it is; `r²` is the fraction of variance one variable linearly
explains in the other.

    r = 0.3 →  9%      r = 0.7  → 49%   ← only here is half the variance shared
    r = 0.5 → 25%      r = 0.95 → 90%

That is why the redundancy threshold sits near 0.95, not at a number that merely *sounds*
high. Two caveats: with Spearman it is shared variance of the *ranks*; and `r² = 0` means
no monotone association, not independence — `y = x²` on symmetric `x` has `r ≈ 0` and is
perfectly determined.

### What EDA is actually for

Feature engineering is one of five outputs and often the smallest:

1. **Validation design.** The class balance decides the fold scheme; get it wrong and
   every number afterwards is fiction.
2. **Floor and plausible ceiling.** Knowing chance is 1/K, and that the best single
   feature could reach ~0.86 on the extreme classes, is what lets you tell a broken
   pipeline from a hard problem when the first baseline lands.
3. **Feature engineering.** The engineered columns themselves.
4. **Where error will concentrate.** A class with fewer features pointing at it will be
   the recall bottleneck. Knowing that before the first run tells you where to look.
5. **Work avoided.** The negative results — no transforms needed, missingness carries no
   signal, category sets match — feel like nothing happened and are usually the largest
   time saving in the whole exercise.

**Most EDA findings are negative. That is the point, not a disappointment.**

---

## 2026-08-28 — Technique

**Counting pairs without a loop.** A confusion matrix counts how often each
**(true, predicted)** pair occurs. Counting utilities count single integers, not tuples —
so squash the pair into one integer with base-*n* positional notation:

    flat = true_index * n_classes + predicted_index

With 2 classes: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3. Unique, no collisions — the same
arithmetic as "row *r*, column *c* of a grid *w* wide is cell `r·w + c`". Count the flat
integers, reshape to *n × n*, and the matrix falls out with no Python loop. A Python loop
over 690k rows is about a second; the decision-rule search calls this hundreds of times.
Generalises to any counting of combinations of small-cardinality integers.
