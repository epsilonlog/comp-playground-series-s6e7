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
| [Joint shift](#2026-08-28--marginals-cannot-see-a-joint-shift) | 13 identical marginals, one clustered joint — why adversarial validation exists |
| [Acting on a shift](#2026-08-28--a-shift-is-only-a-fold-problem-if-it-moves-the-ranking) | finding one is a question, not an answer — price the correction against the resolution |
| [SD vs SE](#2026-08-30--sd-describes-individuals-se-describes-the-estimate) | SD spreads individuals, SE wobbles the average — compare group rates in SEs |
| [The bmi misread](#2026-08-30--misread-which-way-the-bmi-null-effect-points) | missing bmi means *less* unhealthy, and the count was never the carrier |

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

**Deterministic is not the same as frozen, and the gap is the whole point.** A seeded
splitter over unchanged data reproduces the identical partition every run, so writing it
to a file looks redundant. It isn't: determinism is conditional on inputs that are not
themselves pinned. Change the seed, change the fold count, add a row, or re-download the
source in a different row order, and you get a *different* partition with no error and no
warning — every previously logged score silently becomes incomparable. Persisting the
assignment turns that from an invisible event into a detectable one. And **re-deriving it
and comparing beats recording the config alongside it**, because a stored config only
confirms what the writer *claimed*; re-deriving checks the thing itself.

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

---

## 2026-08-28 — Marginals cannot see a joint shift

Adversarial validation: throw away the real target, label every row by which *file* it
came from, and try to classify that. AUC ≈ 0.5 means nothing distinguishes train from
test, which is the licence to use random folds. It converts "are these two 986,000-row
13-dimensional distributions the same?" into "what's the AUC?" — a question you already
have tools for.

**I argued it was probably unnecessary here, and the data said otherwise.** Every
univariate check passed, spectacularly:

    per-column null rates    identical to FIVE decimal places (11.01294 vs 11.01291)
    numeric means            gaps ≤ 0.006 SD
    quantiles                matching to three decimals
    largest Spearman diff    0.006
    solo adversarial AUC     0.4989 – 0.5216 — every one of the 13 at chance

    all 13 features together → AUC 0.6518, per-fold sd 0.0012

The shift was in the **number of nulls per row**:

    nulls/row      independence    train      test
      0                50.7578    50.6635   55.8182
      1                35.8686    35.9569   29.0293
      4                 0.2314     0.2143    0.8548
    mean                 0.6514     0.6514    0.6514    ← identical
    variance             0.6012     0.5967    0.7916    ← test +32%

Train's nulls are drawn independently per column; test's **co-occur**. Same rates, same
mean, clustered instead of scattered.

**Why no plot could have found it.** The shifted quantity is a property of a *row*, not a
*column*. Each of the 13 marginal plots integrates over the other 12 — which is exactly
where the information lives. **You can plot 13 marginals; you cannot plot a
13-dimensional joint.** Adversarial validation is a *search* over the joint, and it is the
only tool that scales past two or three dimensions.

**But it is bad at the question plots are good at — its importances lied.**

    water_intake          gain 32.34%   solo AUC 0.4990   ← ranked 1st, at chance
    gender                gain  2.61%   solo AUC 0.5216   ← ranked 8th, most shifted

Gain counts how much a feature helped *split*, and a continuous column with 12,000
distinct values offers vastly more split points than a 3-level categorical. Here the
ranking was close to **anti-correlated** with real univariate shift. Taking it at face
value sends you hunting in the wrong column. The division of labour: **plots say what
differs, AUC says whether anything differs, and neither substitutes for the other.**

**Two controls that made the result trustworthy**, both cheap, both worth making routine:

- **Shuffle the labels and re-run.** Got 0.5002, so the 0.6518 is not a harness bug. A
  positive adversarial result without this control is an unverified claim.
- **Compare the observed joint against its independence baseline.** Convolving the 13
  per-column null probabilities gives the null-count distribution you'd see if they were
  independent. Train landed on it (50.76 predicted, 50.66 observed); test did not. Same
  reasoning as "equal counts are not evidence of a shared mask" — **only the joint
  distinguishes the two stories, and you need the baseline to read it.**

Exclude `id` before running: competition ids are assigned per file, so train and test
occupy disjoint contiguous ranges (here 0–690,087 and 690,088–985,840) and one split
separates them perfectly. AUC 1.0 that means nothing.

---

## 2026-08-28 — A shift is only a fold problem if it moves the ranking

Adversarial validation came back 0.6518 and the reflex was: the i.i.d. premise is dead,
redesign the folds. That reflex skips the question that actually decides it. **Finding a
shift is a question, not an answer.** A shift only reaches the fold design if it damages
something the folds are there to protect, and there are exactly two ways it can:

- **Level bias** — CV sits systematically above (or below) the leaderboard. Affects the
  absolute number, not which model wins. And frozen paired folds are already immune: the
  bias is the same for every model, so it cancels out of every comparison.
- **Ranking distortion** — test over-weights a region, models differ in that region, and
  the model that wins on CV is not the model that wins on test. *This* is the harm worth
  paying for.

Ranking distortion needs **both** conditions. Test reweighting a region is half the
argument; the other half is that models disagree there, and that half is usually the one
nobody checks.

### Price it before you fix it

The whole thing reduces to arithmetic against the resolution you already computed.

The framing that makes it mechanical: **a score is a weighted average over slices, and CV
and test weight the same two slices differently** — CV = 0.978·light + 0.022·heavy, test =
0.954·light + 0.046·heavy. Price every option by moving only the weights. For one model
that is a *level* move, at most (0.0455 − 0.0217)·0.10 ≈ 0.002 — and it hits every model
identically, so comparisons subtract it away. For two models it is a *ranking* move, and
only their disagreement on the heavy slice is exposed to it.

The shifted region here was rows with ≥3 nulls: 2.17% of train, 4.55% of test. Two models
tied overall but differing by δ on that slice have the gap between them moved by
`(0.0455 − 0.0217)·δ = 0.0238·δ`. Set that against the 0.001 resolution:

    0.0238 · δ  >  0.001    →    δ > 0.042

**A model would need a four-point balanced-accuracy advantage confined to 2% of the rows,
while being otherwise tied, before the reweighting flips anything.** That is the honest
ceiling on the harm, and it is one number.

Same arithmetic kills the tempting middle option — stratifying folds on the shifted
quantity. Per-fold share of the k≥3 bucket has SE `√(0.0217·0.978/138,018) = 0.00039`, so
even a generous 0.10 difficulty gap moves a fold score by **0.00004** against a 0.002
fold-level SE. Four orders below the resolution, for a compound key frozen forever. This
is the third time that calculation has said no; it is worth reaching for by reflex.

And note what stratifying would even do: it makes the folds resemble **each other**, not
test. Variance reduction, not shift correction. Only reweighting or deliberately
test-like folds address the shift at all — and both correct ≤0.002 using *estimated*
weights, which is paying variance to remove a bias smaller than the instrument.

### Two checks that turn "there is a shift" into "here is the shift"

An AUC is one number and one number cannot tell you what to do. Both of these decompose it:

**1. Strip the suspected channel and re-run.** Restricting to complete-case rows — no
nulls at all, so the classifier physically cannot use missingness — took 0.6518 down to
0.5304 against a 0.4999 control. That is an *ablation of the adversarial classifier*, and
it converts a suspicion into a measurement: the shift is the null pattern, and something
small (0.53) remains in the values. Generalises to any suspected channel — drop it,
re-run, read the drop.

**2. Ask whether the shifted quantity predicts the target.** Covariate shift only costs
you where p(y|x) has structure. Conditioning on the one informative indicator:

    p(unhealthy | n_nulls, bmi_null)      k=0      k=1      k=2      k=3
      bmi present                       0.0847   0.0851   0.0832   0.0889
      bmi null                            —      0.0279   0.0299   0.0311

Flat in k inside each row, every deviation within 1–2 SE. So the null *count* carries no
signal beyond the `bmi` indicator — and that indicator's marginal rate is identical in
both files to four decimals (2.014%). **The quantity that shifted predicts nothing; the
quantity that predicts didn't shift.** That sentence is what closed the decision, and no
amount of staring at the 0.6518 would have produced it.

### What to do instead of redesigning

Two things, neither of which touches the splitter:

**Carry the shifted quantity as a diagnostic slice key.** Not a stratification key — a
column frozen alongside the folds, so every experiment can report per-slice recall for
free. It costs one column and it is what would catch this reasoning being wrong. Name the
case where it could be: here, a comparison of *missing-data handling* is the one design
axis a missingness shift could plausibly mis-rank, so that is the experiment to watch.

**Write down the falsifiable consequence.** Test rows carry more nulls, so they are
information-poorer, so **LB should land ~0.001–0.002 below CV**. Committing to the sign
and the magnitude in advance is what makes the first submission informative: a 0.02 gap
then means something is wrong that this analysis does not explain. Without the prediction
it is just a number you have no opinion about — the same move as predicting `cv_std ≈
0.002` before the first run.

**The general shape: a diagnostic that changes no decision has told you something.** The
negative result cost two checks and bought a design you can defend, which is worth more
than the same design chosen by not looking.

---

## 2026-08-30 — SD describes individuals, SE describes the estimate

The rate tables report `mean ± SE`, not `mean ± SD`, and the two answer different
questions. SD is the spread of *individual rows*; SE = SD/√n is the wobble of the *group
average* computed from them. Comparing group rates is a question about averages, so SE is
the yardstick.

For a 0/1 outcome the distinction is stark: SD = √(p(1−p)) is fully determined by the mean
itself. At p = 8.5% the SD is **28 percentage points** — "8.5 ± 28" says nothing the 8.5
didn't. The SE is where the information lives, because it also knows n.

Reading a table with it:

    8.469 ± 0.047  vs  8.892 ± 0.26    gap 0.42pp, combined SE √(0.047² + 0.26²) ≈ 0.26
                                       → 1.6 SE → noise
    8.47  ± 0.05   vs  2.79  ± 0.20    → ~28 SE → unambiguously real

Rule of thumb: two rates within ~2 combined SEs are the same rate until more data says
otherwise.

---

## 2026-08-30 — Misread: which way the bmi-null effect points

First reading of the notebook-03 table: "more nulls correlates with unhealthy." Backwards
on both halves. Rows *missing* `bmi` are **less** often unhealthy (2.79% vs 8.47%) and
*more* often fit (8.3% vs 5.7%) — plausibly, people whose BMI never got measured skew
toward the ones nobody was worried about. And the null *count* is not the carrier at all:
split on `bmi_is_null` and the rate is flat in k inside each half. The apparent count
effect in the unsplit table was one indicator dragging the average — the general trap of
an aggregate trend that dissolves when you condition on its confounder, and rows with more
nulls are more likely to include `bmi` among them by simple arithmetic.

The fold decision never needed the direction anyway. It needed exactly two facts: the
quantity that shifted (the count) predicts nothing; the quantity that predicts
(`bmi_is_null`) did not shift.
