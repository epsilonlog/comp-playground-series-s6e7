# LEARNING.md

Concepts understood while working this competition. Concepts and reasoning only — no
code, no config. If it's in the repo, it doesn't belong here.

Append new sections at the end. Consolidate periodically: merge overlapping entries,
keep the worked numbers, drop the scaffolding.

---

## 2026-08-28 — Balanced accuracy

The macro-average of per-class **recall**. Build the confusion matrix, divide each
diagonal cell by its row sum, average across classes. Each class contributes 1/K
regardless of size.

Macro is the odd one out: micro-averaged recall and weighted-average recall both *equal*
plain accuracy. Only macro discards class sizes. "Average the per-class accuracies" is
the right intuition — and it is not the same as overall accuracy:

    classes 70k / 25k / 5k, predict the majority for everything
    recalls  1.00, 0.00, 0.00   →  balanced accuracy 0.333
                                →  plain accuracy    0.700

**The floor is 1/K, not 0.5.** Always-one-class and random guessing both score 1/K. The
bottom third of the raw scale carries no information, which is what `adjusted=True`
rescales away: `(score − 1/K) / (1 − 1/K)`, chance → 0. Kaggle scores unadjusted, so
keep the conversion as a reality check — **a raw 0.50 at K=3 is an adjusted 0.25**, only
a quarter of the way from chance to perfect.

**Row value.** One extra correct row in class *k* is worth `1/(K·n_k)`. At 70k/25k/5k a
rare-class row is worth **14×** a majority-class row. Same statement as "each sample is
weighted by the inverse prevalence of its true class".

**Consequences for CV.** StratifiedKFold is mandatory — per-class recall on a fold short
of the rare class is high-variance. Expect a larger `cv_std` than usual, because the
metric is a step function (below).

**Why reimplement it when sklearn has it.** sklearn stays the reference and the test
asserts agreement. But: writing it is how you learn what you're scored on; if a class is
absent from `y_true`, sklearn silently drops it and divides by K−1, which is a *different
metric* than the host computes and would distort one fold; the decision-rule search calls
it hundreds of times over ~690k rows; and it gives one call site for every scorer in the
project.

---

## 2026-08-28 — Training optimises one thing, deciding optimises another

**Gradient boosting, mechanically.** For K classes the model keeps K raw scores per row,
softmaxed into probabilities. The loss is log-loss, `L = −log q_y(x)`. Each iteration
fits one tree per class to the negative gradient, which for softmax + cross-entropy
collapses to:

    ∂L/∂F_k  =  q_k − 1[y = k]

The tree fits *(is this the true class?) − (probability we gave it)*. True class B at
`q_B = 0.30` → residual 0.70 → push `F_B` up. Nothing in that loop computes an argmax or
knows what balanced accuracy is.

**You cannot train on balanced accuracy. Nobody can.** It depends only on the argmax:
nudge a probability from 0.51 to 0.52 and nothing moves; nudge it past a flip and it
jumps. Derivative zero where it exists, undefined at the jumps. So every
non-differentiable metric forces the same two-stage shape — train on a differentiable
**surrogate** (log-loss), then evaluate and tune on the **real metric**. The gap between
them is free score.

**Where probabilities stop and labels begin.**

    model → Q (n_rows, K) floats → decision rule → y_pred (n_rows,) → metric → score

The metric cannot distinguish `(0.51, 0.49, 0.00)` from `(0.99, 0.01, 0.00)`; both argmax
to the same class. All confidence information is discarded before it is called.

**Argmax is a choice, not a law.** It is provably optimal *for plain accuracy* — the
default everywhere because accuracy is the unstated default. It's wrong here: the model
learned `p(k|x)` under the training prior, and balanced accuracy declares that prior
worth nothing. Decision theory says maximise `p(k|x)/π_k`:

    p = (0.55, 0.30, 0.15),  π = (0.70, 0.25, 0.05)
      →  (0.79,  1.20,  3.00)   → predict class 3, not class 1

Two routes, and which wins is an experiment: **weight during training**
(`sample_weight ∝ 1/n_class`, so argmax becomes correct — wins when a class is so rare an
unweighted model never learns it), or **adjust after training** (free, instant, applies to
any trained model, keeps probabilities undistorted — usually the better start).

**Calibration.** A model is calibrated when its stated probabilities match observed
frequencies: of rows where it said 0.70, about 70% really are that class. Long boosting
runs overshoot toward 0/1; early stopping and regularisation undershoot; rare classes are
systematically under-predicted; class weighting distorts deliberately.

This breaks divide-by-prior, because that rule assumes `p` is the *true* posterior:

    true posterior:      0.15 / 0.05 = 3.00  → predict it    ✓
    model under-predicts: 0.08 / 0.05 = 1.60  → predict other ✗

**So stop trusting the derivation and let the data pick the number.** Treat the divisor
as a free parameter: predict `argmax_k m_k · q(k|x)`, choose `m` by maximising the metric
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
No gradients. Start with `1/π`; the search is a later experiment worth a few thousandths.

A multiplier can only undo a per-class *multiplicative* distortion. A model overconfident
at high probabilities but accurate at low ones needs explicit calibration (Platt /
isotonic, fitted inside the fold). Check the reliability curve first.

---

## 2026-08-28 — OOF (out-of-fold predictions)

Each fold is predicted by a model trained on the other folds. Stack the blocks back in
row order and you get **one prediction per training row, each from a model that never saw
that row** — same length as `y_train`, aligned index-for-index.

**Why it matters: the difference between a number and a vector.** A CV score of 0.612 is
a scalar; you can only compare it to another scalar. The OOF vector is a prediction for
every row paired with its true label — a dataset you can optimise against: tune the
decision rule, search blend weights, stack base models as meta-features, find which slice
the error concentrates in. None of that is possible from a scalar.

**Save probabilities, not labels.** Trying a new `m` requires `Q`. From stored labels
there is nothing left to search over — the argmax destroyed it.

**Leakage contaminates OOF silently.** Fit a scaler, imputer, or target encoder on the
full training set before splitting and fold *k*'s "unseen" rows influenced the model that
predicts them. The vector still looks fine; everything tuned on it is tuned against a lie,
and it only surfaces on the leaderboard.

**Fold-wise score ≠ whole-vector score.** Scoring each fold separately gives `cv_mean`
*and* `cv_std`. Use that — the std is what tells you whether +0.001 is real.

OOF is **not** your test predictions; those average the K fold-models on test.

---

## 2026-08-28 — Encoding ordered categories

`low < medium < high` is a fact about **words**. Encoding it `0, 1, 2` is an empirical
claim about the **target**, and the words can be ordered while the claim is false.

A tree splits on a threshold, so with three levels the only bipartitions available are
`{low} | {medium, high}` and `{low, medium} | {high}`. **`medium` cannot be isolated in
one split.** The encoding is a *constraint* restricting the model to contiguous groupings
in your chosen order.

**The rule is contiguity, not monotonicity — I had this wrong.** Real data showed
`stress_level` strongly non-monotone in the majority class (0.80 → 0.99 → 0.72, peaking in
the middle), which by a monotonicity rule should favour one-hot. It doesn't: the
*informative* levels were the two extremes, and threshold splits isolate extremes
perfectly. Only the middle level was unreachable, and the middle level was the boring one.

Ask **"are the levels I need to isolate contiguous in this ordering?"** Monotonicity is
sufficient for that, not necessary. A U-shape with interesting ends is fine; an
interesting *middle* is not.

**Scale of the effect.** At 3 levels this is minor — one-hot costs 2 columns, ordinal
costs at most one extra split. It becomes decisive at high cardinality. And GBDTs with
native categorical handling find a good bipartition from gradient statistics with no
ordering at all, sidestepping the question.

**Where verifying monotonicity does pay: monotone constraints.** Forcing a feature's
effect to be monotone eliminates a class of overfitting where the model learns a wiggle
that is really noise. That needs both an ordinal encoding *and* evidence the relationship
is monotone — and monotonicity, unlike encoding, really is the requirement.

---

## 2026-08-28 — Describing data honestly

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

**Standardised effect size, and its blind spots.** Raw gaps between class means aren't
comparable across columns — one is in steps, another in hours. Divide each gap by its own
column's standard deviation and both become rankable. As overlap between two bell curves:

    2.1  →  ~29% overlap    nearly separable on that feature alone
    0.9  →  ~65% overlap
    0.05 →  ~98% overlap    the two curves are the same curve

The same quantity read as **overlapping coefficient**: model each class as a normal with
equal σ, means `d·σ` apart. Put them at ±d/2 so the curves cross at 0; by symmetry each
contributes its own tail beyond the crossing, so `OVL = 2·Φ(−d/2)`. More usefully, the
best accuracy a **single threshold** can reach on that feature (equal priors) is
`1 − OVL/2 = Φ(d/2)` — d = 2.1 gives 86%, d = 0.05 gives 51%, a coin flip. Assumes
normality and equal variances.

Three blind spots, all of which bit here: dividing by the *overall* std understates
separation (the pooled within-class std is correct, so the estimate is conservative); it
compares means only, so equal means with different variances score zero and are still
separable; and with more than two classes it reads only the extremes and is blind to the
middle. Two features scored equally at ~0.82 — one separated all three classes, the other
put two classes on top of each other and only found the third. Under a macro metric those
are not equally useful. The ranking says where to look; the raw per-class means say what
you found.

---

## 2026-08-28 — Monotone transformations are invisible to trees

A tree split asks "is x ≤ 7.2?" — a question about **rank order**. Log, sqrt, Box-Cox and
every other monotone transform preserve rank order, so they produce the identical split,
the identical tree, and identical predictions. Log-transforming a feature for a GBDT is a
no-op.

Transforms exist for models that care about *distance* and *linearity*: a linear model
seeing a curved relationship as straight, a squared loss not being dominated by extreme
values, a distance metric not hijacked by scale. GBDTs have none of those properties.

So on a GBDT-first plan the skew question is moot before you measure it. It returns only
when a linear model or a network joins the blend — and then it is usually standardisation
that is wanted, not a log.

**Clipping is a different question and does matter.** A clipped column piles values at a
wall — income capped at 100,000 with 3% of rows exactly there. That spike is real
structure a tree will happily learn, and it usually deserves an indicator or an explicit
censoring treatment. Distinguish it from a *truncated* sampling range, where the density
at the bound is genuinely low and almost nothing piles up: a clip shows percent-at-bound
in the whole numbers, a truncation shows hundredths.

**And distinguish a real zero from a coded missing.** A spike at exactly zero means "not
applicable" mixed into a continuous column — two populations stacked. If the column also
carries explicit nulls, the zero is genuine; if it has none, suspect the zero *is* the
missing code, and converting it changes every downstream statistic.

---

## 2026-08-28 — Unseen categorical levels

A level present at predict time but absent when the encoder was fitted. Every encoder
fails differently and none fail well: ordinal raises or substitutes a fill; one-hot
raises or emits an all-zeros row — a pattern the model never saw in training; target
encoding has no statistic and falls back to the global mean.

Matching level sets between train and test means none of that handling has to be written,
tested, and kept consistent across every fold and every encoder. It removes a category of
code and a category of bug.

Two distinct versions of the problem: **train vs test**, and — subtler — **inside CV**,
where a level appears in the validation fold but not the training folds. Low cardinality
makes both impossible; thousands of rare categories make the second one routine. The
safety is a property of the data, never of the pipeline.

---

## 2026-08-28 — Counting pairs without a loop

A confusion matrix counts how often each **(true, predicted)** pair occurs. Counting
utilities count single integers, not tuples — so squash the pair into one integer with
base-*n* positional notation:

    flat = true_index * n_classes + predicted_index

With 2 classes: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3. Unique, no collisions — the same
arithmetic as "row *r*, column *c* of a grid *w* wide is cell `r·w + c`". Count the flat
integers, reshape to *n × n*, and the matrix falls out with no Python loop.

A Python loop over 690k rows is about a second; the decision-rule search calls this
hundreds of times. Generalises to any counting of combinations of small-cardinality
integers.
