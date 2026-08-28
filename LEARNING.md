# LEARNING.md

Concepts understood while working this competition. Append-only, newest section last.
Concepts and reasoning only — no code, no config. If it's in the code, it doesn't
belong here; this is the part that transfers to the next competition.

---

## 2026-08-28 — Balanced accuracy

**Definition.** The macro-average of per-class **recall**. Build the confusion matrix,
divide each diagonal cell by its row sum (= how many rows *truly* belong to that class),
average across classes. S6E7 has 3 classes, so each contributes exactly 1/3 regardless of
how many rows it has.

**Macro vs micro vs weighted.**

| | how it combines | one "unit" is |
|---|---|---|
| micro | pool all rows, then compute | a sample |
| weighted | average per class, weighted by class size | a sample |
| macro | average per class, all classes equal | **a class** |

Micro-averaged recall *is* plain accuracy. So is weighted. Macro is the odd one out — it
deliberately discards class sizes. "Average the per-class accuracies" is the right
intuition, and it is **not** the same as overall accuracy.

**The arithmetic that matters.** One extra correct row in class *k* is worth
`1 / (K · n_k)` points. With classes of 70k / 25k / 5k, a rare-class row is worth **14×**
a majority-class row. This is the same statement as sklearn's "each sample is weighted by
the inverse prevalence of its true class".

**The floor is 1/3, not 0.5.** Always-predict-majority scores recall 1 on one class and 0
on the others → 0.333. Identical to random guessing. Don't read a 0.45 as failure until
you've seen the fold spread.

**Consequence for CV.** StratifiedKFold is mandatory. Per-class recall computed on a fold
that received too few rare-class rows is high-variance, and stratification is the direct
fix. Also expect a larger `cv_std` than usual, because the metric is a step function
(see below) — which is exactly why CLAUDE.md demands `cv_std` alongside `cv_mean`.

---

## 2026-08-28 — Gradient boosting, mechanically

For K classes the model maintains **K raw scores per row**, `F_1(x) … F_K(x)`. Softmax
turns them into probabilities. The loss is multiclass log-loss — the negative log of the
probability assigned to the *true* class:

    L = − log q_y(x)

Each boosting iteration fits one regression tree **per class** to the negative gradient.
For softmax + cross-entropy that gradient collapses to:

    ∂L / ∂F_k  =  q_k − 1[y = k]

The tree fits *(is this the true class?) − (probability we gave it)*. True class B with
`q_B = 0.30` → residual 0.70 → push `F_B` up. Multiply by the learning rate, add, repeat.
500 rounds × 3 classes = 1,500 trees.

**The thing to notice:** nothing in that loop computes an argmax, and nothing in it knows
what balanced accuracy is. Training optimizes *probability quality*. Collapsing
probabilities into one label is a separate step, outside the model — and that is where
the competition metric finally enters.

Sample weights plug in as `Σ wᵢ · Lᵢ`. Weighting a rare-class row by 14 makes its gradient
14× larger, shifting the learned probabilities as if that class were 14× more common.

---

## 2026-08-28 — Surrogate loss vs evaluation metric

**You cannot train on balanced accuracy. Nobody can.**

Boosting needs a differentiable loss. Balanced accuracy depends only on the argmax: nudge
a probability from 0.51 to 0.52 and the score doesn't move; nudge it past an argmax flip
and it jumps. The derivative is zero everywhere it exists and undefined at the jumps.
There is nothing to descend.

So every competition with a non-differentiable metric has this two-stage shape:

    train on a differentiable SURROGATE   (log-loss)
            ↓
    evaluate and tune on the REAL metric  (balanced accuracy)

Log-loss is a good surrogate — low log-loss usually means good balanced accuracy — but
"usually" is not "optimally". The gap between the two is free score, recovered at
stage 2. This is a general pattern, not an S6E7 quirk.

---

## 2026-08-28 — Argmax is a choice, not a law

A multiclass model outputs probabilities, not a label. Something must collapse them.
Every library's `.predict()` takes the largest — argmax — because argmax is provably
optimal **for plain accuracy**. It is the default because accuracy is the unstated
default assumption everywhere.

It is wrong for balanced accuracy. The model learned `p(k|x)` from data where the
majority class dominates, so the prior is baked into the probability — and balanced
accuracy has explicitly declared that prior worth nothing.

Decision theory: predicting class *k* pays `p(k|x) · 1/(K·n_k)`, so pick the *k*
maximising `p(k|x) / π_k`. **Divide the probability by the class prior.**

    p = (0.55, 0.30, 0.15),  π = (0.70, 0.25, 0.05)
    → (0.79,  1.20,  3.00)   → predict class 3, not class 1

Same model, same probabilities, opposite answer. Costs zero training time.

Two routes to the same place, and which wins is an **experiment**, not a known:

- **Route 1 — weight during training** (`sample_weight ∝ 1/n_class`). The model learns
  under an effectively balanced prior, so plain argmax becomes roughly correct. Wins when
  a class is so rare that an unweighted model barely learns it at all — no post-hoc fix
  recovers information that was never encoded.
- **Route 2 — adjust after training.** Free, instant, applies to any already-trained
  model, optimizes the real metric directly, and keeps probabilities undistorted.
  Usually the better starting point.

---

## 2026-08-28 — Calibration

A model is **calibrated** when its stated probabilities match observed frequencies: of
all rows where it said 0.70, about 70% really are that class. Plot predicted vs observed
and perfect calibration is the diagonal — that's the reliability curve in
`oof_diagnostics`.

Models drift off it routinely. Long boosting runs push probabilities toward 0/1
(overconfident); early stopping and regularisation stop short (underconfident); rare
classes are systematically under-predicted because the loss barely notices them; class
weighting distorts probabilities deliberately.

**Why this breaks divide-by-prior.** The rule `argmax p(k|x)/π_k` is *derived* assuming
`p` is the true posterior. The model emits a distorted estimate `q`. Dividing by π
corrects the prior and does nothing about the distortion — a correct adjustment applied
to a wrong input.

    true posterior for rare class:  0.15 / 0.05 = 3.00  → predict it   ✓
    model under-predicts it:        0.08 / 0.05 = 1.60  → predict other ✗

**The fix: stop trusting the derivation, let the data pick the number.** Don't divide by
π — treat the divisor as a free parameter. Predict `argmax_k m_k · q(k|x)` and choose `m`
by maximising balanced accuracy directly on the OOF vector.

If the model happens to be calibrated the optimizer lands near `1/π` on its own, so
nothing is lost. If it isn't, the optimizer absorbs the prior correction *and* the
calibration error in one step, without needing to know which is which.

Limit: one multiplier per class can only undo a per-class multiplicative distortion. A
model overconfident at high probabilities but accurate at low ones needs explicit
calibration (Platt / isotonic, fitted inside the fold). Look at the reliability curve
first; escalate only if it's ugly.

**Worked toy** — 6 rows, class counts A=3, B=2, C=1:

| rule | multipliers | balanced accuracy |
|---|---|---|
| plain argmax | (1, 1, 1) | 0.500 |
| divide by prior | (1, 1.5, 3) | 0.778 |
| searched on OOF | (1, 1.2, 2.5) | **0.889** |

`1/π` captures most of the gain — the prior intuition is sound. The search beats it
because the probabilities aren't calibrated. Note the searched rule *sacrifices* a
correct majority-class row to win two minority ones: a bad trade under accuracy, a great
one under balanced accuracy. The search finds that trade without being told.

**Who searches?** Not the model. A small loop in our own code, after CV, over 2 free
parameters (scaling all K multipliers by a constant doesn't change any argmax, so one can
be fixed at 1). No gradients — just evaluate the metric on a few hundred candidates.

Practical ordering: start with `1/π`. The search is a later experiment with its own
`exp_id`, worth perhaps a few thousandths.

---

## 2026-08-28 — OOF (out-of-fold predictions)

In k-fold CV, each fold is predicted by a model trained on the *other* folds. Stack those
prediction blocks back together in row order and you get **one prediction for every
training row, each made by a model that never saw that row** — same length as `y_train`,
aligned index-for-index.

    fold 0  [pred ][            train             ]
    fold 1  [train][pred ][        train          ]
    fold 2  [       train      ][pred ][  train   ]
                        ↓ stack
    OOF     [pred ][pred ][pred ][pred ][pred ]

**Why it matters: the difference between a number and a vector.** A CV score of 0.612 is
a scalar — you can only compare it to another scalar. The OOF vector is a prediction for
every row paired with its true label. That's a dataset, and you can *optimize against it*:

- tune the decision rule (the multipliers above) — needs predictions and truth on the
  same rows
- **blend** — search weights for `0.6 × lgbm + 0.4 × catboost` directly
- **stack** — feed base models' OOF predictions as features to a meta-model
- **error analysis** — `oof_diagnostics` needs per-row error to find the failing slice

None of that is possible from a scalar. Hence CLAUDE.md: *without OOF you cannot blend.*

**Save probabilities, not labels.** `(n_rows, n_classes)` in `oof/exp_XXXX.npy`. You can
always argmax later; you can never recover probabilities from labels — and the entire
decision-rule tuning depends on having them.

**Leakage contaminates OOF silently.** Fit a scaler, imputer, or target encoder on the
full training set before splitting and fold *k*'s "unseen" rows influenced the model that
predicts them. The OOF vector still looks fine. Every weight tuned on it is then tuned
against a lie, and it only surfaces on the leaderboard. This is why every transform must
fit inside the fold.

**Fold-wise score ≠ whole-vector score.** Scoring each fold separately gives `cv_mean`
*and* `cv_std`; scoring the stacked vector once gives a single close-but-different number.
Use the fold-wise version — the std is what tells you whether +0.001 is real.

OOF is **not** your test predictions. Those come from averaging the K fold-models on test
(or refitting on all data). Different object, different purpose.

---

## 2026-08-28 — Counting pairs without a loop

A confusion matrix is just a count of how often each **(true class, predicted class)**
pair occurs. Cell `[i][j]` = rows that were truly *i* but predicted *j*. Every row of
data contributes exactly one pair.

Counting utilities count *single integers*, not tuples. So squash the pair into one
integer using base-*n* positional notation:

    flat = true_index * n_classes + predicted_index

With 2 classes: (0,0)→0, (0,1)→1, (1,0)→2, (1,1)→3. Unique, no collisions — the same
arithmetic as "row *r*, column *c* of a grid *w* wide is cell `r*w + c`". Count the flat
integers, reshape back to *n × n*, and the confusion matrix falls out with no Python
loop at all.

Why it's worth the trick: a Python loop over 690k rows is roughly a second. The
decision-rule search calls this hundreds of times. One C-level pass instead of a loop is
the difference between a search that runs and one that doesn't.

Generalises well beyond confusion matrices — any time you need to count combinations of
a few small-cardinality integers, encode them into one integer and count that.

---

## 2026-08-28 — Where probabilities stop and labels begin

The metric never sees probabilities. The pipeline is:

    model         →  Q        (n_rows, n_classes)  floats, rows sum to 1
    decision rule →  y_pred   (n_rows,)            one label per row
    metric        →  score    one number

Balanced accuracy is defined on the confusion matrix, which is defined on hard labels.
By the time it is called, the collapse has already happened, so it cannot distinguish:

    q = (0.51, 0.49, 0.00)  → argmax 0
    q = (0.99, 0.01, 0.00)  → argmax 0     identical contribution to the score

All confidence information is discarded. That is not an implementation detail — it *is*
the step-function property, and it's why there's no gradient to train on.

Probabilities do their work one step earlier, in the decision rule:

    for each candidate m:
        y_pred = argmax(Q * m)          ← probabilities used here
        score  = metric(y_true, y_pred) ← labels only

This is the concrete reason to save **probabilities, not labels**, as OOF. Trying a new
`m` requires `Q`. From stored labels there is nothing left to search over — the argmax
already destroyed the information.

---

## 2026-08-28 — Why reimplement a metric sklearn already has

Not because sklearn is wrong — sklearn is the **reference**, and the unit test asserts
agreement with it on random inputs. Four reasons:

1. **Writing it is how you learn what you're scored on.** CLAUDE.md workflow step 1.
2. **Edge cases it handles silently.** If a class is absent from `y_true`,
   `balanced_accuracy_score` drops it, warns, and divides by K−1. That is a *different
   metric* than Kaggle computes on the full test set, and it would quietly distort one
   fold. Our version pins the behaviour down with a test.
3. **Speed where it matters.** The multiplier search calls the metric hundreds of times
   over ~690k rows. sklearn re-validates inputs every call and builds the confusion
   matrix through a general path; a lean `bincount` version skips that.
4. **One call site.** `cv.py`, the multiplier search, and `oof_diagnostics` all score
   identically by construction.
