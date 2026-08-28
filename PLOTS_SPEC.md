# plots.py — specification

You implement this file. Claude may review and critique, never write it.

The point isn't pretty charts. It's that each of these answers a specific question
that changes a decision you're about to make. **If a plot doesn't change a decision,
don't build it.**

## Contract

Every function:

- takes a Polars DataFrame (convert internally if a library needs pandas)
- returns a `matplotlib.figure.Figure`
- never calls `plt.show()`, `plt.savefig()`, or `plt.tight_layout()` on a global
- never mutates the input frame
- has type hints and a docstring naming the decision it informs

```python
def target_overview(df: pl.DataFrame, target: str) -> Figure: ...
```

Why return a Figure instead of showing it: the caller decides whether to display in a
notebook, save to disk, or drop into a report. Testable, reusable, no global state.

Suggested shared helper:

```python
def _grid(n: int, ncols: int = 3) -> tuple[Figure, np.ndarray]:
    """Create a figure with n subplots in a grid, hiding unused axes."""
```

## Tier 1 — build first, used on day one

### `target_overview(df, target) -> Figure`

**Decision:** which metric behaviour to expect, and whether stratification is needed.

Show:

- Distribution of the target (histogram if continuous, bar counts if categorical)
- For classification: class counts and percentages. Imbalance below ~10% for the
  minority class means you need `StratifiedKFold`, not `KFold`
- For regression: skew and kurtosis in the title. Heavy skew suggests a log transform,
  or a metric that punishes large errors asymmetrically
- Mark any impossible or sentinel values (negatives where impossible, `-999`, `0` used
  as "missing")

### `missingness(df) -> Figure`

**Decision:** whether missingness is itself a feature, and what to impute.

Show:

- Bar of null fraction per column, sorted descending, columns with zero nulls omitted
- A second panel: missingness co-occurrence — do columns go missing together? Blocks of
  correlated missingness usually mean a structural cause (a sensor offline, a form
  section skipped), and that pattern is often predictive on its own

Don't just count nulls. The correlation panel is the part that earns medals.

### `numeric_grid(df, cols, target=None) -> Figure`

**Decision:** which features need transformation, and which are already informative.

Show, one subplot per column:

- Histogram or KDE
- If `target` is given: overlay the distribution split by target class, or a
  scatter/hexbin against a continuous target
- Flag in the subplot title: high skew, near-zero variance, or a suspicious number of
  exact-duplicate values (a hint of synthetic data or capping)

### `categorical_grid(df, cols, target) -> Figure`

**Decision:** encoding strategy per column.

Show, one subplot per column:

- Value counts, top ~15 with the rest collapsed into "other"
- Target mean (or class rate) per category, with a horizontal line at the global mean
- Cardinality in the title. Under ~10 → one-hot. Over ~50 → target encoding or
  CatBoost's native handling. In between → test both

Also flag categories appearing in test but not train. Those break naive encoders.

### `correlation(df, cols, method="spearman") -> Figure`

**Decision:** which redundant features to drop, and where multicollinearity will
destabilise linear models.

Show:

- Heatmap, masked to the lower triangle
- Spearman by default (rank-based, handles non-linearity and outliers better than
  Pearson for this purpose)
- Optionally a separate sorted bar of each feature's correlation with the target

## Tier 2 — the ones that actually win competitions

These are absent from every tutorial plotting library. Build them after Tier 1.

### `train_test_shift(train, test, cols) -> Figure`

**Decision:** whether your CV can be trusted at all.

This is the visual companion to adversarial validation. Adversarial validation gives you
one AUC number; this shows you which columns caused it.

Show, one subplot per column:

- Train and test distributions overlaid (normalised so different row counts don't
  mislead you)
- A distance measure in the title — KS statistic for numeric, population stability index
  or total-variation distance for categorical
- Sort subplots by that distance, worst first

If the top few columns show visible separation, you have covariate shift. Either design
folds that reproduce it or drop those features.

### `fold_distribution(df, folds, target) -> Figure`

**Decision:** whether your fold assignment is actually valid.

Show:

- Row count per fold (should be near-equal)
- Target mean or class rate per fold, with the global rate as a reference line
- If grouped: confirm zero group overlap between folds, and show group count per fold
- If time-based: show the date range covered by each fold

Run this immediately after generating folds, before any modeling. A silently broken fold
split invalidates every experiment downstream, and it's invisible in the CV number — the
CV will just look suspiciously good.

### `oof_diagnostics(y_true, oof) -> Figure`

**Decision:** what to fix next.

A single CV number tells you how much you're wrong. This tells you where.

Show:

- Predicted vs actual (scatter for regression, or a reliability/calibration curve for
  classification)
- Residual distribution, and residuals against the prediction to reveal
  heteroscedasticity
- Error broken down by segment: quantiles of the prediction, and by the levels of one or
  two key categorical features
- The worst-N rows by absolute error, as a table in the figure

Almost every real gain after the baseline comes from noticing that error is concentrated
in one identifiable slice.

### `importance(model, feature_names, top=30) -> Figure`

**Decision:** what to prune, and whether a leak exists.

Show:

- Importance, sorted, top N
- Mean and standard deviation across folds, not a single fold — a feature that's
  important in one fold and irrelevant in others is noise
- Prefer permutation importance or SHAP over split-count gain; gain is biased toward
  high-cardinality features

One feature dominating overwhelmingly is a leak signal. Investigate before celebrating.

## Order to build

1. `target_overview`, `missingness` — before touching S6E7's features
2. `numeric_grid`, `categorical_grid`, `correlation` — during first EDA
3. `fold_distribution` — the moment you generate folds, no exceptions
4. `train_test_shift` — alongside adversarial validation
5. `oof_diagnostics` — after your first baseline
6. `importance` — after your first tuned model

## Tests

`tests/test_plots.py` — for each function, assert it returns a `Figure` and doesn't raise
on: an empty frame, a frame with all-null columns, a single-row frame, and a column with
one unique value. Plotting code fails on edge cases far more often than modeling code,
and always at the worst moment.
