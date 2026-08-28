"""Tabular EDA summaries. Frames in, frames out — never figures.

`plots.py` owns anything that returns a `Figure`. This module owns the checks whose
answer is a number or a small table, which is most of them: a 7-row table beats a
7-panel grid whenever you only need to compare magnitudes.

Everything here is read-only and fits no state, so it is safe to run on full data
outside a fold (CLAUDE.md rule 3 governs *learned* transforms, not description).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from math import erf, sqrt
from typing import Any, Literal, cast

import numpy as np
import polars as pl

NULL_LABEL = "<null>"


def _classes(df: pl.DataFrame, target: str) -> list[str]:
    return sorted(str(value) for value in df[target].drop_nulls().unique().to_list())


def _normal_cdf(x: float) -> float:
    """Standard normal CDF, via erf — avoids pulling in scipy for one function."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _level_sort_key(level: object, declared: list[str] | None) -> tuple[int, Any]:
    """Declared order first, then alphabetical, with nulls always last."""
    if level is None:
        return (2, "")
    if declared is not None:
        text = str(level)
        return (0, declared.index(text) if text in declared else len(declared))
    return (1, str(level))


def overview(df: pl.DataFrame) -> pl.DataFrame:
    """One row per column: dtype, null count and rate, and distinct value count.

    Decision: which columns are numeric vs categorical, which need imputation, and
    which are effectively constant.

    ``n_unique`` excludes nulls. Polars' own ``n_unique`` counts null as a distinct
    value, which silently reports a 3-level categorical with missing data as having 4
    levels.
    """
    height = df.height
    return pl.DataFrame(
        [
            {
                "column": name,
                "dtype": str(series.dtype),
                "nulls": series.null_count(),
                "null_pct": round(100.0 * series.null_count() / height, 2) if height else 0.0,
                "n_unique": series.drop_nulls().n_unique(),
            }
            for name, series in df.to_dict().items()
        ]
    )


def category_levels(
    train: pl.DataFrame, cols: Sequence[str], test: pl.DataFrame | None = None
) -> pl.DataFrame:
    """The distinct levels of each categorical column, and any train/test disagreement.

    Decision: encoding strategy, and whether naive encoders are safe. A level that
    appears only in test has no encoding learned for it and will break a fitted encoder
    at predict time; a level only in train is dead weight.

    Levels come back in **alphabetical** order, which is rarely the meaningful one —
    ``high, low, medium`` sorts nothing like ``low < medium < high``. Deciding whether a
    column is ordinal, and in what order, is a modelling judgement this function
    deliberately does not make for you.
    """
    rows = []
    for col in cols:
        train_levels = train[col].drop_nulls().unique().sort().to_list()
        test_only: str | None = None
        train_only: str | None = None
        if test is not None:
            test_levels = set(test[col].drop_nulls().unique().to_list())
            test_only = ", ".join(sorted(test_levels - set(train_levels)))
            train_only = ", ".join(sorted(set(train_levels) - test_levels))
        rows.append(
            {
                "column": col,
                "n_levels": len(train_levels),
                "levels": ", ".join(str(level) for level in train_levels),
                "test_only": test_only,
                "train_only": train_only,
            }
        )
    return pl.DataFrame(rows)


def level_target_rates(
    df: pl.DataFrame,
    cols: Sequence[str],
    target: str,
    orders: Mapping[str, Sequence[str]] | None = None,
) -> pl.DataFrame:
    """Class rate at each level of each categorical column, in declared semantic order.

    Decision: is the target *monotone* in the level ordering? That is the empirical
    claim an ordinal integer encoding makes — and the precondition for using a monotone
    constraint later. Non-monotone means one-hot (or native categorical handling) wins,
    because a threshold split cannot isolate a middle level.

    Nulls appear as their own level, which doubles as the test of whether missingness
    in a categorical predicts the target.

    Columns without an entry in ``orders`` are listed alphabetically; nulls always last.
    """
    classes = _classes(df, target)
    height = df.height
    records: list[dict[str, Any]] = []

    for col in cols:
        grouped = df.group_by(col).agg(
            pl.len().alias("rows"),
            *[(pl.col(target) == cls).mean().alias(f"p_{cls}") for cls in classes],
        )
        declared = list(orders[col]) if orders and col in orders else None
        keyed = [(_level_sort_key(rec[col], declared), rec) for rec in grouped.to_dicts()]
        keyed.sort(key=lambda pair: pair[0])

        for _, record in keyed:
            level = record[col]
            records.append(
                {
                    "column": col,
                    "level": NULL_LABEL if level is None else str(level),
                    "rows": record["rows"],
                    "share_pct": round(100.0 * record["rows"] / height, 2),
                    **{f"p_{cls}": round(record[f"p_{cls}"], 4) for cls in classes},
                }
            )

    return pl.DataFrame(records)


def numeric_summary(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """Range, quantisation grid, distribution shape, and mass piled at the bounds.

    Decision: which features need transformation, whether a bound is a real limit or a
    clip, and whether the distribution hides a point mass.

    ``grid`` is the **median** gap between adjacent distinct values — the typical
    quantisation step. Do not use the minimum for this: a single off-grid value splits
    one normal step into two smaller ones and drags the minimum down by an order of
    magnitude, making a clean grid look ragged. ``min_gap`` is reported separately
    precisely so that contamination is visible instead of silently corrupting ``grid``.

    ``pct_at_min`` / ``pct_at_max`` expose clipping: a genuine distribution tapers at
    its extremes, a clipped one piles up there. ``mode_pct`` catches a point mass
    *anywhere* — zero-inflation, a sentinel, or a default value — which the bound
    percentages miss unless the spike happens to sit on a bound.
    """
    records: list[dict[str, Any]] = []
    for col in cols:
        series = df[col].drop_nulls()
        if series.is_empty():
            records.append({"column": col, "n_unique": 0})
            continue

        distinct = np.sort(series.unique().to_numpy())
        gaps = np.diff(distinct)
        low, high = float(distinct[0]), float(distinct[-1])
        n = series.len()
        at_min = cast(int, (series == low).sum())
        at_max = cast(int, (series == high).sum())
        modes = series.value_counts(sort=True).row(0)

        records.append(
            {
                "column": col,
                "n_unique": int(distinct.size),
                "min": low,
                "max": high,
                "grid": float(np.median(gaps)) if gaps.size else 0.0,
                "min_gap": float(gaps.min()) if gaps.size else 0.0,
                "mode": float(modes[0]),
                "mode_pct": round(100.0 * int(modes[1]) / n, 2),
                "pct_at_min": round(100.0 * at_min / n, 2),
                "pct_at_max": round(100.0 * at_max / n, 2),
                "mean": round(cast(float, series.mean() or 0.0), 3),
                "std": round(cast(float, series.std() or 0.0), 3),
                "skew": round(series.skew() or 0.0, 3),
            }
        )
    return pl.DataFrame(records)


def missing_vs_target(df: pl.DataFrame, cols: Sequence[str], target: str) -> pl.DataFrame:
    """Class rate when a column is missing versus when it is present.

    Decision: is missingness itself a feature? If the rates differ, add an indicator
    column and stop worrying about clever imputation. If they match, the nulls were
    injected at random and an indicator is dead weight.

    Sort by ``abs_diff`` descending to find any column where it matters.
    """
    classes = _classes(df, target)
    records: list[dict[str, Any]] = []

    for col in cols:
        grouped = df.group_by(pl.col(col).is_null().alias("is_missing")).agg(
            pl.len().alias("rows"),
            *[(pl.col(target) == cls).mean().alias(f"p_{cls}") for cls in classes],
        )
        lookup = {row["is_missing"]: row for row in grouped.to_dicts()}
        if True not in lookup or False not in lookup:
            continue

        for cls in classes:
            when_missing = lookup[True][f"p_{cls}"]
            when_present = lookup[False][f"p_{cls}"]
            records.append(
                {
                    "column": col,
                    "target_class": cls,
                    "n_missing": lookup[True]["rows"],
                    "p_when_missing": round(when_missing, 4),
                    "p_when_present": round(when_present, 4),
                    "abs_diff": round(abs(when_missing - when_present), 4),
                }
            )
    return pl.DataFrame(records)


def missing_cooccurrence(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """Do columns go missing on the same rows?

    Decision: whether missingness has a structural cause worth modelling. ``ratio`` is
    observed joint missingness over what independence would predict. A ratio near 1
    means the columns were nulled independently; a large ratio means they share a mask,
    which usually points at a real mechanism (a skipped form section, an offline sensor).

    Sorted worst-first. Pairs where either column is never null are omitted.
    """
    height = df.height
    null_rate = {col: df[col].null_count() / height for col in cols if df[col].null_count()}
    pairs = list(combinations(null_rate, 2))
    if not pairs:
        return pl.DataFrame(
            schema={
                "col_a": pl.String,
                "col_b": pl.String,
                "both_missing": pl.Int64,
                "expected": pl.Float64,
                "ratio": pl.Float64,
            }
        )

    joint = df.select(
        [(pl.col(a).is_null() & pl.col(b).is_null()).sum().alias(f"{a}|{b}") for a, b in pairs]
    ).row(0, named=True)

    records: list[dict[str, Any]] = []
    for a, b in pairs:
        expected = height * null_rate[a] * null_rate[b]
        observed = int(joint[f"{a}|{b}"])
        records.append(
            {
                "col_a": a,
                "col_b": b,
                "both_missing": observed,
                "expected": round(expected, 1),
                "ratio": round(observed / expected, 2) if expected else None,
            }
        )
    return pl.DataFrame(records).sort("ratio", descending=True, nulls_last=True)


def class_profile(df: pl.DataFrame, cols: Sequence[str], target: str) -> pl.DataFrame:
    """Per-class mean of each numeric feature, ranked by how far apart the classes sit.

    Decision: which features carry signal, before drawing anything. ``spread_sd`` is the
    gap between the largest and smallest class mean, in units of the column's overall
    standard deviation — a crude effect size. Near zero means the feature cannot
    separate the classes on its own and a density plot will show three curves on top of
    each other.

    ``overlap_pct`` and ``best_split_acc`` restate ``spread_sd`` in units you can act on.
    Modelling each class as a normal with equal variance and means ``d`` standard
    deviations apart, the shared area under the two curves is ``2 * Phi(-d/2)``, and the
    best accuracy a *single threshold* on that feature can reach (equal priors) is
    ``Phi(d/2)``. So d = 2.1 means one cut separates the extreme classes ~86% of the
    time; d = 0.05 means a coin flip. Both assume normality and equal variance.

    It only sees *means*. Two classes with equal means and different variances score
    zero here and are still separable — that is the case where a plot beats a table.
    With more than two classes it compares only the extremes, so a feature that splits
    one class off while leaving the other two superimposed scores the same as one that
    separates all three. Read the per-class means, not just the ranking.
    """
    classes = _classes(df, target)
    means = df.group_by(target).agg([pl.col(col).mean().alias(col) for col in cols])
    by_class = {str(row[target]): row for row in means.to_dicts()}
    overall_std = df.select([pl.col(col).std().alias(col) for col in cols]).row(0, named=True)

    records: list[dict[str, Any]] = []
    for col in cols:
        values = [by_class[cls][col] for cls in classes if by_class[cls][col] is not None]
        std = overall_std[col]
        spread = (max(values) - min(values)) / std if values and std else None
        records.append(
            {
                "column": col,
                **{f"mean_{cls}": round(by_class[cls][col], 3) for cls in classes},
                "overall_std": round(std, 3) if std else None,
                "spread_sd": round(spread, 4) if spread is not None else None,
                "overlap_pct": (
                    round(100.0 * 2.0 * _normal_cdf(-spread / 2.0), 1)
                    if spread is not None
                    else None
                ),
                "best_split_acc": (
                    round(_normal_cdf(spread / 2.0), 3) if spread is not None else None
                ),
            }
        )
    return pl.DataFrame(records).sort("spread_sd", descending=True, nulls_last=True)


def numeric_correlation(
    df: pl.DataFrame, cols: Sequence[str], method: Literal["pearson", "spearman"] = "spearman"
) -> pl.DataFrame:
    """Pairwise correlation between numeric columns, strongest first.

    Decision: which features are redundant. Two columns correlated above ~0.95 are one
    feature wearing two hats — the second adds no information, splits the importance
    between them, and destabilises any linear model in the blend.

    Spearman by default: it is rank-based, so it catches monotone-but-curved
    relationships that Pearson underestimates, and it is unmoved by outliers. Nulls are
    dropped pairwise, so each pair uses every row where *both* values are present.

    A GBDT tolerates redundancy far better than a linear model does — this is mostly a
    warning about feature importance being split, and about what to prune before adding
    a linear model to the blend.
    """
    records: list[dict[str, Any]] = []
    for col_a, col_b in combinations(cols, 2):
        pair = df.select(col_a, col_b).drop_nulls()
        value = (
            pair.select(pl.corr(col_a, col_b, method=method)).item() if pair.height > 1 else None
        )
        records.append(
            {
                "col_a": col_a,
                "col_b": col_b,
                "n_pairs": pair.height,
                method: round(value, 4) if value is not None else None,
                "abs": round(abs(value), 4) if value is not None else None,
            }
        )
    return pl.DataFrame(records).sort("abs", descending=True, nulls_last=True)


def _agg_float(value: object) -> float:
    """A polars aggregation as a plain float, with null as 0.0.

    `Series.mean()` is typed as returning a union that includes `timedelta`, because it is
    valid on temporal columns. Callers here only pass numeric columns, so narrowing is
    safe — and stating it once beats a `cast` at every call site.
    """
    return 0.0 if value is None else float(cast(float, value))


def _chi2_sf(chi2: float, dof: int) -> float:
    """Upper tail of the chi-square distribution, Wilson-Hilferty approximation.

    Cubing the reduced chi-square makes it near-normal. Accurate to a few parts in 10^4
    for the dof used here, and keeps this module free of a scipy dependency.
    """
    if dof <= 0:
        return float("nan")
    reduced = (chi2 / dof) ** (1.0 / 3.0)
    mean = 1.0 - 2.0 / (9.0 * dof)
    sd = sqrt(2.0 / (9.0 * dof))
    return 1.0 - _normal_cdf((reduced - mean) / sd)


def numeric_shift(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cols: Sequence[str],
    *,
    n_bins: int = 50,
) -> pl.DataFrame:
    """Compare each numeric column's distribution in train against test.

    Decision: does any single feature differ between the two files, and if so which?

    ``gap_sd`` states the mean difference in the column's own standard deviations, so
    columns in different units are comparable. ``sd_ratio`` catches a spread difference
    that leaves the mean untouched.

    ``chi2`` / ``p_value`` bin the *combined* values into ``n_bins`` quantile bins and ask
    whether each bin holds its expected share of test rows. That catches a non-monotone
    shift — a hole in the middle, a fatter left tail — which means and quantiles both miss.
    ``max_bin_dev`` is the largest absolute departure from the global test share, and is
    the number to read when n is large enough to make every p-value significant.

    Nulls are dropped per column; ``null_gap`` compares the rates separately.

    **A clean row here is weak evidence.** Marginals cannot see a shift that lives in the
    joint distribution — see `null_count_profile` and `adversarial.run`.
    """
    n_test_share = test.height / (train.height + test.height)
    rows = []
    for col in cols:
        a = train[col].drop_nulls().to_numpy()
        b = test[col].drop_nulls().to_numpy()
        edges = np.unique(np.quantile(np.concatenate([a, b]), np.linspace(0.0, 1.0, n_bins + 1)))
        count_a = np.histogram(a, bins=edges)[0]
        count_b = np.histogram(b, bins=edges)[0]
        total = count_a + count_b
        keep = total > 0
        share_b = len(b) / (len(a) + len(b))
        expected_b = total[keep] * share_b
        expected_a = total[keep] - expected_b
        chi2 = float(
            ((count_a[keep] - expected_a) ** 2 / expected_a).sum()
            + ((count_b[keep] - expected_b) ** 2 / expected_b).sum()
        )
        dof = int(keep.sum()) - 1
        mean_train, sd_train = _agg_float(train[col].mean()), _agg_float(train[col].std())
        mean_test, sd_test = _agg_float(test[col].mean()), _agg_float(test[col].std())
        null_train = 100.0 * train[col].null_count() / train.height
        null_test = 100.0 * test[col].null_count() / test.height
        rows.append(
            {
                "column": col,
                "train_mean": mean_train,
                "test_mean": mean_test,
                "gap_sd": (mean_test - mean_train) / sd_train if sd_train else 0.0,
                "sd_ratio": (sd_test / sd_train) if sd_train else 0.0,
                "train_null_pct": null_train,
                "test_null_pct": null_test,
                "null_gap": null_test - null_train,
                "chi2": chi2,
                "dof": dof,
                "p_value": _chi2_sf(chi2, dof),
                "max_bin_dev": float(np.abs(count_b[keep] / total[keep] - n_test_share).max()),
            }
        )
    return pl.DataFrame(rows).sort("max_bin_dev", descending=True)


def category_shift(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cols: Sequence[str],
) -> pl.DataFrame:
    """Level proportions in train against test, largest disagreement first.

    Decision: has any category's share moved between the two files?

    Proportions are of *all* rows, so nulls appear as their own level and the shares in
    each column sum to 100. Levels are collected from the union of both frames, so a
    level present in only one side shows as 0.00 on the other rather than vanishing.
    """
    rows = []
    for col in cols:
        levels = set(train[col].unique().to_list()) | set(test[col].unique().to_list())
        counts_train = dict(train[col].value_counts().iter_rows())
        counts_test = dict(test[col].value_counts().iter_rows())
        for level in sorted(levels, key=lambda x: _level_sort_key(x, None)):
            pct_train = 100.0 * counts_train.get(level, 0) / train.height
            pct_test = 100.0 * counts_test.get(level, 0) / test.height
            rows.append(
                {
                    "column": col,
                    "level": NULL_LABEL if level is None else str(level),
                    "train_pct": pct_train,
                    "test_pct": pct_test,
                    "diff": pct_test - pct_train,
                }
            )
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("diff").abs().alias("abs_diff"))
        .sort("abs_diff", descending=True)
    )


def _null_count(df: pl.DataFrame, cols: Sequence[str]) -> np.ndarray:
    expr = pl.sum_horizontal(pl.col(c).is_null() for c in cols)
    return df.select(expr.alias("k"))["k"].to_numpy()


def _independent_null_pmf(rates: np.ndarray) -> np.ndarray:
    """Poisson-binomial pmf: how many nulls a row would carry if columns were independent.

    Convolving ``[1-p, p]`` over the columns multiplies out every combination — the same
    thing as enumerating 2**n masks, in n steps instead of 2**n.
    """
    pmf = np.array([1.0])
    for p in rates:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def null_count_profile(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cols: Sequence[str],
) -> pl.DataFrame:
    """Distribution of the number of nulls *per row*, against the independence baseline.

    Decision: do nulls land independently across columns, or do they clump on the same
    rows? Per-column null rates cannot answer this — two files can agree on every column
    to five decimal places and still differ here, because this is a property of a **row**
    and every per-column summary integrates the rows away.

    The baseline is built from *train's* per-column rates, so ``independent_pct`` is what
    train itself would produce if its columns were independent. Compare all three columns:
    a file sitting on the baseline draws nulls independently, one departing from it does
    not.
    """
    rates = np.array([train[c].null_count() / train.height for c in cols])
    pmf = _independent_null_pmf(rates)
    k_train = _null_count(train, cols)
    k_test = _null_count(test, cols)
    highest = int(max(k_train.max(), k_test.max(), np.flatnonzero(pmf > 1e-9).max()))
    return pl.DataFrame(
        [
            {
                "n_nulls": k,
                "independent_pct": 100.0 * (pmf[k] if k < len(pmf) else 0.0),
                "train_pct": 100.0 * float((k_train == k).mean()),
                "test_pct": 100.0 * float((k_test == k).mean()),
                "test_minus_independent": 100.0
                * (float((k_test == k).mean()) - (pmf[k] if k < len(pmf) else 0.0)),
            }
            for k in range(highest + 1)
        ]
    )


def missingness_dispersion(
    train: pl.DataFrame,
    test: pl.DataFrame,
    cols: Sequence[str],
) -> pl.DataFrame:
    """Mean and variance of the per-row null count, beside the independence prediction.

    Decision: the one-line verdict behind `null_count_profile`. Independent draws give
    ``variance = sum p(1-p)``. Equal means with an inflated variance is the signature of
    **clustered** missingness: the same number of nulls overall, concentrated on fewer
    rows.
    """
    rates = np.array([train[c].null_count() / train.height for c in cols])
    predicted_var = float((rates * (1.0 - rates)).sum())
    rows = [{"source": "independent", "mean": float(rates.sum()), "variance": predicted_var}]
    for name, frame in (("train", train), ("test", test)):
        k = _null_count(frame, cols)
        rows.append({"source": name, "mean": float(k.mean()), "variance": float(k.var())})
    return pl.DataFrame(rows).with_columns(
        (pl.col("variance") / predicted_var).alias("var_vs_independent")
    )


def null_count_vs_target(
    df: pl.DataFrame,
    cols: Sequence[str],
    target: str,
    *,
    split_on: str | None = None,
    min_rows: int = 100,
) -> pl.DataFrame:
    """Target rates by per-row null count, optionally split by one column's own nullity.

    Decision: **does a shift in the missingness pattern cost anything?** A train/test
    difference in the per-row null count only matters if that count predicts the target.
    If the rates are flat in `n_nulls`, what shifted carries no signal and the fold design
    does not have to answer for it.

    `split_on` is what turns a suspicion into an answer. A single informative null
    indicator will bend the unsplit rates all on its own, and the bend looks exactly like
    the null *count* mattering. Split on that indicator and the two readings separate:
    rates flat **within** each half mean the count adds nothing beyond the indicator.

    `se_*` is the binomial standard error of the rate in the same percentage units — a
    deviation to compare against, so "flat" is a measurement rather than an impression.
    Buckets thinner than `min_rows` are dropped; below that the error bar is wider than
    any effect worth reading.
    """
    classes = _classes(df, target)
    counts = _null_count(df, cols)
    y = df[target].to_numpy()

    groups: list[tuple[str, np.ndarray]] = [("all", np.ones(df.height, dtype=bool))]
    if split_on is not None:
        is_null = df[split_on].is_null().to_numpy()
        groups = [(f"{split_on} present", ~is_null), (f"{split_on} null", is_null)]

    rows = []
    for label, mask in groups:
        for k in range(int(counts.max()) + 1):
            selected = mask & (counts == k)
            n = int(selected.sum())
            if n < min_rows:
                continue
            row: dict[str, Any] = {"split": label, "n_nulls": k, "n_rows": n}
            for cls in classes:
                p = float((y[selected] == cls).mean())
                row[f"pct_{cls}"] = round(100.0 * p, 3)
                row[f"se_{cls}"] = round(100.0 * sqrt(p * (1.0 - p) / n), 3)
            rows.append(row)
    return pl.DataFrame(rows)


def shift_power(
    n_rows: int,
    *,
    n_bins: int = 50,
    pi: float = 0.3,
    deltas: Sequence[float] = (0.0, 0.002, 0.005, 0.010, 0.015, 0.020, 0.030),
    shifted_bins: int | None = 1,
    trials: int = 400,
    seed: int = 42,
) -> pl.DataFrame:
    """How large a shift must be before `numeric_shift` can see it. Simulated.

    Decision: what a **large** p-value licenses you to conclude, and what counts as a
    **big** `max_bin_dev`. Both need a reference point that no single dataset provides.

    Injects a known deviation `delta` into `shifted_bins` of the bins, runs the same
    chi-square, and reports how often it fires. Two readings come out of it:

    - **The noise floor.** At `delta = 0`, `max_bin_dev` does *not* come back near zero —
      with `n_bins` bins the luckiest one always drifts. That floor, not zero, is what an
      observed `max_bin_dev` has to beat.
    - **The resolution.** The smallest `delta` detected reliably. A large p-value only
      rules out shifts *above* it, so this is what "no shift" actually means here.

    `shifted_bins=None` shifts every bin (a whole-column drift, the easiest case to
    detect); a single bin is the hardest, because 49 well-behaved bins dilute it.

    Pure simulation — it depends only on the shape of the problem, not on the data, so it
    can be run before the real test.
    """
    rng = np.random.default_rng(seed)
    per_bin = np.full(n_bins, n_rows // n_bins)
    k = n_bins if shifted_bins is None else shifted_bins
    rows = []
    for delta in deltas:
        shares = np.full(n_bins, pi)
        shares[:k] = pi + delta
        fired, p_values, devs = 0, [], []
        for _ in range(trials):
            count_b = rng.binomial(per_bin, shares)
            count_a = per_bin - count_b
            expected_b = per_bin * pi
            expected_a = per_bin - expected_b
            chi2 = float(
                (((count_a - expected_a) ** 2) / expected_a).sum()
                + (((count_b - expected_b) ** 2) / expected_b).sum()
            )
            p = _chi2_sf(chi2, n_bins - 1)
            fired += p < 0.05
            p_values.append(p)
            devs.append(float(np.abs(count_b / per_bin - pi).max()))
        rows.append(
            {
                "delta": delta,
                "shifted_bins": k,
                "detected_pct": 100.0 * fired / trials,
                "median_p": float(np.median(p_values)),
                "median_max_bin_dev": float(np.median(devs)),
            }
        )
    return pl.DataFrame(rows)
