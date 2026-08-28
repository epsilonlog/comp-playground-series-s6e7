"""The unit test is the specification. sklearn is the reference implementation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from s6e7.metric import balanced_accuracy, confusion

RNG = np.random.default_rng(42)


@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("n_rows", [50, 5_000])
def test_matches_sklearn_balanced(n_classes: int, n_rows: int) -> None:
    y_true = RNG.integers(0, n_classes, n_rows)
    y_pred = RNG.integers(0, n_classes, n_rows)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )


@pytest.mark.parametrize("seed", range(10))
def test_matches_sklearn_imbalanced(seed: int) -> None:
    """Skewed priors and a correlated (rather than random) predictor."""
    rng = np.random.default_rng(seed)
    y_true = rng.choice(3, size=4_000, p=[0.70, 0.25, 0.05])
    noise = rng.random(4_000) < 0.4
    y_pred = np.where(noise, rng.integers(0, 3, 4_000), y_true)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )


def test_matches_sklearn_adjusted() -> None:
    y_true = RNG.integers(0, 3, 2_000)
    y_pred = RNG.integers(0, 3, 2_000)
    assert balanced_accuracy(y_true, y_pred, adjusted=True) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred, adjusted=True)
    )


def test_matches_sklearn_string_labels() -> None:
    """The target arrives as strings; scoring must not require pre-encoding."""
    classes = np.array(["high", "low", "moderate"])
    y_true = classes[RNG.integers(0, 3, 1_000)]
    y_pred = classes[RNG.integers(0, 3, 1_000)]
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred)
    )


def test_perfect_prediction() -> None:
    y = RNG.integers(0, 3, 500)
    assert balanced_accuracy(y, y) == pytest.approx(1.0)


def test_always_majority_scores_one_over_n_classes() -> None:
    """The floor. 90% accuracy, 0.333 balanced accuracy."""
    y_true = np.array([0] * 900 + [1] * 60 + [2] * 40)
    y_pred = np.zeros_like(y_true)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(1 / 3)


def test_binary_is_mean_of_sensitivity_and_specificity() -> None:
    #             TN TN TN FP | FN TP TP TP
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    specificity = 3 / 4
    sensitivity = 3 / 4
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((specificity + sensitivity) / 2)


def test_hand_computed_three_class() -> None:
    """recall = 1/2, 2/3, 1/1 -> mean 0.7222..."""
    y_true = np.array([0, 0, 1, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0, 2])
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)


def test_class_absent_from_predictions_scores_zero_recall() -> None:
    """Class 2 is never predicted, so its recall is 0 and it still counts in the mean."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    assert balanced_accuracy(y_true, y_pred) == pytest.approx((1 + 1 + 0) / 3)


def test_class_absent_from_y_true_is_dropped_like_sklearn() -> None:
    """The documented trap: sklearn divides by the number of classes *present in y_true*."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 2])  # class 2 predicted but never true
    ours = balanced_accuracy(y_true, y_pred)
    with pytest.warns(UserWarning):
        theirs = balanced_accuracy_score(y_true, y_pred)
    assert ours == pytest.approx(theirs)
    assert ours == pytest.approx((1 / 2 + 1 / 2) / 2)  # divided by 2, not 3


def test_labels_argument_pins_the_class_set() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    assert balanced_accuracy(y_true, y_pred, labels=[0, 1, 2]) == pytest.approx(
        balanced_accuracy(y_true, y_pred)
    )


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        balanced_accuracy(np.array([0, 1, 2]), np.array([0, 1]))


def test_confusion_shape_and_totals() -> None:
    y_true = RNG.integers(0, 4, 300)
    y_pred = RNG.integers(0, 4, 300)
    matrix, labels = confusion(y_true, y_pred)
    assert matrix.shape == (4, 4)
    assert labels.tolist() == [0, 1, 2, 3]
    assert matrix.sum() == 300
