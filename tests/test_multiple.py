"""Multiple-comparison correction, checked against an independent implementation."""

from __future__ import annotations

import numpy as np
import pytest

from amyloscope.analysis.multiple import benjamini_hochberg, holm, n_discoveries


# --------------------------------------------------------------------------- #
# Correctness against scipy
# --------------------------------------------------------------------------- #
def test_bh_matches_scipy_on_random_families() -> None:
    """Agreement to machine precision over 200 families of varied shape."""
    from scipy.stats import false_discovery_control

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(1, 200))
        p = np.clip(rng.random(n) ** rng.uniform(1, 4), 1e-12, 1)
        worst = max(
            worst,
            float(np.abs(benjamini_hochberg(p) - false_discovery_control(p, method="bh")).max()),
        )
    assert worst < 1e-12


def test_bh_worked_example() -> None:
    """Benjamini & Hochberg (1995) table 1."""
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216]
    q = benjamini_hochberg(p)
    assert q == pytest.approx(
        [0.01, 0.04, 0.084, 0.084, 0.084, 0.10, 0.105714, 0.216, 0.216, 0.216],
        abs=1e-5,
    )


def test_holm_worked_example() -> None:
    assert holm([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])


# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #
def test_bh_is_monotone_in_the_ranking() -> None:
    rng = np.random.default_rng(1)
    p = rng.random(500)
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_bh_is_never_smaller_than_the_raw_p() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(200)
    assert np.all(benjamini_hochberg(p) >= p - 1e-12)


def test_bh_is_never_larger_than_holm() -> None:
    """BH is the less conservative of the two, always."""
    rng = np.random.default_rng(3)
    p = rng.random(200)
    assert np.all(benjamini_hochberg(p) <= holm(p) + 1e-12)


def test_single_test_is_unchanged() -> None:
    assert benjamini_hochberg([0.03]) == pytest.approx([0.03])
    assert holm([0.03]) == pytest.approx([0.03])


def test_everything_is_clipped_to_one() -> None:
    assert np.all(benjamini_hochberg([0.9, 0.95, 0.99]) <= 1.0)
    assert np.all(holm([0.9, 0.95, 0.99]) <= 1.0)


def test_input_order_is_preserved() -> None:
    """Shuffling the input shuffles the output identically.

    Not asserted via argmin on a near-uniform family: BH ties the adjusted
    values of a run of similar p's, so the minimum is not unique there and the
    test would pass or fail on tie-breaking rather than on ordering.
    """
    p = np.array([0.5, 0.001, 0.3, 0.2])
    q = benjamini_hochberg(p)
    assert np.argmin(q) == 1

    perm = np.array([2, 0, 3, 1])
    assert benjamini_hochberg(p[perm]) == pytest.approx(q[perm])


# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #
def test_empty_input() -> None:
    assert benjamini_hochberg([]).shape == (0,)
    assert holm([]).shape == (0,)


def test_nan_propagates_and_does_not_join_the_family() -> None:
    """A test that could not be run is not a test that passed."""
    q = benjamini_hochberg([0.01, np.nan, 0.5])
    assert np.isnan(q[1])
    # m = 2, not 3: the NaN is excluded from the correction itself.
    assert q[0] == pytest.approx(0.02)


def test_all_nan() -> None:
    assert np.all(np.isnan(benjamini_hochberg([np.nan, np.nan])))


def test_n_discoveries_ignores_nan() -> None:
    assert n_discoveries([0.01, np.nan, 0.04, 0.9], 0.05) == 2
    assert n_discoveries([], 0.05) == 0
