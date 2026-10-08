"""Uniformity and correctness tests for the relocation null sampler.

Run with ``pytest test_relocation.py`` or directly with ``python3``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from amyloscope.analysis.relocation import relocate_many, relocate_starts


def enumerate_placements(lengths, protein_length):
    """All valid non-overlapping, order-preserving placements."""
    k = len(lengths)
    out = []
    for starts in itertools.product(range(1, protein_length + 1), repeat=k):
        ok = all(
            starts[i] + lengths[i] <= starts[i + 1] for i in range(k - 1)
        ) and starts[-1] + lengths[-1] - 1 <= protein_length
        if ok:
            out.append(starts)
    return out


@pytest.mark.parametrize(
    "lengths,length",
    [([2, 3], 10), ([4], 9), ([1, 1, 1], 6), ([3, 3], 12)],
)
def test_uniform(lengths, length):
    """Sampler hits every valid placement with equal probability."""
    valid = enumerate_placements(lengths, length)
    rng = np.random.default_rng(0)
    n = 4000 * len(valid)
    counts = {v: 0 for v in valid}
    for _ in range(n):
        s = tuple(int(x) for x in relocate_starts(np.array(lengths), length, rng))
        assert s in counts, f"sampler produced invalid placement {s}"
        counts[s] += 1
    obs = np.array([counts[v] for v in valid], dtype=float)
    exp = n / len(valid)
    chi2 = ((obs - exp) ** 2 / exp).sum()
    df = len(valid) - 1
    # 99.9th percentile of chi2(df); generous, this only catches gross bias.
    assert chi2 < df + 5 * np.sqrt(2 * df), f"chi2={chi2:.1f} df={df}"


def test_never_overlaps():
    """No placement ever overlaps, including in crowded proteins."""
    rng = np.random.default_rng(1)
    lengths = np.array([8, 7, 9, 6, 8, 7])  # 45 residues in 50 -- very crowded
    for _ in range(20000):
        s = relocate_starts(lengths, 50, rng)
        ends = s + lengths - 1
        assert (s[1:] > ends[:-1]).all(), f"overlap: {s} / {ends}"
        assert s[0] >= 1 and ends[-1] <= 50


def test_vectorised_matches_scalar():
    """``relocate_many`` has the same distribution as ``relocate_starts``."""
    lengths = np.array([2, 3])
    length = 10
    valid = enumerate_placements(list(lengths), length)
    rng = np.random.default_rng(2)
    many = relocate_many(lengths, length, 200000, rng)
    seen = {tuple(int(x) for x in row) for row in many}
    assert seen <= set(valid)
    assert len(seen) == len(valid), "vectorised sampler misses placements"
    counts = np.zeros(len(valid))
    index = {v: i for i, v in enumerate(valid)}
    for row in many:
        counts[index[tuple(int(x) for x in row)]] += 1
    exp = len(many) / len(valid)
    chi2 = ((counts - exp) ** 2 / exp).sum()
    df = len(valid) - 1
    assert chi2 < df + 5 * np.sqrt(2 * df), f"chi2={chi2:.1f} df={df}"


def test_refuses_impossible():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        relocate_starts(np.array([10, 10]), 15, rng)


def test_exact_fit():
    """Zero slack leaves exactly one placement."""
    rng = np.random.default_rng(4)
    s = relocate_starts(np.array([3, 4, 3]), 10, rng)
    assert list(s) == [1, 4, 8]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
