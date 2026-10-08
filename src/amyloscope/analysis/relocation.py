"""Exact uniform relocation null for within-protein region permutation.

The one relocation null for the package. ``positional_permutation.py`` and
``analysis/domains.py`` both draw from it, so the two positional tests rest on
one sampler rather than two implementations that drift.

Why replace the existing loop
-----------------------------

The current sampler is::

    for _attempt in range(1000):
        start = int(rng.integers(1, length - w + 1)) if length - w >= 1 else 1
        residues = set(range(start, start + w + 1))
        if placed.isdisjoint(residues):
            placed |= residues
            positions.append(_normalized_midpoint(start, start + w, length))
            break
    else:
        positions.append(_normalized_midpoint(start, start + w, length))

Three problems:

1. **The ``else`` branch records an overlapping placement.** After 1000 failed
   attempts the last rejected ``start`` is appended anyway and ``placed`` is
   never updated, so the region overlaps its neighbours and every subsequent
   region in that protein is sampled against a stale occupancy set. It fails
   silently and more often exactly where it matters most -- a crowded protein,
   which is the one carrying the strongest positional signal.

2. **Sequential conditional placement is not uniform.** Placing region 1
   uniformly, then region 2 uniformly *given* region 1, does not give a uniform
   draw over non-overlapping configurations; it over-weights configurations
   reachable by many orderings. The bias is towards spread-out arrangements,
   which is the direction that makes an observed clustered arrangement look
   more extreme than it is.

3. **Throughput, modestly.** Measured on a 621-residue protein with five
   regions, 20,000 draws: 0.35 s for the existing rejection loop, 0.14 s for
   ``relocate_many`` -- about 2x, not the order of magnitude one might expect,
   because the vectorised form draws ``free + k`` random numbers per draw where
   the loop draws ``k``. Speed is a side benefit here; correctness is the
   reason to switch. Note that the module's ``DEFAULT_DRAWS = 200_000`` over
   372 proteins is still many minutes of work either way, and the FXR1
   re-analysis used 2,000 draws, which bounds the p-value at 5e-4.

The construction used here
--------------------------

For ``k`` regions of total length ``S`` in a chain of length ``L``, a
non-overlapping placement is determined by cumulative gap offsets
``0 <= o_1 <= ... <= o_k <= free`` with ``free = L - S``. That is a
``k``-multiset over ``free + 1`` values, and multisets map bijectively to
``k``-subsets of ``free + k`` by adding ``0..k-1``. Sampling a subset without
replacement and subtracting the staircase therefore gives an exactly uniform
placement in one shot, with no rejection and no failure mode.

Note that sorting ``k`` i.i.d. uniforms is *not* equivalent: that samples
orderings, not multisets, and weights each multiset by its number of
permutations.

Verification
------------

``test_relocation.py`` enumerates every valid placement for a small case and
checks the sampler against it with a chi-square goodness-of-fit test.
"""

from __future__ import annotations

import numpy as np

__all__ = ["relocate_starts", "relocate_many"]


def relocate_starts(
    lengths: np.ndarray, protein_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Uniformly random non-overlapping placement of regions of ``lengths``.

    Parameters
    ----------
    lengths:
        Region lengths in residues, 1-based inclusive convention (a region
        ``(start, end)`` has length ``end - start + 1``). Order is preserved:
        the i-th length is placed in the i-th slot along the chain.
    protein_length:
        Chain length in residues.
    rng:
        Source of randomness.

    Returns
    -------
    np.ndarray
        1-based start positions, ascending, guaranteed non-overlapping and
        inside ``[1, protein_length]``.

    Raises
    ------
    ValueError
        If the regions cannot fit in the chain. The caller should decide what
        that means; silently returning an overlapping layout is what the
        previous implementation did and is never the right answer.
    """
    lengths = np.asarray(lengths, dtype=int)
    k = lengths.size
    if k == 0:
        return np.empty(0, dtype=int)
    total = int(lengths.sum())
    free = int(protein_length) - total
    if free < 0:
        raise ValueError(
            f"cannot place {k} region(s) totalling {total} residues "
            f"in a chain of {protein_length}"
        )
    chosen = rng.choice(free + k, size=k, replace=False)
    chosen.sort()
    offsets = chosen - np.arange(k)
    starts = offsets + np.concatenate(([0], np.cumsum(lengths[:-1]))) + 1
    return starts.astype(int)


def relocate_many(
    lengths: np.ndarray, protein_length: int, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """``draws`` independent placements, shape ``(draws, len(lengths))``.

    Vectorised over draws, which buys roughly 2x over calling
    :func:`relocate_starts` in a loop -- the partition is over ``free + k``
    keys per draw, so the saving is real but bounded. Use it when the same
    protein is resampled many times; the distribution is identical.
    """
    lengths = np.asarray(lengths, dtype=int)
    k = lengths.size
    if k == 0:
        return np.empty((draws, 0), dtype=int)
    total = int(lengths.sum())
    free = int(protein_length) - total
    if free < 0:
        raise ValueError(
            f"cannot place {k} region(s) totalling {total} residues "
            f"in a chain of {protein_length}"
        )
    # k-subsets of (free + k) without replacement, per draw: take the rank of
    # random keys, which is equivalent to a uniform subset and vectorises.
    keys = rng.random((draws, free + k))
    chosen = np.argpartition(keys, k - 1, axis=1)[:, :k]
    chosen.sort(axis=1)
    offsets = chosen - np.arange(k)
    base = np.concatenate(([0], np.cumsum(lengths[:-1]))) + 1
    return (offsets + base).astype(int)
