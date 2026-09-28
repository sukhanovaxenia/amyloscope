"""Permutation test for positional enrichment of consensus regions.

Replaces the decile chi-square in ``analysis/statistics.py`` (``sc_stats.chisquare``
against a flat ``len(positions)/10`` expectation). It reuses the *same* null model
already used in ``analysis/domain_overlap.py`` — whole regions relocated uniformly
within their own protein, lengths preserved, Phipson & Smyth +1 correction — so the
two positional tests in the package finally rest on one null instead of two.

Why the chi-square version is wrong here, on two counts:

1. **The asymptotic p is invalid.** ``scipy.stats.chisquare`` uses the asymptotic
   chi-square distribution, which needs an expected count of ~5 per cell (Cochran).
   With a typical panel the decile expectation is ``n/10`` — 0.9 for nine regions —
   so the reported p is not trustworthy. The figure already flags this
   (``underpowered = expected < 5.0``); ``statistics.py`` does not, and it drives
   ``positional_distribution`` and ``enrichment_pattern`` off that invalid p.

2. **The flat ``n/10`` expectation is itself wrong.** A region has width, and its
   midpoint therefore cannot reach the extreme termini: a width-``w`` region in a
   protein of length ``L`` confines its normalized midpoint to
   ``[(w-1)/2(L-1), 1-(w-1)/2(L-1)]``. Pooling such regions, the outer deciles are
   under-accessible, so a uniform reference over-expects the tails and inflates
   chi-square. The relocation null reproduces the *correct* non-flat expected
   decile profile automatically, because it places real-width regions in
   real-length proteins.

The statistic stays the decile chi-square *distance* from uniform, so the figure
and the reported number keep their meaning; only its significance now comes from
the permutation distribution. Pseudoreplication (several regions from one protein
are not independent) is handled by construction, because each region is relocated
only within its own protein.
"""

from __future__ import annotations

import numpy as np

#: Matches domain_overlap.DEFAULT_DRAWS resolution.
DEFAULT_DRAWS = 200_000


def _normalized_midpoint(start: int, end: int, length: int) -> float:
    if length <= 1:
        return 0.0
    return ((start + end) / 2 - 1.0) / (length - 1.0)


def _decile_counts(positions: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(positions, bins=np.linspace(0, 1, 11))
    return counts.astype(float)


def _chi2_distance(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0:
        return 0.0
    expected = n / 10
    return float(np.sum((counts - expected) ** 2 / expected))


def positional_permutation_test(
    regions_by_protein: dict[str, list[tuple[int, int]]],
    lengths: dict[str, int],
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> dict:
    """Permutation test for non-uniform normalized-midpoint placement.

    ``regions_by_protein`` maps protein id -> list of ``(start, end)`` (1-based
    inclusive). ``lengths`` maps protein id -> chain length. Returns the observed
    decile counts, the chi-square distance statistic, the permutation p-value, and
    the null's mean decile profile (the correct, non-flat "expected uniform" line
    for the figure).
    """
    rng = np.random.default_rng(seed)

    observed_positions = np.array(
        [
            _normalized_midpoint(s, e, lengths[p])
            for p, regs in regions_by_protein.items()
            for s, e in regs
        ]
    )
    n = observed_positions.size
    if n == 0:
        return {"n_regions": 0, "p_value": float("nan")}

    observed_counts = _decile_counts(observed_positions)
    observed_stat = _chi2_distance(observed_counts)

    # Per-protein region widths; relocation preserves width and stays within the
    # protein, so each draw is a valid alternative placement of the same regions.
    widths = {p: [e - s for s, e in regs] for p, regs in regions_by_protein.items()}

    at_least_as_extreme = 0
    null_decile_sum = np.zeros(10, dtype=float)
    for _ in range(draws):
        positions = []
        for p, ws in widths.items():
            length = lengths[p]
            placed: set[int] = set()
            for w in ws:
                # Resample this region until it does not overlap the others in
                # the same protein, matching domain_overlap's non-overlap null;
                # observed consensus regions are themselves non-overlapping.
                for _attempt in range(1000):
                    start = int(rng.integers(1, length - w + 1)) if length - w >= 1 else 1
                    residues = set(range(start, start + w + 1))
                    if placed.isdisjoint(residues):
                        placed |= residues
                        positions.append(_normalized_midpoint(start, start + w, length))
                        break
                else:
                    positions.append(_normalized_midpoint(start, start + w, length))
        counts = _decile_counts(np.array(positions))
        null_decile_sum += counts
        if _chi2_distance(counts) >= observed_stat - 1e-12:
            at_least_as_extreme += 1

    p_value = (at_least_as_extreme + 1) / (draws + 1)
    return {
        "n_regions": int(n),
        "n_proteins": len(regions_by_protein),
        "decile_counts": observed_counts.astype(int).tolist(),
        "chi2_distance": observed_stat,
        "p_value": p_value,
        "expected_decile_profile": (null_decile_sum / draws).tolist(),
        "draws": draws,
        "method": "permutation (whole-region relocation within parent protein)",
    }
