"""Multiple-comparison correction, and which one to use where.

Two families of test in this package produce many p-values at once, and they
want different corrections:

**Per-protein tests** -- the terminal-bias test run once per protein, the
per-protein domain enrichment. An orthologue sweep makes hundreds of these, and
the question asked of them is *how many proteins show the effect*, not *is this
particular protein special*. That is an estimation question about a set, and
Benjamini-Hochberg is the right control: it bounds the expected proportion of
false positives among the ones called, which is what a count of affected
proteins needs to be honest. Holm would bound the probability of even one false
positive across all of them, which at n = 372 is so conservative that a real and
uniform effect returns almost nothing.

**A small set of pre-specified contrasts** -- a handful of domain categories, or
the paralogue contrasts. Here a false positive anywhere is the thing to avoid,
the family is small, and Holm costs little. That is what ``analysis/domains.py``
already uses for its exploratory categories.

The distinction is not stylistic. With ``group_by="name"`` the domain analysis
goes from two categories to eight or more, and Holm's penalty grows with the
count while BH's does not in the same way; a package that offers only one of
them forces the wrong choice on whichever analysis grew.

Neither correction does anything about **phylogenetic** non-independence. 372
orthologues are not 372 independent tests, and no p-value adjustment makes them
so -- BH assumes the tests are independent or positively dependent, and shared
ancestry violates that in a direction nobody can bound from the p-values alone.
Correcting for multiplicity and reporting a replication count are two different
jobs, and ``analysis/termini.py`` does both deliberately.
"""

from __future__ import annotations

import numpy as np

__all__ = ["benjamini_hochberg", "holm", "n_discoveries"]


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values), input order.

    Step-up: sort ascending, scale each by ``m / rank``, then enforce
    monotonicity from the largest downwards so a q-value is never smaller than
    one above it in the ranking. Results are clipped to 1.

    NaN inputs are propagated rather than dropped: a test that could not be run
    is not a test that passed, and silently excluding it from ``m`` would make
    the correction weaker than the number of comparisons actually attempted.
    The adjustment is computed over the non-NaN entries only, which is the
    conventional treatment, but the caller can see how many were lost.

    >>> benjamini_hochberg([0.01, 0.02, 0.03, 0.04]).round(3)
    array([0.04, 0.04, 0.04, 0.04])
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p.copy()
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    vals = p[finite]
    m = vals.size
    order = np.argsort(vals, kind="stable")
    ranked = vals[order]
    scaled = ranked * m / np.arange(1, m + 1)
    # Monotone from the top down: q_(i) = min(q_(i), q_(i+1)).
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    out[finite] = adjusted
    return out


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, input order.

    Step-down, controlling the family-wise error rate. Use for a small,
    pre-specified family where any false positive is costly; prefer
    :func:`benjamini_hochberg` for a large per-protein family.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p.copy()
    out = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out

    vals = p[finite]
    m = vals.size
    order = np.argsort(vals, kind="stable")
    ranked = vals[order]
    scaled = ranked * (m - np.arange(m))
    scaled = np.maximum.accumulate(scaled)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    out[finite] = adjusted
    return out


def n_discoveries(qvalues, alpha: float = 0.05) -> int:
    """How many adjusted p-values clear ``alpha``. NaNs never count."""
    q = np.asarray(qvalues, dtype=float)
    return int(np.sum(np.isfinite(q) & (q <= alpha)))
