"""Per-protein positional and domain-overlap analysis.

The panel-level tests in ``analysis/statistics.py`` and ``analysis/domain_overlap.py``
pool every protein's regions before testing. Pooling gains power for a single
directional question ("across the panel, do APRs fall in RNA-binding domains?"),
but it hides two things that are often the actual biology:

* **Opposite per-protein patterns cancel.** One protein clustered N-terminal and
  another C-terminal pool to something that looks unbiased — a Simpson's-paradox
  failure. The merged positional histogram cannot show this; per-protein histograms
  do.
* **Coverage-forced overlap masquerades as enrichment, or dilutes it.** A protein
  whose annotated domains span ~95 % of its length has ~100 % of its APR residues
  "in domain" by construction, contributing no enrichment signal. Pooled with a
  protein whose domain is small and APR-capturing, it dilutes the real effect. The
  per-protein test separates the protein that drives the signal from the ones
  carried along by coverage.

Two asymmetries in what is worth reporting per protein:

* **Per-protein DOMAIN OVERLAP is often well-powered**, and it is the one that
  serves a masking hypothesis directly — *which* proteins put their APRs on the
  functional interface. Reported one-sided (the hypothesis is directional),
  Holm-corrected across proteins, with a ``coverage_forced`` flag when the domain
  covers >= 90 % of the protein and "in domain" therefore carries no information.
* **Per-protein POSITIONAL CLUSTERING is weakly powered.** With the typical 3-6
  regions per protein the within-protein permutation has little resolution, so the
  clustering p is a cautious companion to the histogram (returned only at
  >= ``min_regions_for_p`` regions), never a headline.

Everything relocates whole regions within their own protein, lengths preserved,
Phipson & Smyth +1 correction — the same null as ``domain_overlap`` — through the
single :func:`relocate_within_protein` primitive, so the per-protein and
panel-level results cannot rest on different assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DOMAIN_DRAWS = 50_000
DEFAULT_POSITIONAL_DRAWS = 20_000
DEFAULT_MIN_REGIONS_FOR_P = 4
COVERAGE_FORCED_THRESHOLD = 0.90


# --------------------------------------------------------------------------- #
# The single relocation primitive. domain_overlap._enrichment and
# positional_permutation_test each hand-roll this loop; both should call it.
# --------------------------------------------------------------------------- #
def relocate_within_protein(
    rng: np.random.Generator,
    length: int,
    region_lengths: list[int],
    *,
    max_attempts: int = 1000,
) -> list[int]:
    """One non-overlapping relocation of a protein's regions within itself.

    Returns the 1-based start position of each region, in the given order.
    Regions are resampled until they do not overlap one another, so every draw
    covers exactly as many residues as the observed set; allowing overlaps would
    let a draw cover fewer residues than the observation, biasing the null low.
    Falls back to the last attempt if the protein is too dense to place cleanly.
    """
    want = sum(region_lengths)
    starts: list[int] = []
    for _ in range(max_attempts):
        starts, used = [], set()
        for rlen in region_lengths:
            hi = length - rlen + 2
            start = int(rng.integers(1, hi)) if hi > 1 else 1
            starts.append(start)
            used |= set(range(start, start + rlen))
        if len(used) == want:  # no overlaps
            return starts
    return starts


def placed_residues(starts: list[int], region_lengths: list[int]) -> set[int]:
    """Residue set covered by a placement returned by relocate_within_protein."""
    out: set[int] = set()
    for start, rlen in zip(starts, region_lengths, strict=True):
        out |= set(range(start, start + rlen))
    return out


def _holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    m = p.size
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return list(adj)


def _norm_mid(start: int, rlen: int, length: int) -> float:
    if length <= 1:
        return 0.0
    end = start + rlen - 1
    return ((start + end) / 2 - 1.0) / (length - 1.0)


# --------------------------------------------------------------------------- #
@dataclass
class ProteinDomainEnrichment:
    """Per-protein enrichment of APR residues in one domain category."""

    protein: str
    category: str
    n_regions: int
    apr_aa: int
    domain_aa: int
    protein_length: int
    observed_in_domain: int
    expected_in_domain: float
    p_greater: float
    p_holm: float = float("nan")

    @property
    def coverage(self) -> float:
        return self.domain_aa / self.protein_length if self.protein_length else 0.0

    @property
    def fold(self) -> float:
        return (
            self.observed_in_domain / self.expected_in_domain
            if self.expected_in_domain
            else float("nan")
        )

    @property
    def coverage_forced(self) -> bool:
        return self.coverage >= COVERAGE_FORCED_THRESHOLD


@dataclass
class PerProteinPositional:
    """Per-protein normalized-midpoint description (histogram-first)."""

    protein: str
    n_regions: int
    decile_counts: list[int]
    mean_position: float
    clustering_p: float = float("nan")


def per_protein_domain_enrichment(
    regions_by_protein: dict[str, list[tuple[int, int]]],
    lengths: dict[str, int],
    category_residues: dict[str, set[int]],
    category: str,
    *,
    draws: int = DEFAULT_DOMAIN_DRAWS,
    seed: int = 0,
) -> list[ProteinDomainEnrichment]:
    """Per-protein one-sided enrichment of APR residues in ``category``."""
    rng = np.random.default_rng(seed)
    rows: list[ProteinDomainEnrichment] = []

    for pid, regs in regions_by_protein.items():
        length = lengths[pid]
        target = category_residues.get(pid, set())
        region_lengths = [e - s + 1 for s, e in regs]
        apr = set().union(*(set(range(s, e + 1)) for s, e in regs)) if regs else set()

        observed = len(apr & target)
        null = np.empty(draws, dtype=int)
        for i in range(draws):
            starts = relocate_within_protein(rng, length, region_lengths)
            null[i] = len(placed_residues(starts, region_lengths) & target)
        p_greater = (np.sum(null >= observed) + 1) / (draws + 1)

        rows.append(
            ProteinDomainEnrichment(
                protein=pid,
                category=category,
                n_regions=len(regs),
                apr_aa=len(apr),
                domain_aa=len(target),
                protein_length=length,
                observed_in_domain=observed,
                expected_in_domain=float(null.mean()),
                p_greater=float(p_greater),
            )
        )

    for row, q in zip(rows, _holm([r.p_greater for r in rows]), strict=True):
        row.p_holm = q
    return rows


def per_protein_positional(
    regions_by_protein: dict[str, list[tuple[int, int]]],
    lengths: dict[str, int],
    *,
    draws: int = DEFAULT_POSITIONAL_DRAWS,
    seed: int = 0,
    min_regions_for_p: int = DEFAULT_MIN_REGIONS_FOR_P,
) -> list[PerProteinPositional]:
    """Per-protein normalized-midpoint histogram with a cautious clustering p."""
    rng = np.random.default_rng(seed)
    out: list[PerProteinPositional] = []

    for pid, regs in regions_by_protein.items():
        length = lengths[pid]
        region_lengths = [e - s + 1 for s, e in regs]
        mids = np.array([_norm_mid(s, rl, length)
                         for (s, _), rl in zip(regs, region_lengths, strict=True)])
        counts, _ = np.histogram(mids, bins=np.linspace(0, 1, 11))

        clustering_p = float("nan")
        if len(regs) >= min_regions_for_p:
            observed_span = float(mids.max() - mids.min())
            tighter = 0
            for _ in range(draws):
                starts = relocate_within_protein(rng, length, region_lengths)
                nm = np.array([_norm_mid(st, rl, length)
                               for st, rl in zip(starts, region_lengths, strict=True)])
                if (nm.max() - nm.min()) <= observed_span + 1e-12:
                    tighter += 1
            clustering_p = (tighter + 1) / (draws + 1)

        out.append(
            PerProteinPositional(
                protein=pid,
                n_regions=len(regs),
                decile_counts=counts.astype(int).tolist(),
                mean_position=float(mids.mean()) if mids.size else float("nan"),
                clustering_p=clustering_p,
            )
        )
    return out
