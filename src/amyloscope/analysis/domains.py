"""Domain-overlap analysis.

Quantifies how consensus aggregation-prone regions distribute across a protein's
annotated architecture. The biological question is whether aggregation
determinants concentrate on particular functional modules — RNA/DNA-binding
interfaces, catalytic cores, disordered extensions — or fall independently of
the fold.

Three corrections relative to the previous implementation, each of which changed
a reported number:

**Residues, not per-domain intersections.** ``in_domain_aa`` was accumulated
inside the loop over domains, so a residue covered by two annotations was added
twice. Domains routinely nest — eS2's uS5 core (51-268) contains its
dsRNA-binding domain (96-167) entirely — and the result was 64 "in-domain"
residues against 43 real ones, printed as 148.8 %. Counting is now done on
residue sets, so the in-domain and outside counts partition the APR residues by
construction and no percentage can exceed 100.

**Outside means outside.** A region was counted as outside only when it touched
no domain at all, so a region straddling a boundary contributed its overlap to
the in-domain total and its remainder to nothing. The two counters were then
printed as complements. They now are complements.

**A fraction is not a test.** The previous report classified the in-domain
fraction against fixed thresholds (below 0.4 "depleted", above 0.6 "enriched")
and stated a conclusion. That fraction is uninterpretable without knowing how
much of the protein the annotation covers: eS2's domains span 79.9 % of its
length and eL27's 96.3 %, so 100 % of APR residues falling inside them is close
to what uniform placement gives (p = 0.22). Enrichment is now assessed against a
null in which whole regions are relocated uniformly with their lengths
preserved. Relocating whole regions rather than residues matters: APR residues
arrive in contiguous runs, and a residue-level binomial that ignores this
returns p < 0.0001 for the same data where the region-level null returns 0.017.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..config import PipelineConfig
from ..core.consensus import ConsensusResult
from .per_protein import ProteinDomainEnrichment, per_protein_domain_enrichment

#: Placements drawn for the relocation null. 200 000 resolves a two-sided p to
#: about 1e-5, well below anything this design can claim.
DEFAULT_DRAWS = 200_000


@dataclass
class CategoryEnrichment:
    """Observed vs expected APR residues for one domain category."""

    category: str
    covered_aa: int
    total_aa: int
    observed_aa: int
    expected_aa: float
    p_two_sided: float
    p_greater: float
    p_holm: float = float("nan")

    @property
    def coverage(self) -> float:
        return self.covered_aa / self.total_aa if self.total_aa else 0.0

    @property
    def fold(self) -> float:
        return self.observed_aa / self.expected_aa if self.expected_aa else float("nan")


@dataclass
class ProteinDomainOverlap:
    """Per-protein APR/domain accounting, on residue sets."""

    protein: str
    total_apr_aa: int = 0
    in_domain_aa: int = 0
    outside_domain_aa: int = 0
    #: Per-domain intersections. These overlap by construction when the
    #: annotations do, so they sum to more than in_domain_aa and must never be
    #: presented as a partition.
    per_domain_aa: dict[str, int] = field(default_factory=dict)
    per_category_aa: dict[str, int] = field(default_factory=dict)

    @property
    def domain_fraction(self) -> float:
        if self.total_apr_aa == 0:
            return 0.0
        return self.in_domain_aa / self.total_apr_aa


@dataclass
class DomainOverlapResult:
    config: PipelineConfig
    per_protein: dict[str, ProteinDomainOverlap] = field(default_factory=dict)
    enrichment: list[CategoryEnrichment] = field(default_factory=list)
    primary_category: Optional[str] = None
    draws: int = DEFAULT_DRAWS
    null_hist: dict[str, list[int]] = field(default_factory=dict)
    #: Per-protein enrichment for the primary category, Holm-corrected across
    #: proteins. Empty when no primary category is declared — the per-protein
    #: test operationalises the directional hypothesis, so it is not run for
    #: exploratory categories. This is what shows WHICH proteins drive the
    #: pooled signal and which are carried by domain coverage.
    per_protein_enrichment: list[ProteinDomainEnrichment] = field(default_factory=list)

    # sum(), not len(). The previous versions called len() on a generator
    # expression, which raises TypeError and would abort the report.
    @property
    def total_apr_aa(self) -> int:
        return sum(o.total_apr_aa for o in self.per_protein.values())

    @property
    def total_in_domain_aa(self) -> int:
        return sum(o.in_domain_aa for o in self.per_protein.values())

    @property
    def total_outside_domain_aa(self) -> int:
        return sum(o.outside_domain_aa for o in self.per_protein.values())

    @property
    def global_domain_fraction(self) -> float:
        total = self.total_apr_aa
        return self.total_in_domain_aa / total if total else 0.0


# --------------------------------------------------------------------------- #
# Residue-set helpers
# --------------------------------------------------------------------------- #
def _residues(intervals: Iterable[tuple[int, int]]) -> set[int]:
    out: set[int] = set()
    for start, stop in intervals:
        out |= set(range(start, stop + 1))
    return out


def _category_residues(spec, category: str) -> set[int]:
    return _residues((d.start, d.stop) for d in spec.domains
                     if d.category == category)


def _holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, float)
    m = p.size
    adj = np.empty(m, float)
    running = 0.0
    for rank, idx in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return list(adj)


# --------------------------------------------------------------------------- #
def compute_domain_overlap(
    result: ConsensusResult,
    primary_category: Optional[str] = None,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    per_protein_draws: int = 50_000,   # per protein x this, so smaller than draws
) -> DomainOverlapResult:
    """APR residues inside vs outside domains, plus per-category enrichment.

    ``primary_category`` names the one category the study was designed to test —
    ``rna_binding`` for the ribosome-masking hypothesis. It is reported with a
    one-sided p, because the hypothesis is directional and was stated before the
    analysis; every other category is exploratory, reported two-sided and
    Holm-corrected. Declaring which is which is what keeps the primary result
    from being one of several tests presented as though it were the only one.
    """
    config = result.config
    if primary_category is None:
        # Declared in the config beside `hypothesis`, which is where the
        # directional claim it operationalises already lives.
        primary_category = getattr(config, "primary_domain_category", None)
    out = DomainOverlapResult(config=config, primary_category=primary_category,
                              draws=draws)

    apr_by_protein: dict[str, set[int]] = {}
    for protein_id, regions in result.regions.items():
        if not regions:
            continue
        spec = config.protein(protein_id)
        apr = _residues((r.start, r.end) for r in regions)
        apr_by_protein[protein_id] = apr

        union = _residues((d.start, d.stop) for d in spec.domains)
        acc = ProteinDomainOverlap(
            protein=protein_id,
            total_apr_aa=len(apr),
            in_domain_aa=len(apr & union),
            outside_domain_aa=len(apr - union),
        )
        per_domain: dict[str, int] = defaultdict(int)
        for domain in spec.domains:
            n = len(apr & set(range(domain.start, domain.stop + 1)))
            if n:
                per_domain[domain.name] = n
        acc.per_domain_aa = dict(per_domain)
        acc.per_category_aa = {
            c: len(apr & _category_residues(spec, c))
            for c in sorted({d.category for d in spec.domains})
        }
        out.per_protein[protein_id] = acc

    if apr_by_protein:
        out.enrichment, out.null_hist = _enrichment(
            result, apr_by_protein, draws, seed)
        exploratory = [e for e in out.enrichment if e.category != primary_category]
        for e, q in zip(exploratory, _holm([x.p_two_sided for x in exploratory])):
            e.p_holm = q
        for e in out.enrichment:
            if e.category == primary_category:
                e.p_holm = e.p_greater      # pre-specified, one-sided, m = 1
                # Per-protein enrichment for the pre-specified category only: it
        # operationalises the directional hypothesis, and running it for every
        # exploratory category x every protein would be a multiplicity thicket.
        if primary_category is not None:
            regions_by_protein = {
                p: [(r.start, r.end) for r in result.regions[p]]
                for p in apr_by_protein
            }
            lengths = {
                p: (config.protein(p).length or max(apr_by_protein[p]))
                for p in apr_by_protein
            }
            cat_res = {
                p: _category_residues(config.protein(p), primary_category)
                for p in apr_by_protein
            }
            out.per_protein_enrichment = per_protein_domain_enrichment(
                regions_by_protein, lengths, cat_res, primary_category,
                draws=per_protein_draws, seed=seed,
            )
    return out


def _enrichment(result: ConsensusResult, apr_by_protein: dict[str, set[int]],
                draws: int, seed: int
                ) -> tuple[list[CategoryEnrichment], dict[str, list[int]]]:
    """Relocation null: whole regions, uniform position, lengths preserved."""
    config = result.config
    rng = np.random.default_rng(seed)
    pids = list(apr_by_protein)
    specs = {p: config.protein(p) for p in pids}
    lengths = {p: (specs[p].length or max(apr_by_protein[p]))
               for p in pids}
    region_lengths = {p: [r.end - r.start + 1 for r in result.regions[p]]
                      for p in pids}
    categories = sorted({d.category for p in pids for d in specs[p].domains})

    cat_res = {c: {p: _category_residues(specs[p], c) for p in pids}
               for c in categories}
    observed = {c: sum(len(apr_by_protein[p] & cat_res[c][p]) for p in pids)
                for c in categories}

    # Placements are resampled until the regions of a protein do not overlap
    # one another, so every draw covers exactly as many residues as the observed
    # set. Allowing overlaps would let a draw cover fewer residues than the
    # observation it is compared against, biasing the null low and the
    # enrichment high. With six 6-11 residue regions in 293 positions a pair
    # collides in roughly 4 % of attempts, which is far too often to ignore.
    null = {c: np.empty(draws, int) for c in categories}
    want = {p: sum(region_lengths[p]) for p in pids}
    for i in range(draws):
        placed = {}
        for p in pids:
            N = lengths[p]
            for _ in range(1000):
                s = set()
                for L in region_lengths[p]:
                    st = int(rng.integers(1, N - L + 2))
                    s |= set(range(st, st + L))
                if len(s) == want[p]:
                    break
            placed[p] = s
        for c in categories:
            null[c][i] = sum(len(placed[p] & cat_res[c][p]) for p in pids)

    total_aa = sum(lengths[p] for p in pids)
    rows = []
    for c in categories:
        obs = observed[c]
        nl = null[c]
        # +1 correction: a permutation p is never zero (Phipson & Smyth 2010).
        p_hi = (np.sum(nl >= obs) + 1) / (draws + 1)
        p_lo = (np.sum(nl <= obs) + 1) / (draws + 1)
        rows.append(CategoryEnrichment(
            category=c,
            covered_aa=sum(len(cat_res[c][p]) for p in pids),
            total_aa=total_aa,
            observed_aa=obs,
            expected_aa=float(nl.mean()),
            p_two_sided=float(min(1.0, 2 * min(p_hi, p_lo))),
            p_greater=float(p_hi),
        ))
    hist = {c: np.bincount(null[c]).tolist() for c in categories}
    return rows, hist


def format_overlap_report(result: DomainOverlapResult) -> str:
    """Plain-text report on the domain-placement hypothesis."""
    primary = result.primary_category
    lines = [
        "=" * 70,
        "DOMAIN / APR OVERLAP ANALYSIS",
        "=" * 70,
        "H0: consensus APRs are placed independently of domain annotation",
        f"H1: consensus APRs are enriched within {primary or 'a named category'}",
        "",
        "Counts are residue sets: a residue covered by two annotations is",
        "counted once. Per-domain intersections below overlap and are reported",
        "for orientation only; they do not partition the APR residues.",
        "",
        "PER-PROTEIN",
        "-" * 40,
    ]
    for protein_id, acc in result.per_protein.items():
        if acc.total_apr_aa == 0:
            continue
        frac = acc.domain_fraction * 100
        label = result.config.label_for(protein_id)
        lines.append(f"\n{label} ({protein_id}):")
        lines.append(f"  Total APR residues: {acc.total_apr_aa}")
        lines.append(f"  In any domain: {acc.in_domain_aa} ({frac:.1f}%)")
        lines.append(f"  Outside:       {acc.outside_domain_aa} ({100 - frac:.1f}%)")
        if acc.per_domain_aa:
            lines.append("  By domain (overlapping counts):")
            for name, aa in sorted(acc.per_domain_aa.items()):
                lines.append(f"    {name}: {aa} aa")

    total = result.total_apr_aa
    lines += [
        "",
        "GLOBAL",
        "-" * 40,
        f"Total APR residues: {total}",
        f"Within domains:  {result.total_in_domain_aa} "
        f"({result.global_domain_fraction * 100:.1f}%)",
        f"Outside domains: {result.total_outside_domain_aa} "
        f"({(1 - result.global_domain_fraction) * 100:.1f}%)",
        "",
        "ENRICHMENT BY DOMAIN CATEGORY",
        "-" * 40,
        f"Null: whole regions relocated uniformly, lengths preserved "
        f"({result.draws:,} placements)",
        "",
        f"{'category':18s}{'coverage':>12s}{'obs':>6s}{'exp':>8s}"
        f"{'fold':>7s}{'p':>9s}{'p_adj':>9s}",
    ]
    for e in result.enrichment:
        tag = " (primary, one-sided)" if e.category == primary else ""
        p_raw = e.p_greater if e.category == primary else e.p_two_sided
        lines.append(
            f"{e.category:18s}{f'{100 * e.coverage:.0f}%':>12s}{e.observed_aa:6d}"
            f"{e.expected_aa:8.1f}{e.fold:7.2f}{p_raw:9.4f}{e.p_holm:9.4f}{tag}"
        )
    if result.per_protein_enrichment:
        lines += [
            "",
            f"PER-PROTEIN ENRICHMENT ({result.primary_category}, one-sided)",
            "-" * 40,
            "Which proteins drive the pooled signal. 'forced' marks a domain",
            "covering >=90% of the protein, where 'in domain' is inevitable and",
            "carries no enrichment information.",
            "",
            f"{'protein':16s}{'n':>3s}{'cover':>8s}{'obs':>6s}{'exp':>8s}"
            f"{'fold':>7s}{'p':>9s}{'p_adj':>9s}",
        ]
        for e in sorted(result.per_protein_enrichment, key=lambda r: r.p_greater):
            label = result.config.label_for(e.protein)
            flag = "  forced" if e.coverage_forced else ""
            lines.append(
                f"{label[:16]:16s}{e.n_regions:3d}{e.coverage * 100:7.0f}%"
                f"{e.observed_in_domain:6d}{e.expected_in_domain:8.1f}"
                f"{e.fold:7.2f}{e.p_greater:9.4f}{e.p_holm:9.4f}{flag}"
            )
    lines += ["", "INTERPRETATION", "-" * 40,
              _interpret(result), "", "=" * 70]
    return "\n".join(lines)


def _interpret(result: DomainOverlapResult) -> str:
    if result.total_apr_aa == 0:
        return "No consensus APRs were called; overlap analysis is not applicable."
    frac = result.global_domain_fraction
    primary = next((e for e in result.enrichment
                    if e.category == result.primary_category), None)
    parts = [
        f"{result.total_in_domain_aa} of {result.total_apr_aa} APR residues "
        f"({frac * 100:.1f}%) lie within an annotated domain. This figure is not "
        f"interpretable on its own: it depends on how much of each sequence the "
        f"annotation covers, and is reported here only for completeness."
    ]
    if primary is None:
        parts.append(
            "No primary category was declared, so every category below is "
            "exploratory and Holm-corrected; none of them supports a directional "
            "claim on its own."
        )
    elif primary.p_greater < 0.05:
        parts.append(
            f"Against the relocation null, APR residues are enriched "
            f"{primary.fold:.2f}-fold within {primary.category} domains "
            f"({primary.observed_aa} observed vs {primary.expected_aa:.1f} "
            f"expected, one-sided p = {primary.p_greater:.4f}). This is the "
            f"pre-specified test; remaining categories are exploratory."
        )
    else:
        parts.append(
            f"Against the relocation null, enrichment within {primary.category} "
            f"domains is {primary.fold:.2f}-fold and does not reach significance "
            f"(one-sided p = {primary.p_greater:.4f})."
        )
    drivers = [e for e in result.per_protein_enrichment
               if e.p_holm < 0.05 and not e.coverage_forced]
    if result.per_protein_enrichment:
        if drivers:
            names = ", ".join(result.config.label_for(e.protein) for e in drivers)
            parts.append(
                f"Per-protein, the enrichment is carried by {names} "
                f"(Holm p < 0.05); proteins whose domains span most of their "
                f"length contribute overlap forced by coverage, not enrichment."
            )
        else:
            parts.append(
                "No single protein reaches per-protein significance after Holm "
                "correction; the pooled result rests on aggregated weak signal "
                "rather than one strong protein."
            )
    return "\n".join(parts)