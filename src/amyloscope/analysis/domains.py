"""Domain-overlap analysis.

Quantifies how consensus aggregation-prone regions distribute across a protein's
annotated architecture. The biological question is whether aggregation
determinants are depleted from functional modules — RNA/DNA-binding interfaces,
catalytic cores, conserved folds — as expected if selection disfavours
aggregation in regions whose misfolding would be catastrophic, while tolerating
APRs in disordered linkers and dispensable extensions.

This module computes overlap statistics only; rendering lives in
``amyloscope.viz.domains``, keeping the numeric test reproducible independently
of any figure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..config import PipelineConfig
from ..core.consensus import ConsensusResult


@dataclass
class ProteinDomainOverlap:
    """Per-protein APR/domain overlap accounting (residue counts)."""

    protein: str
    total_apr_aa: int = 0
    in_domain_aa: int = 0
    outside_domain_aa: int = 0
    per_domain_aa: dict[str, int] = field(default_factory=dict)

    @property
    def domain_fraction(self) -> float:
        if self.total_apr_aa == 0:
            return 0.0
        return self.in_domain_aa / self.total_apr_aa


@dataclass
class DomainOverlapResult:
    config: PipelineConfig
    per_protein: dict[str, ProteinDomainOverlap] = field(default_factory=dict)

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


def compute_domain_overlap(result: ConsensusResult) -> DomainOverlapResult:
    """Accumulate APR residues falling inside vs. outside annotated domains."""
    config = result.config
    out = DomainOverlapResult(config=config)

    for protein_id, regions in result.regions.items():
        spec = config.protein(protein_id)
        acc = ProteinDomainOverlap(protein=protein_id)
        per_domain: dict[str, int] = defaultdict(int)

        for region in regions:
            acc.total_apr_aa += region.length
            covered_by_any = False
            for domain in spec.domains:
                ov_start = max(region.start, domain.start)
                ov_end = min(region.end, domain.stop)
                if ov_start <= ov_end:
                    overlap_aa = ov_end - ov_start + 1
                    per_domain[domain.name] += overlap_aa
                    acc.in_domain_aa += overlap_aa
                    covered_by_any = True
            if not covered_by_any:
                acc.outside_domain_aa += region.length

        acc.per_domain_aa = dict(per_domain)
        out.per_protein[protein_id] = acc
    return out


def format_overlap_report(result: DomainOverlapResult) -> str:
    """Plain-text report on the functional-constraint hypothesis."""
    lines = [
        "=" * 70,
        "DOMAIN / APR OVERLAP ANALYSIS",
        "=" * 70,
        "H0: consensus APRs are distributed independently of domain annotation",
        "H1: consensus APRs are depleted within functional domains",
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
        lines.append(f"  In domains:    {acc.in_domain_aa} ({frac:.1f}%)")
        lines.append(f"  Outside:       {acc.outside_domain_aa} ({100 - frac:.1f}%)")
        if acc.per_domain_aa:
            lines.append("  By domain:")
            for name, aa in sorted(acc.per_domain_aa.items()):
                lines.append(f"    {name}: {aa} aa")

    total = result.total_apr_aa
    in_dom = result.total_in_domain_aa
    frac = result.global_domain_fraction
    lines += [
        "",
        "GLOBAL",
        "-" * 40,
        f"Total APR residues: {total}",
        (
            f"Within domains:  {in_dom} ({frac * 100:.1f}%)"
            if total
            else "Within domains: n/a"
        ),
        f"Outside domains: {result.total_outside_domain_aa} "
        f"({(1 - frac) * 100:.1f}%)" if total else "Outside domains: n/a",
        "",
        "INTERPRETATION",
        "-" * 40,
        _interpret_overlap(frac, total),
        "",
        "=" * 70,
    ]
    return "\n".join(lines)


def _interpret_overlap(domain_fraction: float, total_apr_aa: int) -> str:
    if total_apr_aa == 0:
        return "No consensus APRs were called; overlap analysis is not applicable."
    if domain_fraction < 0.4:
        return (
            "APRs are markedly depleted from annotated functional domains, "
            "consistent with selection against aggregation in regions whose "
            "misfolding would compromise the fold or its binding interfaces; "
            "APRs instead localise to disordered/linker/terminal regions of "
            "lower functional cost."
        )
    if domain_fraction > 0.6:
        return (
            "APRs are enriched within functional domains, which may indicate "
            "structurally constrained aggregation determinants embedded in the "
            "fold, or regulated assembly surfaces, rather than dispensable "
            "low-complexity segments."
        )
    return (
        "APRs show no strong domain bias; aggregation determinants are spread "
        "across the architecture, compatible with tolerance of dispersed APRs "
        "that do not concentrate catastrophic risk in any single module."
    )
