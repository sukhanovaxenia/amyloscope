"""Statistical characterisation of consensus regions.

Tests whether aggregation-prone regions are distributed uniformly along the
sequence or concentrated in particular zones. A uniform distribution is the null
expected if APRs are tolerated wherever they arise; significant positional
enrichment implies a structural or functional constraint shaping where
aggregation determinants can persist. The biological reading of either outcome
is supplied by the user via the configured hypothesis, not assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..config import PipelineConfig
from ..core.consensus import ConsensusResult
from ..analysis.positional_permutation import positional_permutation_test
from .per_protein import PerProteinPositional, per_protein_positional

@dataclass
class ConsensusStatistics:
    """Aggregate statistics across all consensus regions in a panel."""

    total_regions: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    tier_percentages: dict[str, float] = field(default_factory=dict)
    mean_length: float = 0.0
    median_length: float = 0.0
    min_length: int = 0
    max_length: int = 0
    classic_apr_count: int = 0  # 5-15 aa steric-zipper-scale spines
    extended_apr_count: int = 0  # > 15 aa
    chi2_statistic: float = 0.0
    chi2_p_value: float = 1.0
    positional_distribution: str = "uniform"
    n_terminal_count: int = 0
    central_count: int = 0
    c_terminal_count: int = 0
    enrichment_pattern: str = "No significant enrichment"
    proteins_with_regions: int = 0
    total_proteins: int = 0
    decile_counts: list[int] = field(default_factory=list)
    expected_decile_profile: list[float] = field(default_factory=list)
    per_protein_positional: list[PerProteinPositional] = field(default_factory=list)

def compute_statistics(result: ConsensusResult) -> ConsensusStatistics:
    """Derive distributional statistics from a consensus result."""
    config: PipelineConfig = result.config
    regions = result.all_regions()
    stats = ConsensusStatistics(
        total_regions=len(regions),
        total_proteins=len(config.proteins),
    )
    if not regions:
        return stats

    # Tier distribution.
    for tier in config.consensus.tiers:
        count = sum(1 for r in regions if r.tier == tier.name)
        stats.tier_counts[tier.name] = count
        stats.tier_percentages[tier.name] = 100.0 * count / len(regions)

    # Length statistics and steric-zipper-scale classification.
    lengths = np.array([r.length for r in regions])
    stats.mean_length = float(lengths.mean())
    stats.median_length = float(np.median(lengths))
    stats.min_length = int(lengths.min())
    stats.max_length = int(lengths.max())
    stats.classic_apr_count = int(np.sum((lengths >= 5) & (lengths <= 15)))
    stats.extended_apr_count = int(np.sum(lengths > 15))

    # Normalised midpoint positions for distributional testing.
    positions = []
    for protein_id, regs in result.regions.items():
        plen = _protein_length(result, protein_id)
        for r in regs:
            positions.append(r.normalized_position(plen))
    positions = np.array([p for p in positions if 0.0 <= p <= 1.0])

    # Permutation test against a uniform distribution.
    regions_by_protein = {
        pid: [(r.start, r.end) for r in regs]
        for pid, regs in result.regions.items() if regs
    }
    lengths = {pid: _protein_length(result, pid) for pid in regions_by_protein}

    perm = positional_permutation_test(regions_by_protein, lengths, seed=0)
    stats.decile_counts = perm["decile_counts"]
    stats.chi2_statistic = perm["chi2_distance"]     # now a distance, not a test stat
    stats.chi2_p_value = perm["p_value"]             # permutation p, valid at any n
    stats.positional_distribution = (
        "uniform" if perm["p_value"] > 0.05 else "non-uniform"
    )
    stats.per_protein_positional = per_protein_positional(
        regions_by_protein, lengths, seed=0
    )
    # Optional but recommended: keep the null's expected profile for the figure, so
    # the "expected uniform" line reflects the achievable (non-flat) distribution.
    stats.expected_decile_profile = perm["expected_decile_profile"]

    # Tertile partition for an interpretable N/centre/C summary.
    stats.n_terminal_count = int(np.sum(positions < 0.33))
    stats.c_terminal_count = int(np.sum(positions > 0.67))
    stats.central_count = int(
        np.sum((positions >= 0.33) & (positions <= 0.67))
    )
    if stats.chi2_p_value < 0.05:
        if stats.n_terminal_count > max(stats.central_count, stats.c_terminal_count):
            stats.enrichment_pattern = "N-terminal enrichment"
        elif stats.c_terminal_count > max(stats.n_terminal_count, stats.central_count):
            stats.enrichment_pattern = "C-terminal enrichment"
        else:
            stats.enrichment_pattern = "Central enrichment"

    stats.proteins_with_regions = sum(
        1 for regs in result.regions.values() if regs
    )
    return stats


def _protein_length(result: ConsensusResult, protein_id: str) -> int:
    spec = result.config.protein(protein_id)
    if spec.length:
        return spec.length
    # Fall back to the furthest region end seen for this protein.
    regs = result.regions.get(protein_id, [])
    return max((r.end for r in regs), default=1)


def format_report(
    stats: ConsensusStatistics, config: PipelineConfig
) -> str:
    """Render a plain-text statistical report.

    The interpretive paragraph is the configured hypothesis when provided;
    otherwise a neutral, distribution-based statement is emitted.
    """
    lines = [
        "=" * 70,
        "CONSENSUS AGGREGATION-PRONE REGION STATISTICAL REPORT",
        "=" * 70,
        f"Project: {config.name}",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Predictors: {', '.join(config.tool_names)} (n={len(config.enabled_tools)})",
        f"Proteins analysed: {stats.total_proteins}",
        f"Proteins with consensus regions: {stats.proteins_with_regions}",
        "",
        "EVIDENCE-TIER DISTRIBUTION",
        "-" * 40,
        f"Total consensus regions: {stats.total_regions}",
        "",
    ]
    for tier in config.consensus.tiers_by_strength:
        count = stats.tier_counts.get(tier.name, 0)
        pct = stats.tier_percentages.get(tier.name, 0.0)
        threshold = _tier_threshold_label(tier, config)
        lines.append(
            f"{tier.name.capitalize():<15} {count:>4} ({pct:>5.1f}%)  {threshold}"
        )

    lines += [
        "",
        "REGION LENGTH ANALYSIS",
        "-" * 40,
        f"Mean length:   {stats.mean_length:.1f} aa",
        f"Median length: {stats.median_length:.0f} aa",
        f"Range:         {stats.min_length}-{stats.max_length} aa",
        "",
        "Steric-zipper-scale classification (Sawaya et al., Nature 2007):",
        f"  Spine-scale (5-15 aa): {stats.classic_apr_count}",
        f"  Extended (>15 aa):     {stats.extended_apr_count}",
        "",
        "POSITIONAL DISTRIBUTION",
        "-" * 40,
        "Positional enrichment (permutation vs. relocation null):",
        f"  chi2 distance = {stats.chi2_statistic:.3f}",
        f"  p (perm) = {stats.chi2_p_value:.4f}",
        f"  -> {stats.positional_distribution.upper()}",
        "",
        "Tertile partition:",
        f"  N-terminal (0-0.33):  {stats.n_terminal_count}",
        f"  Central (0.33-0.67):  {stats.central_count}",
        f"  C-terminal (0.67-1):  {stats.c_terminal_count}",
        "",
        f"Enrichment pattern: {stats.enrichment_pattern}",
        "",
        "INTERPRETATION",
        "-" * 40,
    ]
    if stats.per_protein_positional:
        lines += [
            "",
            "PER-PROTEIN POSITIONAL",
            "-" * 40,
            "Where each protein's regions fall on its own 0-1 axis. Pooling can",
            "cancel opposite per-protein biases; the clustering p is shown only",
            "where a protein has enough regions to resolve it (>=4), otherwise",
            "the mean position is descriptive.",
            "",
        ]
        for pp in stats.per_protein_positional:
            label = config.label_for(pp.protein)
            zone = ("N-term" if pp.mean_position < 0.4
                    else "C-term" if pp.mean_position > 0.6 else "central")
            ptxt = (f"clustering p = {pp.clustering_p:.3f}"
                    if pp.clustering_p == pp.clustering_p
                    else "clustering p n/a (n<4)")
            lines.append(
                f"  {label:14s} n={pp.n_regions:<3d} mean={pp.mean_position:.2f} "
                f"({zone});  {ptxt}"
            )
                
    if config.hypothesis:
        lines.append(config.hypothesis.strip())
    else:
        lines.append(_default_interpretation(stats))
    lines += ["", "=" * 70, "END OF REPORT", "=" * 70]
    return "\n".join(lines)


def _tier_threshold_label(tier, config: PipelineConfig) -> str:
    from ..core.consensus import resolve_count

    n = len(config.enabled_tools)
    return f"(>={resolve_count(tier.min_fraction, n)}/{n} predictors)"


def _default_interpretation(stats: ConsensusStatistics) -> str:
    if stats.chi2_p_value > 0.05:
        return (
            "Consensus regions are distributed uniformly along the sequence "
            "(no significant decile bias). Under a uniform null, no particular "
            "sequence zone is preferentially enriched for aggregation "
            "determinants in this panel."
        )
    return (
        f"Consensus regions show a non-uniform distribution "
        f"({stats.enrichment_pattern.lower()}), indicating that aggregation "
        f"determinants are concentrated rather than dispersed; downstream "
        f"interpretation should consider structural or functional constraints "
        f"specific to the enriched zone."
    )
