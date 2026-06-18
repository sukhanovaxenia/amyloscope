"""Cross-predictor consensus calling.

A consensus aggregation-prone region is a window where independently-derived
predictors converge. Convergence across mechanistically orthogonal models
(hydrophobic clustering, beta-pairing free energy, position-specific matrices,
beta-arch structural compatibility) is stronger evidence for an intrinsic
aggregation determinant than any single score, because the shared signal is
unlikely to be a model-specific artefact.

The original implementation fixed the evidence tiers at literal predictor counts
(==8 / >=6 / >=4). Here tiers are fractions of the active panel, resolved to
counts at call time, so the same configuration is meaningful whether three
predictors are run or thirty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..config import ConsensusConfig, PipelineConfig
from .regions import call_regions

Interval = tuple[int, int]


@dataclass
class ConsensusRegion:
    """A consensus APR with its supporting predictors and evidence tier."""

    protein: str
    start: int
    end: int
    tools: list[str]
    tier: str
    sequence: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def n_tools(self) -> int:
        return len(self.tools)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2

    def normalized_position(self, protein_length: int) -> float:
        return self.midpoint / protein_length if protein_length else 0.0

    def as_dict(self) -> dict:
        return {
            "Protein": f"{self.protein}:{self.start}-{self.end}",
            "Class": self.tier,
            "Region": self.sequence,
            "Tools": ",".join(self.tools),
            "Length": self.length,
            "n_tools": self.n_tools,
        }


def resolve_count(fraction: float, n_tools: int) -> int:
    """Minimum predictor count for a fractional tier on an ``n_tools`` panel."""
    return max(1, math.ceil(fraction * n_tools))


def _per_tool_regions(
    protein_tracks, config: PipelineConfig
) -> dict[str, list[Interval]]:
    """Call APR intervals for each loaded predictor on one protein."""
    out: dict[str, list[Interval]] = {}
    min_len = config.consensus.min_region_length
    for tool in config.enabled_tools:
        if tool.name not in protein_tracks.tracks:
            continue
        df = protein_tracks.tracks[tool.name]
        out[tool.name] = call_regions(df, tool.detection, min_length=min_len)
    return out


def _covering_tools(
    per_tool: dict[str, list[Interval]],
    window: Interval,
    coverage_fraction: float,
) -> set[str]:
    """Predictors whose own call spans >= ``coverage_fraction`` of ``window``.

    Requiring substantial coverage (rather than a single overlapping residue)
    prevents incidental boundary clipping from inflating apparent agreement, in
    the spirit of the overlap criteria used by consensus metapredictors
    (Maurer-Stroh et al., Nature Methods 2010).
    """
    w_start, w_end = window
    w_len = w_end - w_start + 1
    covering: set[str] = set()
    for tool, intervals in per_tool.items():
        for t_start, t_end in intervals:
            ov = min(w_end, t_end) - max(w_start, t_start) + 1
            if ov > 0 and ov / w_len >= coverage_fraction:
                covering.add(tool)
                break
    return covering


def _resolve_overlaps(
    regions: list[ConsensusRegion], max_overlap: float
) -> list[ConsensusRegion]:
    """Greedily retain the strongest non-redundant regions.

    Priority: more supporting predictors, then longer span, then earlier start
    (Goldschmidt et al., PNAS 2010 — distinct amyloid spines are separable).
    """
    ordered = sorted(
        regions, key=lambda r: (-r.n_tools, -r.length, r.start)
    )
    kept: list[ConsensusRegion] = []
    covered: set[int] = set()
    for region in ordered:
        positions = set(range(region.start, region.end + 1))
        overlap = len(positions & covered) / len(positions)
        if overlap < max_overlap:
            kept.append(region)
            covered |= positions
    return sorted(kept, key=lambda r: r.start)


def _classify(n_tools: int, cfg: ConsensusConfig, panel_size: int) -> str:
    """Assign a region to the strongest tier its support satisfies."""
    for tier in cfg.tiers_by_strength:
        if n_tools >= resolve_count(tier.min_fraction, panel_size):
            return tier.name
    return cfg.tiers_by_strength[-1].name


def consensus_for_protein(
    protein_tracks, config: PipelineConfig
) -> list[ConsensusRegion]:
    """Compute classified consensus regions for one protein."""
    cfg = config.consensus
    per_tool = _per_tool_regions(protein_tracks, config)
    if not per_tool:
        return []

    panel_size = (
        len(config.enabled_tools)
        if cfg.denominator == "configured"
        else len(per_tool)
    )
    length = protein_tracks.length
    if length <= 0:
        return []

    # Position-wise tool coverage over the resolved length.
    position_tools: list[set[str]] = [set() for _ in range(length + 1)]
    for tool, intervals in per_tool.items():
        for start, end in intervals:
            for pos in range(start, min(end + 1, length + 1)):
                position_tools[pos].add(tool)

    floor_count = resolve_count(cfg.min_fraction(), panel_size)

    # Contiguous windows meeting the weakest tier's support floor.
    raw_regions: list[ConsensusRegion] = []
    in_window = False
    w_start = 0
    for pos in range(1, length + 1):
        meets = len(position_tools[pos]) >= floor_count
        if meets and not in_window:
            w_start, in_window = pos, True
        elif not meets and in_window:
            _finalise_window(
                raw_regions, per_tool, (w_start, pos - 1), cfg, panel_size,
                protein_tracks,
            )
            in_window = False
    if in_window:
        _finalise_window(
            raw_regions, per_tool, (w_start, length), cfg, panel_size,
            protein_tracks,
        )

    return _resolve_overlaps(raw_regions, cfg.overlap_resolution_max)


def _finalise_window(
    out: list[ConsensusRegion],
    per_tool: dict[str, list[Interval]],
    window: Interval,
    cfg: ConsensusConfig,
    panel_size: int,
    protein_tracks,
) -> None:
    w_start, w_end = window
    covering = _covering_tools(per_tool, window, cfg.coverage_fraction)
    floor_count = resolve_count(cfg.min_fraction(), panel_size)
    if len(covering) < floor_count or (w_end - w_start + 1) < cfg.min_region_length:
        return
    seq = protein_tracks.sequence[w_start - 1 : w_end] if protein_tracks.sequence else ""
    out.append(
        ConsensusRegion(
            protein=protein_tracks.spec.id,
            start=w_start,
            end=w_end,
            tools=sorted(covering),
            tier=_classify(len(covering), cfg, panel_size),
            sequence=seq,
        )
    )


@dataclass
class ConsensusResult:
    """Consensus regions across the whole panel, indexed by protein."""

    config: PipelineConfig
    regions: dict[str, list[ConsensusRegion]] = field(default_factory=dict)

    def all_regions(self) -> list[ConsensusRegion]:
        return [r for regs in self.regions.values() for r in regs]

    def to_dataframe(self):
        import pandas as pd

        rows = [r.as_dict() for r in self.all_regions()]
        return pd.DataFrame(rows)


def compute_consensus(dataset) -> ConsensusResult:
    """Compute consensus regions for every protein in a loaded dataset."""
    result = ConsensusResult(config=dataset.config)
    for protein_tracks in dataset:
        result.regions[protein_tracks.spec.id] = consensus_for_protein(
            protein_tracks, dataset.config
        )
    return result
