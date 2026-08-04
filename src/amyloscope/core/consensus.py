"""Cross-predictor consensus calling.

A consensus aggregation-prone region is a window where independently-derived
predictors converge. Convergence across mechanistically orthogonal models
(hydrophobic clustering, beta-pairing free energy, position-specific matrices,
beta-arch structural compatibility) is stronger evidence for an intrinsic
aggregation determinant than any single score, because the shared signal is
unlikely to be a model-specific artefact.

Tiers are fractions of the active panel, resolved to counts at call time, so
the same configuration is meaningful whether three predictors are run or thirty.

Two quantities are recorded per region and they are NOT the same number:

``n_tools``
    predictors whose own call covers at least ``coverage_fraction`` of the
    whole region. This is the conservative support used for tier assignment.
``peak_tools``
    the maximum position-wise agreement anywhere inside the region.

``peak_tools >= n_tools`` always. The position-wise profile in the figures
plots the peak quantity, so a region annotated "(5 tools)" can legitimately sit
under bars reaching 8 — report which one you mean in the figure legend.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from ..config import ConsensusConfig, PipelineConfig
from .regions import call_regions

Interval = tuple[int, int]

log = logging.getLogger(__name__)


@dataclass
class ConsensusRegion:
    """A consensus APR with its supporting predictors and evidence tier."""

    protein: str
    start: int
    end: int
    tools: list[str]
    tier: str
    sequence: str = ""
    peak_tools: int = 0
    peak_tool_names: list[str] = field(default_factory=list)

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
        """Midpoint on a 0 (N-terminus) to 1 (C-terminus) scale.

        Anchored on residue 1 rather than residue 0: with ``midpoint / L`` a
        region at the extreme N-terminus maps to 1/L instead of 0, which
        shifts every value in the positional-enrichment histogram towards the
        C-terminus by one bin-width fraction.
        """
        if protein_length <= 1:
            return 0.0
        return (self.midpoint - 1.0) / (protein_length - 1.0)

    def as_dict(self) -> dict:
        return {
            "Protein": self.protein,
            "Start": self.start,
            "End": self.end,
            "Locus": f"{self.protein}:{self.start}-{self.end}",
            "Class": self.tier,
            "Region": self.sequence,
            "Tools": ",".join(self.tools),
            "Length": self.length,
            "n_tools": self.n_tools,
            "peak_tools": self.peak_tools,
            "Peak_tools": ",".join(self.peak_tool_names),
        }


@dataclass
class RejectedWindow:
    """A candidate window that met the support floor position-wise but failed
    the region-level coverage or length criteria.

    Kept so the number of discarded candidates can be reported: silently
    dropping them is what makes a permissive ``coverage_fraction`` look like
    an absence of signal.
    """

    protein: str
    start: int
    end: int
    peak_tools: int
    covering_tools: int
    reason: str


def resolve_count(fraction: float, n_tools: int) -> int:
    """Minimum predictor count for a fractional tier on an ``n_tools`` panel."""
    return max(1, math.ceil(fraction * n_tools))


def _merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Union of possibly overlapping/adjacent intervals, sorted."""
    out: list[Interval] = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


def _per_tool_regions(
    protein_tracks, config: PipelineConfig
) -> dict[str, list[Interval]]:
    """Call APR intervals for each loaded predictor on one protein.

    ``min_tool_region_length`` is honoured when the config defines it, so the
    predictor-level short-hit filter can be set independently of the minimum
    consensus-region length; it falls back to ``min_region_length``.
    """
    out: dict[str, list[Interval]] = {}
    cfg = config.consensus
    min_len = getattr(cfg, "min_tool_region_length", None) or cfg.min_region_length
    for tool in config.enabled_tools:
        if tool.name not in protein_tracks.tracks:
            log.info(
                "%s: no track for %s — scored as a negative across the whole "
                "sequence because denominator=%s",
                getattr(protein_tracks.spec, "id", "?"), tool.name, cfg.denominator,
            )
            continue
        df = protein_tracks.tracks[tool.name]
        out[tool.name] = call_regions(df, tool.detection, min_length=min_len)
    return out


def _covering_tools(
    per_tool: dict[str, list[Interval]],
    window: Interval,
    coverage_fraction: float,
) -> set[str]:
    """Predictors whose calls jointly span >= ``coverage_fraction`` of ``window``.

    Coverage is summed over the union of a tool's own intervals. Testing each
    interval separately (the previous behaviour) discards a predictor whose
    call is split by a single sub-threshold residue, even when its intervals
    together cover the whole window — a frequent pattern for smoothed profile
    scores that dip momentarily across the cutoff.

    Note the criterion is window-relative and therefore gets monotonically
    harder as the window lengthens: a broad region of shifting membership can
    fail even though every position in it is supported by the required number
    of tools. Rejections are reported by :func:`consensus_for_protein`.
    """
    w_start, w_end = window
    w_len = w_end - w_start + 1
    if w_len <= 0:
        return set()
    covering: set[str] = set()
    for tool, intervals in per_tool.items():
        overlap = 0
        for t_start, t_end in _merge_intervals(intervals):
            lo, hi = max(w_start, t_start), min(w_end, t_end)
            if hi >= lo:
                overlap += hi - lo + 1
        if overlap / w_len >= coverage_fraction:
            covering.add(tool)
    return covering


def _resolve_overlaps(
    regions: list[ConsensusRegion], max_overlap: float
) -> list[ConsensusRegion]:
    """Greedily retain the strongest non-redundant regions.

    Priority: more supporting predictors, then longer span, then earlier start
    (Goldschmidt et al., PNAS 2010 — distinct amyloid spines are separable).

    Redundancy is judged against each retained region individually. Testing
    against the cumulative union of everything kept so far (the previous
    behaviour) could reject a region that overlaps several retained regions by
    a little each while exceeding the threshold with none of them, which is not
    what ``overlap_resolution_max`` is documented to mean.
    """
    ordered = sorted(regions, key=lambda r: (-r.n_tools, -r.length, r.start))
    kept: list[ConsensusRegion] = []
    for region in ordered:
        positions = set(range(region.start, region.end + 1))
        redundant = False
        for other in kept:
            other_pos = set(range(other.start, other.end + 1))
            if len(positions & other_pos) / len(positions) >= max_overlap:
                redundant = True
                break
        if not redundant:
            kept.append(region)
    return sorted(kept, key=lambda r: r.start)


def _classify(n_tools: int, cfg: ConsensusConfig, panel_size: int) -> str:
    """Assign a region to the strongest tier its support satisfies."""
    for tier in cfg.tiers_by_strength:
        if n_tools >= resolve_count(tier.min_fraction, panel_size):
            return tier.name
    return cfg.tiers_by_strength[-1].name


def consensus_for_protein(
    protein_tracks, config: PipelineConfig
) -> tuple[list[ConsensusRegion], list[RejectedWindow]]:
    """Compute classified consensus regions for one protein.

    Returns ``(regions, rejected_windows)``.
    """
    cfg = config.consensus
    per_tool = _per_tool_regions(protein_tracks, config)
    if not per_tool:
        return [], []

    panel_size = (
        len(config.enabled_tools)
        if cfg.denominator == "configured"
        else max(1, len(per_tool))
    )
    length = protein_tracks.length
    if not length or length <= 0:
        return [], []

    # Position-wise tool coverage over the resolved length (1-based).
    position_tools: list[set[str]] = [set() for _ in range(length + 1)]
    for tool, intervals in per_tool.items():
        for start, end in intervals:
            lo, hi = max(1, start), min(end, length)
            for pos in range(lo, hi + 1):
                position_tools[pos].add(tool)

    floor_count = resolve_count(cfg.min_fraction(), panel_size)

    raw_regions: list[ConsensusRegion] = []
    rejected: list[RejectedWindow] = []
    in_window = False
    w_start = 0
    for pos in range(1, length + 1):
        meets = len(position_tools[pos]) >= floor_count
        if meets and not in_window:
            w_start, in_window = pos, True
        elif not meets and in_window:
            _finalise_window(
                raw_regions, rejected, per_tool, position_tools,
                (w_start, pos - 1), cfg, panel_size, floor_count, protein_tracks,
            )
            in_window = False
    if in_window:
        _finalise_window(
            raw_regions, rejected, per_tool, position_tools,
            (w_start, length), cfg, panel_size, floor_count, protein_tracks,
        )

    if rejected:
        log.info(
            "%s: %d candidate window(s) met the %d/%d support floor "
            "position-wise but failed the region criteria",
            protein_tracks.spec.id, len(rejected), floor_count, panel_size,
        )
    return _resolve_overlaps(raw_regions, cfg.overlap_resolution_max), rejected


def _finalise_window(
    out: list[ConsensusRegion],
    rejected: list[RejectedWindow],
    per_tool: dict[str, list[Interval]],
    position_tools: list[set[str]],
    window: Interval,
    cfg: ConsensusConfig,
    panel_size: int,
    floor_count: int,
    protein_tracks,
) -> None:
    w_start, w_end = window
    w_len = w_end - w_start + 1
    covering = _covering_tools(per_tool, window, cfg.coverage_fraction)

    peak_pos = max(range(w_start, w_end + 1), key=lambda p: len(position_tools[p]))
    peak_set = position_tools[peak_pos]

    if w_len < cfg.min_region_length:
        rejected.append(RejectedWindow(
            protein_tracks.spec.id, w_start, w_end, len(peak_set), len(covering),
            f"shorter than min_region_length ({w_len} < {cfg.min_region_length})",
        ))
        return
    if len(covering) < floor_count:
        rejected.append(RejectedWindow(
            protein_tracks.spec.id, w_start, w_end, len(peak_set), len(covering),
            f"only {len(covering)} predictor(s) cover >= "
            f"{cfg.coverage_fraction:.0%} of the window (need {floor_count})",
        ))
        return

    seq = (
        protein_tracks.sequence[w_start - 1 : w_end]
        if protein_tracks.sequence else ""
    )
    out.append(
        ConsensusRegion(
            protein=protein_tracks.spec.id,
            start=w_start,
            end=w_end,
            tools=sorted(covering),
            tier=_classify(len(covering), cfg, panel_size),
            sequence=seq,
            peak_tools=len(peak_set),
            peak_tool_names=sorted(peak_set),
        )
    )


@dataclass
class ConsensusResult:
    """Consensus regions across the whole panel, indexed by protein."""

    config: PipelineConfig
    regions: dict[str, list[ConsensusRegion]] = field(default_factory=dict)
    rejected: dict[str, list[RejectedWindow]] = field(default_factory=dict)

    def all_regions(self) -> list[ConsensusRegion]:
        return [r for regs in self.regions.values() for r in regs]

    def all_rejected(self) -> list[RejectedWindow]:
        return [w for ws in self.rejected.values() for w in ws]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([r.as_dict() for r in self.all_regions()])

    def rejected_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([vars(w) for w in self.all_rejected()])


def compute_consensus(dataset) -> ConsensusResult:
    """Compute consensus regions for every protein in a loaded dataset."""
    result = ConsensusResult(config=dataset.config)
    for protein_tracks in dataset:
        regions, rejected = consensus_for_protein(protein_tracks, dataset.config)
        result.regions[protein_tracks.spec.id] = regions
        result.rejected[protein_tracks.spec.id] = rejected
    return result