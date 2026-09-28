"""Overlapping-tier APR clusters, and whether a shoulder is evidence or window width.

The observation this module exists for: a moderate-tier region frequently
overlaps a narrower strong or unanimous one. The tempting move is to promote the
whole overlapping pattern to the higher tier. This module deliberately does
**not** do that, and reports the two extents separately instead.

Why not promote
---------------
A tier *is* the measured agreement count. Relabelling a residue that 5 of 10
predictors called as "unanimous" because it abuts a residue 10 of 10 called
asserts evidence that was not observed, and it does so irreversibly: once the
tier column says unanimous, nothing downstream can recover which residues were
actually unanimous. Every statistic built on tiers — enrichment, correlation
with a measurement, the tier-rank ramp in the figures — silently changes
meaning.

The deeper hazard is that the shoulder is partly manufactured by the panel
itself. Predictors differ enormously in how much of a protein they call: in this
project's own audit, Cross-Beta labelled ~39-51 % of an average chain and
ArchCandy ~26 %, against Waltz at ~4 % and TANGO at ~6 %. A broad caller trained
on long windows — Cross-Beta's model is fitted on 15-residue windows, so it
detects extended aggregation-prone stretches and is not built to resolve a
nucleating hexapeptide, a steric-zipper core or a short gatekeeper — will paint a
halo around any genuine core. Promoting that halo lets the broadest model in the
panel set the reported boundary of every APR, which is precisely the limitation
one would want a consensus to average away.

Why the extent still matters
----------------------------
The opposite error is just as real. Experimentally confirmed amyloid cores are
routinely far longer than a hexapeptide, and reporting only the narrow
high-agreement core understates the aggregation-competent segment. A nucleus
sitting inside a longer APR is the expected architecture, not an artifact.

What this module does instead
-----------------------------
It keeps both, and adds a discriminator so the choice is made on evidence rather
than taste. For each cluster it asks **which predictors support the shoulder**:

* if one or more *narrow* callers extend into it, the extension is supported by
  models that are capable of resolving a short region and chose not to stop at
  the core — ``extension_supported``;
* if the shoulder is held up only by *broad* callers, it is consistent with
  window width rather than sequence signal — ``window_artifact``;
* anything else — ``ambiguous``.

Breadth is measured from the run itself (each tool's called fraction of the
panel), not hardcoded, so the classification follows whatever panel is actually
configured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..core.consensus import (
    ConsensusRegion,
    ConsensusResult,
    _classify,
    _per_tool_regions,
)

#: Verdicts for a cluster's shoulder.
NO_SHOULDER = "no_shoulder"
EXTENSION_SUPPORTED = "extension_supported"
WINDOW_ARTIFACT = "window_artifact"
AMBIGUOUS = "ambiguous"


@dataclass
class ToolBreadth:
    """How much of the panel one predictor calls, measured on this run."""

    tool: str
    called_aa: int
    total_aa: int

    @property
    def fraction(self) -> float:
        return self.called_aa / self.total_aa if self.total_aa else 0.0


@dataclass
class APRCluster:
    """A high-tier core plus the contiguous lower-tier regions around it."""

    protein: str
    core_start: int
    core_stop: int
    core_tier: str
    extent_start: int
    extent_stop: int
    extent_tier: str = ""
    peak_tools: int = 0
    members: list[ConsensusRegion] = field(default_factory=list)
    core_tools: list[str] = field(default_factory=list)
    shoulder_tools: list[str] = field(default_factory=list)
    narrow_tools_in_shoulder: list[str] = field(default_factory=list)
    broad_tools_in_shoulder: list[str] = field(default_factory=list)
    verdict: str = NO_SHOULDER
    core_sequence: str = ""
    extent_sequence: str = ""

    @property
    def core_length(self) -> int:
        return self.core_stop - self.core_start + 1

    @property
    def extent_length(self) -> int:
        return self.extent_stop - self.extent_start + 1

    @property
    def shoulder_length(self) -> int:
        return self.extent_length - self.core_length

    def as_dict(self) -> dict:
        return {
            "protein": self.protein,
            "core_start": self.core_start,
            "core_stop": self.core_stop,
            "core_tier": self.core_tier,
            "core_length": self.core_length,
            "core_sequence": self.core_sequence,
            "extent_tier": self.extent_tier,
            "peak_tools": self.peak_tools,
            "extent_start": self.extent_start,
            "extent_stop": self.extent_stop,
            "extent_length": self.extent_length,
            "shoulder_length": self.shoulder_length,
            "n_member_regions": len(self.members),
            "member_tiers": ";".join(r.tier for r in self.members),
            "core_tools": ";".join(self.core_tools),
            "shoulder_only_tools": ";".join(self.shoulder_tools),
            "narrow_tools_in_shoulder": ";".join(self.narrow_tools_in_shoulder),
            "broad_tools_in_shoulder": ";".join(self.broad_tools_in_shoulder),
            "verdict": self.verdict,
        }


@dataclass
class ClusterResult:
    clusters: dict[str, list[APRCluster]] = field(default_factory=dict)
    breadth: dict[str, ToolBreadth] = field(default_factory=dict)
    narrow_tools: list[str] = field(default_factory=list)
    broad_tools: list[str] = field(default_factory=list)

    def all_clusters(self) -> list[APRCluster]:
        return [c for cs in self.clusters.values() for c in cs]

    def to_dataframe(self):
        import pandas as pd

        return pd.DataFrame([c.as_dict() for c in self.all_clusters()])

    def breadth_to_dataframe(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "tool": b.tool,
                    "called_aa": b.called_aa,
                    "total_aa": b.total_aa,
                    "fraction": round(b.fraction, 4),
                    "class": (
                        "narrow" if b.tool in self.narrow_tools
                        else "broad" if b.tool in self.broad_tools
                        else "intermediate"
                    ),
                }
                for b in sorted(self.breadth.values(), key=lambda x: x.fraction)
            ]
        )


def measure_breadth(dataset, config) -> dict[str, ToolBreadth]:
    """Called fraction per predictor, pooled over the whole panel."""
    called: dict[str, int] = {}
    total = 0
    for protein_tracks in dataset:
        length = protein_tracks.length
        total += length
        per_tool = _per_tool_regions(protein_tracks, config)
        for tool, intervals in per_tool.items():
            residues = {p for a, b in intervals for p in range(a, b + 1)}
            called[tool] = called.get(tool, 0) + len(residues)
    return {t: ToolBreadth(tool=t, called_aa=n, total_aa=total) for t, n in called.items()}


def classify_breadth(
    breadth: dict[str, ToolBreadth], *, narrow_max: float | None = None,
    broad_min: float | None = None,
) -> tuple[list[str], list[str]]:
    """Split predictors into narrow and broad callers.

    Cut points default to the terciles of the observed breadth distribution, so
    the labels describe the panel that was actually run rather than a fixed idea
    of which tools are broad. Explicit fractions override that when a panel is
    too small for terciles to mean anything.
    """
    values = sorted(b.fraction for b in breadth.values())
    if not values:
        return [], []
    if narrow_max is None or broad_min is None:
        k = max(1, len(values) // 3)
        auto_narrow = values[k - 1]
        auto_broad = values[-k]
        narrow_max = auto_narrow if narrow_max is None else narrow_max
        broad_min = auto_broad if broad_min is None else broad_min
    # The bands must be disjoint or the labels mean nothing. A panel whose tools
    # all call a similar fraction (or a synthetic fixture where every tool shares
    # the same windows) collapses the terciles onto one value; returning no
    # classification is the honest outcome and makes every verdict AMBIGUOUS
    # rather than inventing a split that the data does not support.
    if narrow_max >= broad_min:
        return [], []
    narrow = sorted(t for t, b in breadth.items() if b.fraction <= narrow_max)
    broad = sorted(t for t, b in breadth.items() if b.fraction >= broad_min)
    return narrow, sorted(set(broad) - set(narrow))


def _link(regions: list[ConsensusRegion], gap: int) -> list[list[ConsensusRegion]]:
    """Group regions into overlap-or-adjacency connected components."""
    ordered = sorted(regions, key=lambda r: (r.start, r.end))
    groups: list[list[ConsensusRegion]] = []
    for region in ordered:
        if groups and region.start <= groups[-1][-1].end + gap + 1:
            groups[-1].append(region)
            # keep the group's running end monotone
            if region.end < groups[-1][-1].end:
                groups[-1][-1] = groups[-1][-1]
        else:
            groups.append([region])
    return groups


def _runs(positions: set[int]) -> list[tuple[int, int]]:
    """Contiguous 1-based runs from a set of residue positions."""
    out: list[list[int]] = []
    for pos in sorted(positions):
        if out and pos == out[-1][1] + 1:
            out[-1][1] = pos
        else:
            out.append([pos, pos])
    return [(a, b) for a, b in out]


def build_clusters(
    dataset,
    result: ConsensusResult,
    *,
    tier_rank: list[str] | None = None,
    gap: int = 0,
    narrow_max: float | None = None,
    broad_min: float | None = None,
) -> ClusterResult:
    """Separate each consensus region into its peak-agreement core and shoulder.

    The overlap that matters is not usually between two *regions* — overlap
    resolution already collapses those — but *within* one. A region is tiered on
    the number of predictors covering it as a whole, while agreement varies
    residue by residue inside it, so a region labelled ``moderate`` routinely
    contains a short run at strong or unanimous agreement. RPS2 80-86 in this
    project is exactly that: six covering tools, peak eight.

    Core  = the maximal run at the region's peak agreement.
    Extent = the region as called.
    Neither is relabelled as the other.
    """
    config = result.config
    ranking = tier_rank or [t.name for t in config.consensus.tiers]
    rank_of = {name: i for i, name in enumerate(ranking)}

    breadth = measure_breadth(dataset, config)
    narrow, broad = classify_breadth(breadth, narrow_max=narrow_max, broad_min=broad_min)

    out = ClusterResult(breadth=breadth, narrow_tools=narrow, broad_tools=broad)
    by_protein = {pt.spec.id: pt for pt in dataset}
    panel_size = len(config.enabled_tools)

    source = result.raw_regions if getattr(result, "raw_regions", None) else result.regions
    for protein_id, regions in source.items():
        clusters: list[APRCluster] = []
        protein_tracks = by_protein.get(protein_id)
        per_tool = _per_tool_regions(protein_tracks, config) if protein_tracks else {}
        sequence = protein_tracks.sequence if protein_tracks else ""
        tool_residues = {
            tool: {p for a, b in intervals for p in range(a, b + 1)}
            for tool, intervals in per_tool.items()
        }

        for group in _link(regions, gap):
            extent_start = min(r.start for r in group)
            extent_stop = max(r.end for r in group)
            span = range(extent_start, extent_stop + 1)

            # per-residue agreement inside the region
            support = {
                pos: {t for t, res in tool_residues.items() if pos in res} for pos in span
            }
            counts = {pos: len(s) for pos, s in support.items()}
            peak = max(counts.values()) if counts else 0
            core_runs = _runs({p for p, n in counts.items() if n == peak})
            core_start, core_stop = (
                max(core_runs, key=lambda r: r[1] - r[0]) if core_runs
                else (extent_start, extent_stop)
            )
            core_tier = _classify(peak, config.consensus, panel_size) or "below_tiers"
            region_tier = min(
                (r.tier for r in group), key=lambda t: rank_of.get(t, 99)
            )

            core_positions = set(range(core_start, core_stop + 1))
            shoulder = set(span) - core_positions
            core_tools = sorted(set().union(*[support[p] for p in core_positions]) if core_positions else set())
            shoulder_support = sorted(
                {t for p in shoulder for t in support[p]} - set(core_tools)
            ) if shoulder else []
            # A tool "extends" the shoulder if it calls shoulder residues at all,
            # whether or not it also called the core: what is being asked is
            # which models are responsible for the region being wider than its
            # peak, and a narrow caller that spans both is evidence, not noise.
            extending = sorted({t for p in shoulder for t in support[p]}) if shoulder else []
            narrow_in = sorted(set(extending) & set(narrow))
            broad_in = sorted(set(extending) & set(broad))

            if not shoulder or core_tier == region_tier:
                verdict = NO_SHOULDER
            elif narrow_in:
                verdict = EXTENSION_SUPPORTED
            elif broad_in:
                verdict = WINDOW_ARTIFACT
            else:
                verdict = AMBIGUOUS

            clusters.append(
                APRCluster(
                    protein=protein_id,
                    core_start=core_start,
                    core_stop=core_stop,
                    core_tier=core_tier,
                    extent_start=extent_start,
                    extent_stop=extent_stop,
                    extent_tier=region_tier,
                    peak_tools=peak,
                    members=list(group),
                    core_tools=core_tools,
                    shoulder_tools=shoulder_support,
                    narrow_tools_in_shoulder=narrow_in,
                    broad_tools_in_shoulder=broad_in,
                    verdict=verdict,
                    core_sequence=sequence[core_start - 1 : core_stop] if sequence else "",
                    extent_sequence=(
                        sequence[extent_start - 1 : extent_stop] if sequence else ""
                    ),
                )
            )
        out.clusters[protein_id] = clusters
    return out


def format_cluster_report(result: ClusterResult) -> str:
    """Plain-text summary, written so the verdict's basis is visible."""
    lines = ["APR clusters: high-tier cores and their contiguous shoulders", ""]
    lines.append("Predictor breadth measured on this run (called fraction of panel):")
    for b in sorted(result.breadth.values(), key=lambda x: -x.fraction):
        label = (
            "broad" if b.tool in result.broad_tools
            else "narrow" if b.tool in result.narrow_tools
            else "intermediate"
        )
        lines.append(f"  {b.tool:16s} {b.fraction:6.1%}  {label}")
    lines += [
        "",
        "A shoulder held up only by broad callers is consistent with window",
        "width rather than sequence signal; one that a narrow caller also",
        "reaches is an extension those models chose not to stop short of.",
        "Tiers are NEVER promoted: core_tier is the agreement actually measured.",
        "",
    ]
    for protein_id, clusters in result.clusters.items():
        if not clusters:
            continue
        lines.append(f"{protein_id}:")
        for c in clusters:
            lines.append(
                f"  core {c.core_start:>4}-{c.core_stop:<4} [{c.core_tier:9s}] "
                f"{c.core_sequence:<24s} extent {c.extent_start}-{c.extent_stop} "
                f"(+{c.shoulder_length} aa)  {c.verdict}"
            )
            if c.verdict != NO_SHOULDER:
                lines.append(
                    f"       shoulder support: narrow={c.narrow_tools_in_shoulder or '-'} "
                    f"broad={c.broad_tools_in_shoulder or '-'}"
                )
        lines.append("")
    if not any(result.clusters.values()):
        lines.append("(no consensus regions to cluster)")
    return "\n".join(lines)
