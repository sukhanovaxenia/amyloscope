"""Configuration schema for amyloscope.

Every quantity that the original ribosomal-protein pipeline hardcoded is
expressed here as a validated, serialisable object: the predictor set and their
per-tool APR-calling rules, the protein panel (identity, length, sequence,
domain architecture), the consensus evidence tiers, and the rendering palette.

The biological rationale for making consensus *fractional* rather than an
absolute predictor count: the meaningful signal is the proportion of
mechanistically orthogonal models (hydrophobic-cluster, beta-pairing energy,
position-specific matrices, structural beta-arch compatibility) that converge on
a window. That proportion is what licenses an inference of an intrinsic
aggregation determinant; an absolute count of "6 tools" is only interpretable
relative to how many were run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Per-tool APR detection strategy
# --------------------------------------------------------------------------- #

#: Supported primitives for converting a parsed per-residue track into a
#: boolean "residue lies in an aggregation-prone region" mask. Each registered
#: adapter declares a default strategy; users may override the threshold.
VALID_METHODS = {"above", "below", "flag", "notnull", "nonzero", "present"}


@dataclass
class DetectionStrategy:
    """How to turn one predictor's track into a per-residue APR mask.

    Parameters
    ----------
    method
        One of :data:`VALID_METHODS`.

        ``above``    residue is APR when ``column >= threshold`` (hydrophobicity-
                     and propensity-scaled scores, e.g. Aggrescan, AggreProt).
        ``below``    residue is APR when ``column <= threshold``. Required for
                     free-energy scores where *lower* (more negative) values
                     denote more stable cross-beta pairing (PASTA 2.0).
        ``flag``     residue is APR when ``column`` holds one of
                     ``flag_true_values`` (FoldAmyloid ``Fold == 'f'``, APPNN
                     ``is_hotspot``).
        ``notnull``  residue is APR when ``column`` is populated (Aggrescan
                     emits a ``Prediction`` label only inside hotspots).
        ``nonzero``  residue is APR when ``column != 0`` (Waltz emits 0 outside
                     matrix hits).
        ``present``  any parsed row marks an APR (region-list formats such as
                     ArchCandy that only report hit intervals).
    column
        Column the method reads. Ignored for ``present``.
    threshold
        Cutoff for ``above`` / ``below``. ``None`` is invalid for those methods.
    flag_true_values
        Values denoting an APR for ``flag``.
    """

    method: str = "above"
    column: str = "Score"
    threshold: float | None = None
    flag_true_values: list[Any] = field(default_factory=list)

    def validate(self, tool_name: str) -> None:
        if self.method not in VALID_METHODS:
            raise ConfigError(
                f"tool '{tool_name}': detection.method '{self.method}' is not one "
                f"of {sorted(VALID_METHODS)}"
            )
        if self.method in {"above", "below"} and self.threshold is None:
            raise ConfigError(
                f"tool '{tool_name}': detection.method '{self.method}' requires a "
                f"numeric 'threshold'"
            )
        if self.method == "flag" and not self.flag_true_values:
            raise ConfigError(
                f"tool '{tool_name}': detection.method 'flag' requires non-empty "
                f"'flag_true_values'"
            )


@dataclass
class ToolSpec:
    """A single predictor: where its output lives and how to read/call it."""

    name: str
    adapter: str
    path_template: str
    detection: DetectionStrategy = field(default_factory=DetectionStrategy)
    color: str = "#333333"
    enabled: bool = True

    def path_for(self, protein_id: str) -> Path:
        """Resolve ``path_template`` for a given protein id."""
        return Path(self.path_template.format(protein=protein_id))


# --------------------------------------------------------------------------- #
# Protein panel
# --------------------------------------------------------------------------- #


@dataclass
class Domain:
    """A sequence-annotated structural/functional module.

    ``category`` is an abstract label (e.g. ``rna_binding``, ``structured_core``,
    ``disordered``, ``regulatory``) that the palette maps to a colour. Keeping
    the category abstract is what decouples the renderer from any one protein
    family: ribosomal dsRBD/KOW folds and, say, a kinase activation loop both
    collapse onto the same small vocabulary of functional categories.
    """

    name: str
    start: int
    stop: int
    category: str = "default"

    def validate(self, protein_id: str) -> None:
        if self.start < 1 or self.stop < self.start:
            raise ConfigError(
                f"protein '{protein_id}', domain '{self.name}': invalid interval "
                f"[{self.start}, {self.stop}]"
            )


@dataclass
class ProteinSpec:
    """One protein under analysis.

    Parameters
    ----------
    id
        Stable identifier used to resolve file paths (often a UniProt accession
        or gene symbol).
    display_name
        Label used in figures. Falls back to ``id`` when omitted; this is the
        single configured replacement for the previously triplicated
        ``RPS2 -> eS2`` maps.
    length
        Residue count. ``None`` defers inference to the loaded tracks (the max
        residue index observed), which is the safe default when a curated length
        is unavailable.
    sequence
        Optional one-letter sequence; when present it is the authoritative
        source for region sequence extraction.
    domains
        Architecture for overlap analysis. May be empty.
    """

    id: str
    display_name: str | None = None
    length: int | None = None
    sequence: str | None = None
    domains: list[Domain] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.display_name or self.id

    def validate(self) -> None:
        if not self.id:
            raise ConfigError("every protein requires a non-empty 'id'")
        if self.length is not None and self.length < 1:
            raise ConfigError(f"protein '{self.id}': length must be >= 1")
        if self.sequence:
            self.sequence = re.sub(r"\s+", "", self.sequence).upper()
            if self.length and len(self.sequence) != self.length:
                raise ConfigError(
                    f"protein '{self.id}': sequence length {len(self.sequence)} "
                    f"!= declared length {self.length}"
                )
        for dom in self.domains:
            dom.validate(self.id)


# --------------------------------------------------------------------------- #
# Consensus model
# --------------------------------------------------------------------------- #


@dataclass
class ConsensusTier:
    """An evidence tier defined as a fraction of the active predictor set."""

    name: str
    min_fraction: float
    color: str

    def validate(self) -> None:
        if not 0.0 < self.min_fraction <= 1.0:
            raise ConfigError(
                f"consensus tier '{self.name}': min_fraction must be in (0, 1], "
                f"got {self.min_fraction}"
            )


@dataclass
class ConsensusConfig:
    """Parameters governing cross-tool consensus calling.

    Parameters
    ----------
    tiers
        Ordered evidence tiers (strongest first). Defaults reproduce the
        original unanimous/strong/moderate scheme as 1.0/0.75/0.5.
    min_region_length
        Minimum APR length in residues. The 5-residue floor reflects the
        steric-zipper spine length below which a contiguous cross-beta segment
        is not structurally meaningful (Sawaya et al., Nature 2007).
    coverage_fraction
        A predictor is counted toward a consensus window only if its own call
        spans at least this fraction of the window, preventing single-residue
        clipping from inflating apparent agreement.
    overlap_resolution_max
        Maximum positional overlap permitted between two retained consensus
        regions during de-duplication.
    denominator
        ``configured`` resolves tier counts against the full enabled tool set
        (stable thresholds across proteins). ``available`` resolves against the
        tools that returned data for each protein (tolerant of missing inputs).
    """

    tiers: list[ConsensusTier] = field(
        default_factory=lambda: [
            ConsensusTier("unanimous", 1.0, "#B2182B"),
            ConsensusTier("strong", 0.75, "#2166AC"),
            ConsensusTier("moderate", 0.5, "#FDAE61"),
        ]
    )
    min_region_length: int = 5
    coverage_fraction: float = 0.8
    overlap_resolution_max: float = 0.3
    denominator: str = "configured"

    def validate(self) -> None:
        if not self.tiers:
            raise ConfigError("consensus.tiers must be non-empty")
        for tier in self.tiers:
            tier.validate()
        if self.denominator not in {"configured", "available"}:
            raise ConfigError(
                f"consensus.denominator must be 'configured' or 'available', "
                f"got '{self.denominator}'"
            )
        if not 0.0 < self.coverage_fraction <= 1.0:
            raise ConfigError("consensus.coverage_fraction must be in (0, 1]")

    @property
    def tiers_by_strength(self) -> list[ConsensusTier]:
        """Tiers sorted strongest-first by ``min_fraction``."""
        return sorted(self.tiers, key=lambda t: t.min_fraction, reverse=True)

    def min_fraction(self) -> float:
        return min(t.min_fraction for t in self.tiers)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

#: Abstract functional-category palette. Keys are matched case-insensitively as
#: substrings of a domain's ``category``; the first match wins. This vocabulary
#: is intentionally family-agnostic.
DEFAULT_DOMAIN_PALETTE: dict[str, str] = {
    "rna_binding": "#4292C6",
    "dna_binding": "#4292C6",
    "structured_core": "#41AB5D",
    "catalytic": "#41AB5D",
    "disordered": "#FEC44F",
    "low_complexity": "#FEC44F",
    "linker": "#FEC44F",
    "localization": "#FB6A4A",
    "regulatory": "#9E9AC8",
    "terminal": "#756BB1",
    "conserved": "#99000D",
    "default": "#CCCCCC",
}


@dataclass
class VizConfig:
    """Figure rendering options."""

    dpi: int = 300
    save_svg: bool = True
    font_family: str = "Arial"
    x_padding: int = 15
    domain_palette: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_DOMAIN_PALETTE)
    )


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    """Fully-resolved analysis configuration."""

    name: str
    proteins: list[ProteinSpec]
    tools: list[ToolSpec]
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    output_dir: str = "amyloscope_output"
    #: Optional free-text hypothesis surfaced in the statistics report. The
    #: ribosome-protection narrative belongs here when relevant, rather than
    #: hardcoded into the report generator.
    hypothesis: str | None = None

    # -- convenience views ------------------------------------------------- #

    @property
    def enabled_tools(self) -> list[ToolSpec]:
        return [t for t in self.tools if t.enabled]

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.enabled_tools]

    @property
    def tool_colors(self) -> dict[str, str]:
        return {t.name: t.color for t in self.enabled_tools}

    @property
    def protein_ids(self) -> list[str]:
        return [p.id for p in self.proteins]

    def protein(self, protein_id: str) -> ProteinSpec:
        for p in self.proteins:
            if p.id == protein_id:
                return p
        raise KeyError(protein_id)

    def label_for(self, protein_id: str) -> str:
        return self.protein(protein_id).label

    # -- validation -------------------------------------------------------- #

    def validate(self) -> None:
        if not self.proteins:
            raise ConfigError("at least one protein is required")
        if not self.enabled_tools:
            raise ConfigError("at least one enabled tool is required")
        seen_tools: set[str] = set()
        for tool in self.tools:
            if tool.name in seen_tools:
                raise ConfigError(f"duplicate tool name '{tool.name}'")
            seen_tools.add(tool.name)
            tool.detection.validate(tool.name)
        seen_proteins: set[str] = set()
        for prot in self.proteins:
            if prot.id in seen_proteins:
                raise ConfigError(f"duplicate protein id '{prot.id}'")
            seen_proteins.add(prot.id)
            prot.validate()
        self.consensus.validate()


class ConfigError(ValueError):
    """Raised on a malformed or internally inconsistent configuration."""


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #


def _build_detection(raw: dict[str, Any] | None) -> DetectionStrategy:
    raw = raw or {}
    return DetectionStrategy(
        method=raw.get("method", "above"),
        column=raw.get("column", "Score"),
        threshold=raw.get("threshold"),
        flag_true_values=list(raw.get("flag_true_values", [])),
    )


def _build_tool(raw: dict[str, Any]) -> ToolSpec:
    try:
        return ToolSpec(
            name=raw["name"],
            adapter=raw["adapter"],
            path_template=raw["path"],
            detection=_build_detection(raw.get("detection")),
            color=raw.get("color", "#333333"),
            enabled=raw.get("enabled", True),
        )
    except KeyError as exc:
        raise ConfigError(f"tool entry missing required key {exc}") from exc


def _build_protein(raw: dict[str, Any]) -> ProteinSpec:
    try:
        domains = [
            Domain(
                name=d["name"],
                start=int(d["start"]),
                stop=int(d["stop"]),
                category=d.get("category", "default"),
            )
            for d in raw.get("domains", [])
        ]
        return ProteinSpec(
            id=raw["id"],
            display_name=raw.get("display_name"),
            length=raw.get("length"),
            sequence=raw.get("sequence"),
            domains=domains,
        )
    except KeyError as exc:
        raise ConfigError(f"protein entry missing required key {exc}") from exc


def _build_consensus(raw: dict[str, Any] | None) -> ConsensusConfig:
    if not raw:
        return ConsensusConfig()
    base = ConsensusConfig()
    tiers = (
        [
            ConsensusTier(t["name"], float(t["min_fraction"]), t.get("color", "#999999"))
            for t in raw["tiers"]
        ]
        if "tiers" in raw
        else base.tiers
    )
    return ConsensusConfig(
        tiers=tiers,
        min_region_length=raw.get("min_region_length", base.min_region_length),
        coverage_fraction=raw.get("coverage_fraction", base.coverage_fraction),
        overlap_resolution_max=raw.get(
            "overlap_resolution_max", base.overlap_resolution_max
        ),
        denominator=raw.get("denominator", base.denominator),
    )


def _build_viz(raw: dict[str, Any] | None) -> VizConfig:
    if not raw:
        return VizConfig()
    base = VizConfig()
    palette = dict(DEFAULT_DOMAIN_PALETTE)
    palette.update(raw.get("domain_palette", {}))
    return VizConfig(
        dpi=raw.get("dpi", base.dpi),
        save_svg=raw.get("save_svg", base.save_svg),
        font_family=raw.get("font_family", base.font_family),
        x_padding=raw.get("x_padding", base.x_padding),
        domain_palette=palette,
    )


def load_config(path: str | Path) -> PipelineConfig:
    """Parse, build and validate a :class:`PipelineConfig` from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")

    cfg = PipelineConfig(
        name=raw.get("name", path.stem),
        proteins=[_build_protein(p) for p in raw.get("proteins", [])],
        tools=[_build_tool(t) for t in raw.get("tools", [])],
        consensus=_build_consensus(raw.get("consensus")),
        viz=_build_viz(raw.get("viz")),
        output_dir=raw.get("output_dir", "amyloscope_output"),
        hypothesis=raw.get("hypothesis"),
    )
    cfg.validate()
    return cfg


def with_overrides(cfg: PipelineConfig, **changes: Any) -> PipelineConfig:
    """Return a shallow copy of ``cfg`` with top-level fields replaced."""
    new = replace(cfg, **changes)
    new.validate()
    return new
