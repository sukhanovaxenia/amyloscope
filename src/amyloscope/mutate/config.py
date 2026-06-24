"""Configuration schema for the in-silico mutagenesis module.

A run targets one or more proteins. Each protein declares its own wild-type
sequence, its targets, and any protected positions (all of which are inherently
protein-specific); the grammars, vectors, combinatorial limits and scoring
weights are shared across the panel. As with the rest of amyloscope the config
is parsed into typed dataclasses and validated up front, so a malformed run
fails before any sequences are written.

Two equivalent spellings are accepted: a top-level ``proteins:`` list, or — for a
single protein — top-level ``sequence:`` and ``targets:`` blocks, which are
folded into a one-element panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .effect import EffectWeights

VALID_MODES = {"amyloid", "condensate"}
VALID_VECTORS = {"activating", "inhibiting", "neutral", "nonsense"}


class MutateConfigError(ValueError):
    """Raised when a mutagenesis configuration is malformed."""


@dataclass
class SequenceSpec:
    id: str
    sequence: str

    def __post_init__(self) -> None:
        self.sequence = "".join(self.sequence.split()).upper()


@dataclass
class TargetSpec:
    """Where to mutate. The resolved positions are the union of all sources."""

    consensus_tsv: str | None = None
    consensus_protein: str | None = None      # filter the TSV to one protein id
    consensus_min_tier: str | None = None      # optional tier floor (by name)
    regions: list[tuple[int, int]] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    scan_whole_sequence: bool = False


@dataclass
class ProteinJob:
    """One protein's wild-type sequence, targets and protected positions."""

    sequence: SequenceSpec
    targets: TargetSpec
    protected_positions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        # When a shared consensus table is targeted without an explicit filter,
        # default to this protein's own id so a panel reads the right rows.
        if self.targets.consensus_tsv and not self.targets.consensus_protein:
            self.targets.consensus_protein = self.sequence.id


@dataclass
class MutationSpec:
    orders: list[int] = field(default_factory=lambda: [1])
    top_k_per_position: int = 2          # substitutions kept per position/vector
    top_k_positions: int = 12            # leverage-ranked positions for multi-site
    max_per_order: int = 200             # cap variants per (mode, vector, order)
    max_total: int = 1000                # cap per protein
    protected_positions: list[int] = field(default_factory=list)  # legacy carrier
    min_abs_effect: float = 0.0          # ignore sub-threshold substitutions
    truncate_at: str = "region_start"    # 'region_start' | 'position'


@dataclass
class MutagenesisConfig:
    # Canonical field; a single-protein config supplies sequence/targets instead
    # and they are folded into `proteins` in __post_init__.
    proteins: list[ProteinJob] = field(default_factory=list)
    sequence: SequenceSpec | None = None     # legacy single-protein spelling
    targets: TargetSpec | None = None        # legacy single-protein spelling
    modes: list[str] = field(default_factory=lambda: ["amyloid"])
    vectors: list[str] = field(
        default_factory=lambda: ["activating", "inhibiting", "neutral"]
    )
    mutation: MutationSpec = field(default_factory=MutationSpec)
    weights: EffectWeights = field(default_factory=EffectWeights)
    output_dir: str = "mutants"
    name: str = "mutagenesis"

    def __post_init__(self) -> None:
        if not self.proteins and self.sequence is not None:
            self.proteins = [
                ProteinJob(
                    sequence=self.sequence,
                    targets=self.targets or TargetSpec(),
                    protected_positions=list(self.mutation.protected_positions),
                )
            ]

    # -- derived -----------------------------------------------------------
    @property
    def substitution_vectors(self) -> list[str]:
        return [v for v in self.vectors if v != "nonsense"]

    def _validate_shared(self) -> None:
        bad_modes = set(self.modes) - VALID_MODES
        if bad_modes:
            raise MutateConfigError(
                f"unknown modes {sorted(bad_modes)}; allowed: {sorted(VALID_MODES)}"
            )
        if not self.modes:
            raise MutateConfigError("at least one mode is required")
        bad_vectors = set(self.vectors) - VALID_VECTORS
        if bad_vectors:
            raise MutateConfigError(
                f"unknown vectors {sorted(bad_vectors)}; "
                f"allowed: {sorted(VALID_VECTORS)}"
            )
        if not self.vectors:
            raise MutateConfigError("at least one vector is required")
        if any(o < 1 for o in self.mutation.orders):
            raise MutateConfigError("mutation orders must be >= 1")
        if self.mutation.truncate_at not in {"region_start", "position"}:
            raise MutateConfigError(
                "mutation.truncate_at must be 'region_start' or 'position'"
            )

    def _validate_job(self, job: ProteinJob) -> None:
        where = f"protein '{job.sequence.id}'"
        if not job.sequence.sequence:
            raise MutateConfigError(f"{where}: sequence is empty")
        n = len(job.sequence.sequence)
        for p in list(job.protected_positions) + list(job.targets.positions):
            if not 1 <= p <= n:
                raise MutateConfigError(f"{where}: position {p} out of range 1..{n}")
        for a, b in job.targets.regions:
            if not 1 <= a <= b <= n:
                raise MutateConfigError(f"{where}: region [{a},{b}] out of range 1..{n}")
        no_targets = (
            not job.targets.consensus_tsv
            and not job.targets.regions
            and not job.targets.positions
            and not job.targets.scan_whole_sequence
        )
        if no_targets:
            raise MutateConfigError(
                f"{where}: no targets specified (set consensus_tsv, regions, "
                "positions, or scan_whole_sequence)"
            )

    def validate(self) -> None:
        if not self.proteins:
            raise MutateConfigError(
                "no proteins specified (provide a 'proteins' list or a "
                "'sequence'/'targets' block)"
            )
        ids = [j.sequence.id for j in self.proteins]
        if len(set(ids)) != len(ids):
            raise MutateConfigError(f"duplicate protein ids: {ids}")
        self._validate_shared()
        for job in self.proteins:
            self._validate_job(job)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def _read_fasta(path: Path) -> tuple[str, str]:
    seq_id, chunks = path.stem, []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                seq_id = line[1:].split()[0] or seq_id
            elif line:
                chunks.append(line)
    if not chunks:
        raise MutateConfigError(f"{path}: no sequence found")
    return seq_id, "".join(chunks)


def _read_sequence(block: dict, base: Path) -> SequenceSpec:
    if "sequence" in block:
        return SequenceSpec(id=block.get("id", "query"), sequence=block["sequence"])
    if "fasta" in block:
        seq_id, seq = _read_fasta(base / block["fasta"])
        return SequenceSpec(id=block.get("id", seq_id), sequence=seq)
    raise MutateConfigError("sequence block needs 'sequence' or 'fasta'")


def _read_targets(t: dict) -> TargetSpec:
    return TargetSpec(
        consensus_tsv=t.get("consensus_tsv"),
        consensus_protein=t.get("consensus_protein"),
        consensus_min_tier=t.get("consensus_min_tier"),
        regions=[tuple(r) for r in t.get("regions", [])],
        positions=list(t.get("positions", [])),
        scan_whole_sequence=bool(t.get("scan_whole_sequence", False)),
    )


def _read_protein(entry: dict, base: Path) -> ProteinJob:
    return ProteinJob(
        sequence=_read_sequence(entry.get("sequence", entry), base),
        targets=_read_targets(entry.get("targets", {})),
        protected_positions=list(entry.get("protected_positions", [])),
    )


def _read_mutation(m: dict) -> MutationSpec:
    return MutationSpec(
        orders=list(m.get("orders", [1])),
        top_k_per_position=int(m.get("top_k_per_position", 2)),
        top_k_positions=int(m.get("top_k_positions", 12)),
        max_per_order=int(m.get("max_per_order", 200)),
        max_total=int(m.get("max_total", 1000)),
        protected_positions=list(m.get("protected_positions", [])),
        min_abs_effect=float(m.get("min_abs_effect", 0.0)),
        truncate_at=m.get("truncate_at", "region_start"),
    )


def _read_weights(w: dict) -> EffectWeights:
    return EffectWeights(
        hydrophobicity=float(w.get("hydrophobicity", 1.0)),
        beta=float(w.get("beta", 1.0)),
        charge_penalty=float(w.get("charge_penalty", 0.7)),
        breaker_penalty=float(w.get("breaker_penalty", 1.2)),
        window=int(w.get("window", 7)),
        flank=int(w.get("flank", 3)),
    )


def load_mutate_config(path: str | Path) -> MutagenesisConfig:
    path = Path(path)
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    base = path.parent

    mutation = _read_mutation(raw.get("mutation", {}))

    # Panel form (proteins: [...]) takes precedence; otherwise fold a single
    # top-level sequence/targets block, attaching the legacy protected_positions.
    proteins: list[ProteinJob] = []
    if raw.get("proteins"):
        proteins = [_read_protein(entry, base) for entry in raw["proteins"]]
    elif "sequence" in raw:
        job = _read_protein(
            {"sequence": raw["sequence"], "targets": raw.get("targets", {})}, base
        )
        job.protected_positions = list(mutation.protected_positions)
        proteins = [job]

    cfg = MutagenesisConfig(
        proteins=proteins,
        modes=list(raw.get("modes", ["amyloid"])),
        vectors=list(raw.get("vectors", ["activating", "inhibiting", "neutral"])),
        mutation=mutation,
        weights=_read_weights(raw.get("weights", {})),
        output_dir=raw.get("output_dir", "mutants"),
        name=raw.get("name", path.stem),
    )
    cfg.validate()
    return cfg
