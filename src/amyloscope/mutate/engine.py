"""Mutagenesis orchestration.

For each protein in the panel, resolves targets (consensus regions, explicit
regions/positions, or a whole-sequence scan), generates directed variants across
the requested grammars, vectors and mutation orders, ranks them by predicted
effect, and applies the configured caps. Multi-site variants are produced by
directed enumeration over the highest-leverage positions rather than the full
combinatorial product, so the search stays bounded. Proteins are independent —
each is generated and capped on its own.

The headline per-variant number is the change in global *aggregation load*
relative to wild type for the variant's grammar — a fast intrinsic heuristic.
It exists to prioritise which sequences are worth running through the external
predictor panel, not to replace it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from .config import MutagenesisConfig, ProteinJob, TargetSpec
from .effect import aggregation_load, charge_patterning
from .rules import Substitution, propose

_REGION_RE = re.compile(r"^(?P<pid>.+):(?P<start>\d+)-(?P<stop>\d+)$")


@dataclass(frozen=True)
class TargetPosition:
    position: int          # 1-based
    region_start: int      # region whose windowed propensity is scored
    region_stop: int
    source: str            # 'consensus' | 'region' | 'position' | 'scan'


@dataclass
class MutantRecord:
    protein: str
    id: str
    mode: str
    vector: str
    order: int
    sequence: str
    substitutions: list[Substitution]
    truncation: int | None
    delta_load: float
    delta_ncpr: float
    delta_fcr: float
    delta_scd: float

    @property
    def mutation_label(self) -> str:
        if self.truncation is not None:
            return f"p.{self.truncation}*"  # nonsense / truncation
        return ",".join(s.label[2:] for s in self.substitutions)


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #


def _tier_rank(tsv_classes: list[str]) -> dict[str, int]:
    # Strength order is not encoded in the TSV; fall back to first-seen order,
    # which matches the writer's strongest-first emission.
    rank, seen = {}, 0
    for c in tsv_classes:
        if c not in rank:
            rank[c] = seen
            seen += 1
    return rank


def _from_consensus(spec: TargetSpec) -> list[TargetPosition]:
    df = pd.read_csv(spec.consensus_tsv, sep="\t")
    out: list[TargetPosition] = []
    classes = df["Class"].tolist() if "Class" in df.columns else []
    rank = _tier_rank(classes)
    floor = rank.get(spec.consensus_min_tier, len(rank)) if spec.consensus_min_tier \
        else len(rank)
    for _, row in df.iterrows():
        m = _REGION_RE.match(str(row["Protein"]))
        if not m:
            continue
        if spec.consensus_protein and m["pid"] != spec.consensus_protein:
            continue
        if spec.consensus_min_tier and rank.get(row.get("Class"), 99) > floor:
            continue
        start, stop = int(m["start"]), int(m["stop"])
        for pos in range(start, stop + 1):
            out.append(TargetPosition(pos, start, stop, "consensus"))
    return out


def resolve_targets(
    cfg: MutagenesisConfig, job: ProteinJob | None = None
) -> list[TargetPosition]:
    """Union of every configured target source for one protein, de-duplicated.

    ``job`` defaults to the first protein in the panel for single-protein use.
    """
    job = job or cfg.proteins[0]
    seq_len = len(job.sequence.sequence)
    flank = cfg.weights.flank
    found: dict[int, TargetPosition] = {}

    def add(tp: TargetPosition) -> None:
        found.setdefault(tp.position, tp)  # first source wins for context

    if job.targets.consensus_tsv:
        for tp in _from_consensus(job.targets):
            if 1 <= tp.position <= seq_len:
                add(tp)
    for start, stop in job.targets.regions:
        for pos in range(start, stop + 1):
            add(TargetPosition(pos, start, stop, "region"))
    for pos in job.targets.positions:
        # Bare positions get a local window as their scoring region.
        add(TargetPosition(pos, pos, pos, "position"))
    if job.targets.scan_whole_sequence:
        for pos in range(1, seq_len + 1):
            add(TargetPosition(pos, max(1, pos - flank),
                              min(seq_len, pos + flank), "scan"))

    protected = set(job.protected_positions)
    return [tp for p, tp in sorted(found.items()) if p not in protected]


# --------------------------------------------------------------------------- #
# Variant construction
# --------------------------------------------------------------------------- #


def _patterning_deltas(wt: str, mut: str) -> tuple[float, float, float]:
    a, b = charge_patterning(wt), charge_patterning(mut)
    return b.ncpr - a.ncpr, b.fcr - a.fcr, b.scd - a.scd


def _make_record(
    cfg: MutagenesisConfig, job: ProteinJob, idx: int, mode: str, vector: str,
    order: int, subs: list[Substitution], truncation: int | None,
) -> MutantRecord:
    wt = job.sequence.sequence
    if truncation is not None:
        mut_seq = wt[: truncation - 1]
    else:
        chars = list(wt)
        for s in subs:
            chars[s.position - 1] = s.mut
        mut_seq = "".join(chars)
    dload = aggregation_load(mut_seq, mode, cfg.weights) - aggregation_load(
        wt, mode, cfg.weights
    )
    dn, df_, ds = _patterning_deltas(wt, mut_seq)
    rid = f"{job.sequence.id}_{mode}_{vector}_o{order}_{idx:04d}"
    return MutantRecord(
        protein=job.sequence.id, id=rid, mode=mode, vector=vector, order=order,
        sequence=mut_seq, substitutions=subs, truncation=truncation,
        delta_load=dload, delta_ncpr=dn, delta_fcr=df_, delta_scd=ds,
    )


def _rank(subs: list[Substitution], vector: str) -> list[Substitution]:
    if vector == "activating":
        return sorted(subs, key=lambda s: s.effect, reverse=True)
    if vector == "inhibiting":
        return sorted(subs, key=lambda s: s.effect)
    return sorted(subs, key=lambda s: abs(s.effect))


def _single_site(
    cfg: MutagenesisConfig, job: ProteinJob, mode: str, vector: str,
    targets: list[TargetPosition],
) -> list[list[Substitution]]:
    """Candidate single substitutions, pre-ranked by local region effect.

    ``propose`` uses the windowed region effect to pick the best substitutions
    at each position (this is where gatekeeper context matters); the final
    cross-position ranking and capping happen in :func:`generate_for_protein` on
    the global load delta, so the surviving set agrees with the reported metric.
    """
    wt = job.sequence.sequence
    proposals: list[Substitution] = []
    for tp in targets:
        proposals.extend(
            propose(
                wt, tp.position, tp.region_start, tp.region_stop, mode, vector,
                cfg.weights, top_k=cfg.mutation.top_k_per_position,
                min_abs_effect=cfg.mutation.min_abs_effect,
            )
        )
    proposals = _rank(proposals, vector)[: cfg.mutation.max_per_order * 4]
    return [[s] for s in proposals]


def _multi_site(
    cfg: MutagenesisConfig, job: ProteinJob, mode: str, vector: str, order: int,
    targets: list[TargetPosition],
) -> list[list[Substitution]]:
    wt = job.sequence.sequence
    # Best single substitution per position, then combine the top-leverage ones.
    best: dict[int, Substitution] = {}
    for tp in targets:
        cand = propose(
            wt, tp.position, tp.region_start, tp.region_stop, mode, vector,
            cfg.weights, top_k=1, min_abs_effect=cfg.mutation.min_abs_effect,
        )
        if cand:
            best[tp.position] = cand[0]
    leverage = _rank(list(best.values()), vector)[: cfg.mutation.top_k_positions]
    positions = [s.position for s in leverage]

    combos: list[list[Substitution]] = []
    for combo in combinations(sorted(positions), order):
        combos.append([best[p] for p in combo])
        if len(combos) >= cfg.mutation.max_per_order * 4:
            break  # bound enumeration before record-building
    return combos


def _rank_records(records: list[MutantRecord], vector: str) -> list[MutantRecord]:
    if vector == "activating":
        return sorted(records, key=lambda r: r.delta_load, reverse=True)
    if vector == "inhibiting":
        return sorted(records, key=lambda r: r.delta_load)
    return sorted(records, key=lambda r: abs(r.delta_load))


def _truncation_points(
    job: ProteinJob, targets: list[TargetPosition], truncate_at: str
) -> list[int]:
    if truncate_at == "region_start":
        pts = {tp.region_start for tp in targets if tp.source in {"consensus", "region"}}
        if not pts:  # fall back to positions if no regions were given
            pts = {tp.position for tp in targets}
    else:
        pts = {tp.position for tp in targets}
    return sorted(p for p in pts if p > 1)


def generate_for_protein(
    cfg: MutagenesisConfig, job: ProteinJob
) -> list[MutantRecord]:
    """Ranked, capped mutant records for a single protein.

    Within each (grammar, vector, order) cell, candidates are ranked and capped
    by their global aggregation-load change, the same quantity reported in the
    manifest, so the output reads top-down by predicted impact.
    """
    targets = resolve_targets(cfg, job)
    records: list[MutantRecord] = []
    counter = 0

    for mode in cfg.modes:
        for vector in cfg.substitution_vectors:
            for order in cfg.mutation.orders:
                groups = (
                    _single_site(cfg, job, mode, vector, targets)
                    if order == 1
                    else _multi_site(cfg, job, mode, vector, order, targets)
                )
                cell: list[MutantRecord] = []
                for subs in groups:
                    counter += 1
                    cell.append(
                        _make_record(cfg, job, counter, mode, vector, order, subs, None)
                    )
                records.extend(
                    _rank_records(cell, vector)[: cfg.mutation.max_per_order]
                )

    if "nonsense" in cfg.vectors:
        # Truncations are grammar-agnostic; load delta is computed on the
        # variant's own (shortened) sequence regardless of attributed mode.
        mode = cfg.modes[0]
        points = _truncation_points(job, targets, cfg.mutation.truncate_at)
        for point in points:
            counter += 1
            records.append(
                _make_record(cfg, job, counter, mode, "nonsense", 1, [], point)
            )

    # De-duplicate identical sequences, then cap the per-protein total.
    seen: set[str] = set()
    unique: list[MutantRecord] = []
    for rec in records:
        if rec.sequence in seen or rec.sequence == job.sequence.sequence:
            continue
        seen.add(rec.sequence)
        unique.append(rec)
    return unique[: cfg.mutation.max_total]


def generate(cfg: MutagenesisConfig) -> list[MutantRecord]:
    """Mutant records for every protein in the panel, concatenated."""
    records: list[MutantRecord] = []
    for job in cfg.proteins:
        records.extend(generate_for_protein(cfg, job))
    return records
