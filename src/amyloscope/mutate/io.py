"""Output writers and the top-level mutagenesis run.

For each protein in the panel two artifacts are written, named by protein id: a
FASTA of mutant sequences (wild type first, so a re-scoring run has its own
baseline) ready to feed to the external predictor panel, and a TSV manifest
recording each variant's grammar, vector, edit, predicted aggregation-load
change, and charge-patterning shifts. When the panel holds more than one
protein, a combined manifest spanning all of them is written under the run name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import MutagenesisConfig
from .engine import MutantRecord, generate_for_protein

_MANIFEST_COLS = [
    "protein", "mutant_id", "mode", "vector", "order", "mutation", "length",
    "delta_load", "delta_NCPR", "delta_FCR", "delta_SCD",
]


@dataclass
class MutagenesisArtifacts:
    records: list[MutantRecord]
    written_files: list[str]


def _wrap(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def write_fasta(
    seq_id: str, wt_seq: str, records: list[MutantRecord], path: Path
) -> None:
    """Write the wild type followed by every mutant for one protein."""
    lines = [f">{seq_id}|wildtype|len={len(wt_seq)}", _wrap(wt_seq)]
    for rec in records:
        header = (
            f">{rec.id}|{rec.mode}|{rec.vector}|{rec.mutation_label}"
            f"|dLoad={rec.delta_load:+.3f}"
        )
        lines.append(header)
        lines.append(_wrap(rec.sequence))
    path.write_text("\n".join(lines) + "\n")


def write_manifest(records: list[MutantRecord], path: Path) -> None:
    """Write a TSV manifest (one row per variant), protein-labelled."""
    rows = ["\t".join(_MANIFEST_COLS)]
    for r in records:
        rows.append(
            "\t".join(
                str(x)
                for x in (
                    r.protein, r.id, r.mode, r.vector, r.order, r.mutation_label,
                    len(r.sequence), f"{r.delta_load:+.4f}",
                    f"{r.delta_ncpr:+.4f}", f"{r.delta_fcr:+.4f}",
                    f"{r.delta_scd:+.4f}",
                )
            )
        )
    path.write_text("\n".join(rows) + "\n")


def run(cfg: MutagenesisConfig) -> MutagenesisArtifacts:
    """Generate variants for every protein and write per-protein artifacts."""
    cfg.validate()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[MutantRecord] = []
    written: list[str] = []
    for job in cfg.proteins:
        recs = generate_for_protein(cfg, job)
        all_records.extend(recs)
        fasta_path = out_dir / f"{job.sequence.id}_mutants.fasta"
        manifest_path = out_dir / f"{job.sequence.id}_manifest.tsv"
        write_fasta(job.sequence.id, job.sequence.sequence, recs, fasta_path)
        write_manifest(recs, manifest_path)
        written += [str(fasta_path), str(manifest_path)]

    if len(cfg.proteins) > 1:
        combined = out_dir / f"{cfg.name}_manifest.tsv"
        write_manifest(all_records, combined)
        written.append(str(combined))

    return MutagenesisArtifacts(records=all_records, written_files=written)


def run_from_file(path: str | Path) -> MutagenesisArtifacts:
    """Load a mutagenesis YAML config and run it."""
    from .config import load_mutate_config

    return run(load_mutate_config(path))
