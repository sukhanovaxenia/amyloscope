"""End-to-end orchestration.

One call takes a :class:`~amyloscope.config.PipelineConfig` to the full artefact
set: consensus regions (TSV), the statistical and domain-overlap reports, and
the figures. This is the generalised, config-driven replacement for the
hardcoded ``main()`` in the original ``complete_analysis_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .analysis.domains import compute_domain_overlap, format_overlap_report
from .analysis.statistics import compute_statistics, format_report
from .config import PipelineConfig, load_config
from .core.consensus import ConsensusResult, compute_consensus
from .io.loader import Dataset, load_dataset
from .viz import consensus as viz_consensus
from .viz import domains as viz_domains
from .viz import style as viz_style
from .viz import tracks as viz_tracks


@dataclass
class PipelineArtifacts:
    """Handles to the in-memory results and the files written to disk."""

    dataset: Dataset
    consensus: ConsensusResult
    written_files: list[str] = field(default_factory=list)


def run(config: PipelineConfig, make_figures: bool = True) -> PipelineArtifacts:
    """Execute the full pipeline for a resolved configuration."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    dataset = load_dataset(config)
    result = compute_consensus(dataset)

    # Consensus regions table.
    regions_tsv = out_dir / "consensus_regions.tsv"
    result.to_dataframe().to_csv(regions_tsv, sep="\t", index=False)
    written.append(str(regions_tsv))

    # Statistical report.
    stats = compute_statistics(result)
    stats_path = out_dir / "consensus_statistics.txt"
    stats_path.write_text(format_report(stats, config), encoding="utf-8")
    written.append(str(stats_path))

    # Domain overlap report (only if any protein declares domains).
    if any(p.domains for p in config.proteins):
        overlap = compute_domain_overlap(result)
        overlap_path = out_dir / "domain_overlap.txt"
        overlap_path.write_text(format_overlap_report(overlap), encoding="utf-8")
        written.append(str(overlap_path))

    if make_figures:
        written += _render_figures(dataset, result, out_dir)

    return PipelineArtifacts(dataset=dataset, consensus=result, written_files=written)


def run_from_file(
    config_path: str | Path, make_figures: bool = True
) -> PipelineArtifacts:
    """Convenience wrapper: load a YAML config and run it."""
    return run(load_config(config_path), make_figures=make_figures)


def _render_figures(dataset, result, out_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    config = dataset.config
    written: list[str] = []

    def _save(fig, name: str) -> None:
        written.extend(viz_style.save_figure(fig, out_dir / name, config.viz))
        plt.close(fig)

    # Panel-level consensus figures.
    _save(viz_consensus.plot_distribution(result), "consensus_distribution.png")
    _save(
        viz_consensus.plot_positional_enrichment(result),
        "positional_enrichment.png",
    )

    if any(p.domains for p in config.proteins):
        _save(
            viz_domains.plot_domain_architecture(result),
            "domain_architecture.png",
        )

    # Per-protein predictor tracks.
    for protein_tracks in dataset:
        if not protein_tracks.tracks:
            continue
        _save(
            viz_tracks.plot_tool_tracks(protein_tracks, config),
            f"{protein_tracks.spec.id}_tracks.png",
        )

    return written
