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


@dataclass
class PipelineArtifacts:
    """Handles to the in-memory results and the files written to disk."""

    dataset: Dataset
    consensus: ConsensusResult
    written_files: list[str] = field(default_factory=list)
    #: Domain-overlap analysis, when any protein declares domains. Exposed so a
    #: caller working in Python has the enrichment table without re-running the
    #: permutation null.
    overlap: object | None = None


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

    # Domain overlap (only if any protein declares domains). The result is kept
    # rather than discarded: the enrichment null is the expensive part of the
    # analysis and the figure needs the same draws the report was written from,
    # so recomputing it in the plotting layer would be both slow and capable of
    # disagreeing with the text.
    overlap = None
    if any(p.domains for p in config.proteins):
        overlap = compute_domain_overlap(result)
        overlap_path = out_dir / "domain_overlap.txt"
        overlap_path.write_text(format_overlap_report(overlap), encoding="utf-8")
        written.append(str(overlap_path))

    if make_figures:
        written += _render_figures(dataset, result, out_dir, overlap=overlap)

    return PipelineArtifacts(dataset=dataset, consensus=result,
                             written_files=written, overlap=overlap)


def run_from_file(
    config_path: str | Path, make_figures: bool = True
) -> PipelineArtifacts:
    """Convenience wrapper: load a YAML config and run it."""
    return run(load_config(config_path), make_figures=make_figures)


def _render_figures(dataset, result, out_dir: Path, overlap=None) -> list[str]:
    # Imported here, not at module scope, so `import amyloscope` and the
    # sequence-only modules (e.g. amyloscope.mutate) work without matplotlib;
    # the plotting stack is only needed when figures are actually rendered.
    import matplotlib.pyplot as plt

    from .viz import consensus as viz_consensus
    from .viz import correlation as viz_correlation
    from .viz import detailed as viz_detailed
    from .viz import domains as viz_domains
    from .viz import overlap as viz_overlap
    from .viz import style as viz_style
    from .viz import tracks as viz_tracks

    config = dataset.config
    written: list[str] = []

    def _save(fig, name: str) -> None:
        if fig is None:
            return
        written.extend(viz_style.save_figure(fig, out_dir / name, config.viz))
        plt.close(fig)

    # ---- panel-level figures ---------------------------------------------- #
    _save(viz_consensus.plot_distribution(result), "consensus_distribution.png")
    _save(viz_consensus.plot_positional_enrichment(result),
          "positional_enrichment.png")

    if any(p.domains for p in config.proteins):
        _save(viz_domains.plot_domain_architecture(result),
              "domain_architecture.png")

    # The overlap figure takes the analysis object, not the ConsensusResult the
    # other builders take, because the enrichment null belongs to the analysis
    # layer. Gated on that object existing rather than on the domains, so a run
    # that skipped the analysis cannot reach the figure with nothing to draw.
    if overlap is not None and getattr(overlap, "enrichment", None):
        _save(viz_overlap.plot_domain_overlap(overlap), "domain_overlap.png")

    if config.measurements and config.measurements.values:
        _save(
            viz_correlation.plot_measurement_correlation(
                result, config.measurements.values,
                dataset=dataset, config=config,
                measurement_label=config.measurements.label,
            ),
            "measurement_correlation.png",
        )

    # Cross-protein figures are panel-level and belong here, before the loop.
    # They previously sat inside it at the same indent as the per-protein
    # blocks, so each was rendered once per protein and overwritten every time
    # but the last — five renders of the two most expensive figures in the set.
    if getattr(config.viz, "detailed", False) and any(
        pt.tracks for pt in dataset
    ):
        _save(viz_detailed.plot_cross_protein_overview(result, dataset, config),
              "all_proteins_consensus.png")
        _save(viz_detailed.plot_all_proteins_agreement(result, dataset, config),
              "all_protein_agreement.png")

    # ---- per-protein figures ---------------------------------------------- #
    for protein_tracks in dataset:
        if not protein_tracks.tracks:
            continue
        _save(viz_tracks.plot_tool_tracks(protein_tracks, config),
              f"{protein_tracks.spec.id}_tracks.png")
        if getattr(config.viz, "detailed", False):
            _save(
                viz_detailed.plot_tool_analysis(protein_tracks, result, config),
                f"{protein_tracks.spec.id}_analysis.png",
            )
            _save(
                viz_detailed.plot_consensus_analysis(
                    protein_tracks, result, config
                ),
                f"{protein_tracks.spec.id}_consensus_analysis.png",
            )

    return written