"""End-to-end orchestration.

One call takes a :class:`~amyloscope.config.PipelineConfig` to the full artefact
set: consensus regions (TSV), the statistical and domain-overlap reports, and
the figures. This is the generalised, config-driven replacement for the
hardcoded ``main()`` in the original ``complete_analysis_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .analysis.clusters import build_clusters, format_cluster_report
from .analysis.independence import check_independence, pairwise_overlap
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

    # APR clusters: the peak-agreement core inside each called region, with a
    # provenance verdict on the shoulder. Written unconditionally because the
    # table is the deliverable; the figure is optional.
    # Declared non-independence: warns, and records the effective denominator.
    independence = check_independence(config)
    overlap_rows = pairwise_overlap(dataset, config)
    independence_path = out_dir / "predictor_independence.txt"
    text = independence.as_text() + (
        "\n\nMeasured pairwise overlap (top 12 by containment).\n"
        "Read containment WITH breadth_ratio: a narrow caller is almost always\n"
        "contained in a broad one by arithmetic alone. High containment at a\n"
        "breadth_ratio near 1 -- two tools of similar reach calling the same\n"
        "residues -- is the pattern that suggests a shared signal.\n\n")
    for row in overlap_rows[:12]:
        text += (f"  {row['tool_a']:16s} {row['tool_b']:16s} "
                 f"jaccard={row['jaccard']:.3f} containment={row['containment_min']:.3f} "
                 f"breadth_ratio={row['breadth_ratio']:.3f} shared={row['shared_aa']}\n")
    independence_path.write_text(text, encoding="utf-8")
    written.append(str(independence_path))
    import pandas as _pd
    overlap_path = out_dir / "predictor_overlap.tsv"
    _pd.DataFrame(overlap_rows).to_csv(overlap_path, sep="\t", index=False)
    written.append(str(overlap_path))

    clusters = build_clusters(dataset, result)
    cluster_path = out_dir / "apr_clusters.tsv"
    clusters.to_dataframe().to_csv(cluster_path, sep="\t", index=False)
    written.append(str(cluster_path))
    breadth_path = out_dir / "predictor_breadth.tsv"
    clusters.breadth_to_dataframe().to_csv(breadth_path, sep="\t", index=False)
    written.append(str(breadth_path))
    report_path = out_dir / "apr_clusters.txt"
    report_path.write_text(format_cluster_report(clusters), encoding="utf-8")
    written.append(str(report_path))

    if make_figures:
        written += _render_figures(dataset, result, stats, out_dir, overlap=overlap,
                                   clusters=clusters)

    return PipelineArtifacts(dataset=dataset, consensus=result,
                             written_files=written, overlap=overlap)


def run_from_file(
    config_path: str | Path, make_figures: bool = True
) -> PipelineArtifacts:
    """Convenience wrapper: load a YAML config and run it."""
    return run(load_config(config_path), make_figures=make_figures)


def _render_figures(dataset, result, stats, out_dir: Path, overlap=None,
                    clusters=None) -> list[str]:
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
    from .viz import per_protein as viz_per_protein

    config = dataset.config
    written: list[str] = []

    def _save(fig, name: str) -> None:
        if fig is None:
            return
        written.extend(viz_style.save_figure(fig, out_dir / name, config.viz))
        plt.close(fig)

    # ---- panel-level figures ---------------------------------------------- #
    _save(viz_consensus.plot_distribution(result), "consensus_distribution.png")
    # Rendered ONCE. The call was duplicated verbatim, so the figure was built
    # and written twice per run and appeared twice in `written` -- which made the
    # artifact list look like two different files and doubled the cost of the
    # panel's second-most expensive figure.
    _save(viz_consensus.plot_positional_enrichment(result, stats),
          "positional_enrichment.png")
    # Per-protein positional breakdown: the panel-level enrichment pools every
    # protein, which hides whether a positional trend is shared or driven by one
    # chain. With a panel this small that distinction decides the interpretation.
    _save(viz_per_protein.plot_per_protein_positional(result, stats),
          "per_protein_positional.png")
    if clusters is not None:
        from .viz import clusters as viz_clusters
        _save(viz_clusters.plot_cluster_cores(clusters, config), "apr_clusters.png")

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