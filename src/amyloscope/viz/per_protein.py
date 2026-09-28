"""Per-protein positional small-multiples figure.

One mini decile panel per protein, shared axes, each annotated with the region
count, the mean position, and — where the protein has enough regions to resolve
it — the within-protein clustering p. This is the figure that makes the
Simpson's-paradox case visible: two proteins with opposite positional bias that
pool to an unremarkable panel-level histogram are seen here to be individually
skewed in opposite directions.

Reads the per-protein positional description already computed in
:func:`amyloscope.analysis.statistics.compute_statistics` (``stats.per_protein_positional``)
rather than recomputing it, so the figure and the report cannot disagree.
"""

from __future__ import annotations

import numpy as np

from ..analysis.statistics import ConsensusStatistics
from ..core.consensus import ConsensusResult
from . import labels as _labels
from . import style as _style


def plot_per_protein_positional(
    result: ConsensusResult, stats: ConsensusStatistics, figsize=None
):
    import matplotlib.pyplot as plt

    config = result.config
    viz = config.viz
    _style.apply_style(viz)
    tr = _labels.translator(viz.language)

    rows = list(getattr(stats, "per_protein_positional", []))
    fs = getattr(viz, "resolved_figure_scale", 1.0)

    if not rows:
        fig, ax = plt.subplots(figsize=(8 * fs, 3 * fs), facecolor="white")
        ax.text(0.5, 0.5, tr("no_regions"), ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    n = len(rows)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    figsize = figsize or (4.2 * ncol * fs, 2.8 * nrow * fs)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, facecolor="white",
                             squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.ravel()

    centers = np.linspace(0.05, 0.95, 10)
    y_max = max((max(pp.decile_counts) for pp in rows), default=1)

    for ax, pp in zip(axes_flat, rows):
        # Bars shaded N->C so the direction of any skew is legible at a glance.
        cmap = _style.get_cmap("RdYlBu_r")
        colours = [cmap(0.1 + 0.8 * k / 9) for k in range(10)]
        ax.bar(centers, pp.decile_counts, width=0.09, color=colours,
               edgecolor="#333333", linewidth=0.4, alpha=0.85)
        ax.axvline(pp.mean_position, color="#333333", linestyle=":",
                   linewidth=1.0, alpha=0.7)

        label = config.label_for(pp.protein)
        if pp.clustering_p == pp.clustering_p:  # not NaN
            sig = ("*" if pp.clustering_p < 0.05 else "")
            sub = f"n={pp.n_regions}  clustering p={pp.clustering_p:.3f}{sig}"
        else:
            sub = f"n={pp.n_regions}  (p n/a, <4 regions)"
        ax.set_title(f"{label}\n{sub}", fontsize=12, linespacing=1.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, y_max + 0.5)
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.tick_params(labelsize="x-small")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # Blank any unused cells in the grid.
    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel(tr("axis_norm_position"), fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel(tr("axis_n_regions"), fontsize=10)

    fig.suptitle(tr("title_enrichment"), fontweight="bold", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig
