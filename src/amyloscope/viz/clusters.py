"""Core-versus-extent figure for APR clusters.

One row per cluster. The extent is drawn as a light bar in the tier the region
was actually called at; the peak-agreement core is drawn on top of it in its own
tier colour. Reading the two marks against each other is the whole point — the
figure exists to stop a narrow high-agreement core and its wider low-agreement
surround being collapsed into a single claim.

Colours come from the configured tier palette, so the figure stays on the same
ordinal ramp as every other consensus figure in the package rather than
introducing a second colour language for the same quantity.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from . import style as _style
from ..analysis.clusters import (
    EXTENSION_SUPPORTED,
    WINDOW_ARTIFACT,
    ClusterResult,
)

#: Marker drawn beside a cluster whose shoulder has a provenance verdict.
_VERDICT_MARK = {
    EXTENSION_SUPPORTED: ("✓", "narrow caller extends"),
    WINDOW_ARTIFACT: ("✗", "broad callers only"),
}


def _tier_colours(config) -> dict[str, str]:
    return {t.name: t.color for t in config.consensus.tiers}


def plot_cluster_cores(result: ClusterResult, config, *, width_in: float = 9.0):
    """Horizontal core/extent chart, newest-to-oldest by protein then position."""
    clusters = [c for c in result.all_clusters()]
    if not clusters:
        return None

    colours = _tier_colours(config)
    default = "#999999"
    height = max(2.2, 0.42 * len(clusters) + 1.4)
    fig, ax = plt.subplots(figsize=(width_in, height))
    fonts = _style.figure_fonts(config.viz, width_in)

    labels = []
    for row, c in enumerate(reversed(clusters)):
        y = row
        ax.barh(
            y, c.extent_length, left=c.extent_start - 0.5, height=0.62,
            color=colours.get(c.extent_tier, default), alpha=0.28,
            edgecolor=colours.get(c.extent_tier, default), linewidth=0.8,
        )
        ax.barh(
            y, c.core_length, left=c.core_start - 0.5, height=0.62,
            color=colours.get(c.core_tier, default), alpha=0.95, linewidth=0,
        )
        mark = _VERDICT_MARK.get(c.verdict)
        if mark:
            ax.text(
                c.extent_start + c.extent_length + 1.5, y, mark[0],
                va="center", ha="left", fontsize=fonts["label"],
                color="#333333",
            )
        labels.append(
            f"{c.protein} {c.extent_start}-{c.extent_stop}"
            + (f"  {c.core_sequence}" if c.core_sequence else "")
        )

    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(labels, fontsize=fonts["tick"])
    ax.set_xlabel("residue position", fontsize=fonts["label"])
    ax.tick_params(axis="x", labelsize=fonts["tick"])
    ax.set_title(
        "APR clusters: peak-agreement core (solid) within the called region (pale)",
        fontsize=fonts["title"], loc="left",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours.get(t.name, default))
        for t in config.consensus.tiers
    ]
    legend_labels = [t.name for t in config.consensus.tiers]
    for verdict, (glyph, text) in _VERDICT_MARK.items():
        handles.append(plt.Line2D([], [], linestyle="none", marker="$%s$" % glyph,
                                  color="#333333"))
        legend_labels.append(f"shoulder: {text}")
    ax.legend(handles, legend_labels, fontsize=fonts["tick"], loc="lower right",
              frameon=False, ncol=2)
    ax.margins(x=0.04)
    fig.tight_layout()
    return fig
