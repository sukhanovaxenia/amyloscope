"""Domain-architecture figure.

Overlays consensus aggregation-prone regions on each protein's annotated
domains, one row per protein. Domain fills come from the abstract
category->colour palette in :class:`~amyloscope.config.VizConfig`, and APR rows
are coloured and stacked by evidence tier resolved from the configured fractions
— removing the hardcoded ribosomal domain categories and the fixed 8/6/4 tier
counts of the original.
"""

from __future__ import annotations

import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

from ..core.consensus import ConsensusResult
from . import style as _style


def plot_domain_architecture(result: ConsensusResult, figsize=(16, 10)):
    import matplotlib.pyplot as plt

    config = result.config
    _style.apply_style(config.viz)
    proteins = config.proteins

    fig, axes = plt.subplots(len(proteins), 1, figsize=figsize,
                             gridspec_kw={"hspace": 0.5})
    if len(proteins) == 1:
        axes = [axes]

    order = _style.tier_order(config.consensus)  # weakest -> strongest
    # APR rows ascend with tier strength; place above the backbone/domains.
    tier_y = {name: 2.0 + 0.35 * idx for idx, name in enumerate(reversed(order))}

    for ax, spec in zip(axes, proteins, strict=True):
        length = _length(result, spec.id)
        ax.set_xlim(-5, length + 10)
        ax.set_ylim(-0.5, 2.0 + 0.35 * len(order) + 0.6)

        # Backbone.
        ax.add_patch(Rectangle((1, 1.4), length, 0.4, facecolor="#E0E0E0",
                               edgecolor="black", linewidth=0.7, zorder=1))

        # Domains on a single lane beneath the backbone.
        for domain in sorted(spec.domains, key=lambda d: d.start):
            colour = _style.domain_color(domain.category, config.viz)
            ax.add_patch(
                FancyBboxPatch(
                    (domain.start, 0.9), domain.stop - domain.start + 1, 0.45,
                    boxstyle="round,pad=0.01", facecolor=colour, alpha=0.45,
                    edgecolor="black", linewidth=0.5, zorder=2,
                )
            )
            ax.text((domain.start + domain.stop) / 2, 1.125, domain.name,
                    ha="center", va="center", fontsize=8, zorder=10)

        # Consensus regions stacked by tier.
        for region in result.regions.get(spec.id, []):
            colour = _style.tier_color(region.tier, config.consensus)
            y = tier_y.get(region.tier, 2.0)
            ax.add_patch(Rectangle((region.start, y), region.length, 0.3,
                                   facecolor=colour, alpha=0.85, edgecolor="black",
                                   linewidth=0.5, zorder=5))
            ax.text(region.midpoint, y + 0.32, f"{region.start}-{region.end}",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_ylabel(spec.label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Position (aa)", fontsize=9)
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.15, linestyle=":", linewidth=0.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.text(length + 2, 1.6, f"{length} aa", fontsize=10, style="italic",
                color="#666666")

    legend = [
        patches.Patch(facecolor=_style.tier_color(t.name, config.consensus),
                      alpha=0.85, edgecolor="black", linewidth=0.5,
                      label=f"{t.name.capitalize()} consensus")
        for t in config.consensus.tiers_by_strength
    ]
    # Show the active domain categories present in the panel.
    seen = {d.category for p in proteins for d in p.domains}
    for cat in sorted(seen):
        legend.append(
            patches.Patch(facecolor=_style.domain_color(cat, config.viz), alpha=0.45,
                          edgecolor="black", linewidth=0.5, label=f"domain: {cat}")
        )

    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.92, 0.5),
               fontsize=8, title="Legend", title_fontsize=11)
    fig.suptitle(f"Domain architecture and consensus APR mapping — {config.name}",
                 fontsize=14, fontweight="bold", y=0.98)
    # An explicit adjustment is used instead of tight_layout because the legend
    # is anchored outside the axes (right margin), which tight_layout cannot
    # account for; reserving the margins directly avoids a layout warning.
    fig.subplots_adjust(left=0.08, right=0.9, top=0.93, bottom=0.06, hspace=0.5)
    return fig


def _length(result: ConsensusResult, protein_id: str) -> int:
    spec = result.config.protein(protein_id)
    if spec.length:
        return spec.length
    regs = result.regions.get(protein_id, [])
    return max((r.end for r in regs), default=100)
