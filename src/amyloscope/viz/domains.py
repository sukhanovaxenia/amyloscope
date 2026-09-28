"""Domain-architecture figure.

Overlays consensus aggregation-prone regions on each protein's annotated
domains, one row per protein. Domain fills come from the abstract
category->colour palette in :class:`~amyloscope.config.VizConfig`, and APR rows
are coloured and stacked by evidence tier resolved from the configured
fractions.

Tier rows run weakest at the bottom to strongest at the top, matching the
consensus-region track in :mod:`amyloscope.viz.detailed`. The previous
``enumerate(reversed(order))`` placed the strongest tier lowest, so the same
region appeared above the moderate tier in one figure and below it in the
other.
"""

from __future__ import annotations

import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

from ..core.consensus import ConsensusResult
from . import labels as _labels
from . import style as _style


def plot_domain_architecture(result: ConsensusResult, figsize=None):
    import matplotlib.pyplot as plt

    config = result.config
    viz = config.viz
    _style.apply_style(viz)
    tr = _labels.translator(viz.language)
    proteins = config.proteins
    n = len(proteins)
    order = _style.tier_order(config.consensus)  # weakest -> strongest

    # ------------------------------------------------------------------ #
    # Vertical layout, in data units, bottom -> top. These are the knobs:
    # raise REGION_Y0 to push the APR tiers further off the backbone, and
    # TIER_STEP / LABEL_DY to spread the evidence tiers and their labels.
    # ------------------------------------------------------------------ #
    DOMAIN_Y, DOMAIN_H = 0.0, 0.55       # domain lane
    BACKBONE_Y, BACKBONE_H = 0.72, 0.40  # backbone, above the domain lane
    REGION_Y0 = 1.80                     # baseline of the lowest evidence tier
    TIER_STEP = 0.52                     # gap between evidence tiers
    REGION_H = 0.30                      # consensus-region box height
    LABEL_DY = 0.40                      # label lift per stagger row
    DOM_LABEL_DY = 0.34                  # domain-name row step (stacked below lane)

    # Figure width is independent of height, so resolve the label rows up
    # front: this sizes the canvas for the densest panel and lets every
    # subplot share one y-range (consistent spacing, aligned baselines).
    fs = viz.resolved_figure_scale
    fig_w = 16 * fs
    axis_width_in = fig_w * 0.82
    label_pt = viz.resolved_base_font * 0.8
    rows_by_protein = {
        spec.id: _style.stagger_label_rows(
            [r.midpoint for r in result.regions.get(spec.id, [])],
            _length(result, spec.id) + 15, axis_width_in, label_pt,
        )
        for spec in proteins
    }
    global_max_row = max(
        (max(rs, default=0) for rs in rows_by_protein.values()), default=0
    )

    # Domain names collide horizontally when a protein has many/narrow domains,
    # so stagger them below the lane on the same principle as the APR labels.
    dom_rows_by_protein = {}
    for spec in proteins:
        doms = sorted(spec.domains, key=lambda d: d.start)
        centers = [(d.start + d.stop) / 2 for d in doms]
        n_chars = min(32, max((len(d.name) for d in doms), default=8))
        dom_rows_by_protein[spec.id] = _style.stagger_label_rows(
            centers, _length(result, spec.id) + 15, axis_width_in,
            viz.resolved_base_font * 0.7, n_chars=n_chars,
        )
    global_max_dom_row = max(
        (max(rs, default=0) for rs in dom_rows_by_protein.values()), default=0
    )

    y_bottom = DOMAIN_Y - 0.14 - (global_max_dom_row + 1) * DOM_LABEL_DY - 0.10
    y_top = (REGION_Y0 + TIER_STEP * (len(order) - 1) + REGION_H + 0.06
             + (global_max_row + 1) * LABEL_DY + 0.15)

    # Content-driven height: ~3 in per panel, more when either the APR labels
    # or the domain-name labels stack into extra rows.
    fig_h = max(8.0, (3.0 + 0.30 * global_max_row
                      + 0.22 * global_max_dom_row) * n)
    figsize = figsize or (fig_w, fig_h)

    fig, axes = plt.subplots(n, 1, figsize=figsize, gridspec_kw={"hspace": 0.4})
    if n == 1:
        axes = [axes]

    # order is weakest-first, so index 0 (lowest row) is the weakest tier —
    # the same vertical convention as the detailed consensus-region track.
    tier_y = {name: REGION_Y0 + TIER_STEP * idx
              for idx, name in enumerate(order)}

    for ax, spec in zip(axes, proteins, strict=True):
        length = _length(result, spec.id)
        regs = result.regions.get(spec.id, [])
        rows = rows_by_protein[spec.id]
        ax.set_xlim(-5, length + 10)
        ax.set_ylim(y_bottom, y_top)

        # Backbone.
        ax.add_patch(Rectangle((1, BACKBONE_Y), length, BACKBONE_H,
                               facecolor="#E0E0E0", edgecolor="black",
                               linewidth=0.7, zorder=1))

        # Domain lane beneath the backbone; names staggered below the lane with
        # thin leaders so many/long names don't overprint each other.
        doms = sorted(spec.domains, key=lambda d: d.start)
        dom_rows = dom_rows_by_protein[spec.id]
        for domain, drow in zip(doms, dom_rows, strict=True):
            colour = _style.domain_color(domain.category, config.viz)
            ax.add_patch(FancyBboxPatch(
                (domain.start, DOMAIN_Y), domain.stop - domain.start + 1, DOMAIN_H,
                boxstyle="round,pad=0.01", facecolor=colour, alpha=0.45,
                edgecolor="black", linewidth=0.5, zorder=2))
            cx = (domain.start + domain.stop) / 2
            label_y = DOMAIN_Y - 0.14 - drow * DOM_LABEL_DY
            ax.plot([cx, cx], [DOMAIN_Y, label_y + 0.04], color="#AAAAAA",
                    linewidth=0.5, zorder=3)
            name = domain.name if len(domain.name) <= 44 else domain.name[:42] + "…"
            ax.text(cx, label_y, name, ha="center", va="top",
                    fontsize="x-small", zorder=10)

        # Consensus regions stacked by tier; labels staggered to avoid overlap.
        for region, row in zip(regs, rows, strict=True):
            colour = _style.tier_color(region.tier, config.consensus)
            y = tier_y.get(region.tier, REGION_Y0)
            ax.add_patch(Rectangle((region.start, y), region.length, REGION_H,
                                   facecolor=colour, alpha=0.85, edgecolor="black",
                                   linewidth=0.5, zorder=5))
            ax.text(region.midpoint, y + REGION_H + 0.06 + row * LABEL_DY,
                    f"{region.start}-{region.end}", ha="center", va="bottom",
                    fontsize="small", fontweight="bold")

        ax.set_ylabel(spec.label, fontweight="bold", fontsize=14)
        if ax is axes[-1]:
            ax.set_xlabel(tr("axis_position_aa"), fontweight="bold", fontsize=14)
        ax.set_yticks([])
        ax.tick_params(labelsize=12)
        ax.grid(True, axis="x", alpha=0.15, linestyle=":", linewidth=0.5)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.text(length + 2, BACKBONE_Y + BACKBONE_H / 2,
                tr("length_aa", length=length), fontsize="small",
                style="italic", color="#666666", va="center")

    # Legend ordered strongest-first to match the visual stacking read top-down,
    # and localised through the label table so a Russian run is not half English.
    legend = [
        patches.Patch(facecolor=_style.tier_color(t.name, config.consensus),
                      alpha=0.85, edgecolor="black", linewidth=0.5,
                      label=tr("legend_tier_consensus",
                               tier=_labels.tier_name(viz.language, t.name)))
        for t in config.consensus.tiers_by_strength
    ]
    # Show the active domain categories present in the panel.
    seen = {d.category for p in proteins for d in p.domains}
    for cat in sorted(seen):
        legend.append(
            patches.Patch(facecolor=_style.domain_color(cat, config.viz), alpha=0.45,
                          edgecolor="black", linewidth=0.5,
                          label=tr("legend_domain", cat=cat))
        )

    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(0.92, 0.5),
               title=tr("legend_title"))
    fig.suptitle(tr("title_domains", name=config.name),
                 fontweight="bold", y=0.99)
    # An explicit adjustment is used instead of tight_layout because the legend
    # is anchored outside the axes (right margin), which tight_layout cannot
    # account for; reserving the margins directly avoids a layout warning.
    fig.subplots_adjust(left=0.10, right=0.9, top=0.95, bottom=0.05, hspace=0.4)
    return fig


def _length(result: ConsensusResult, protein_id: str) -> int:
    spec = result.config.protein(protein_id)
    if spec.length:
        return spec.length
    regs = result.regions.get(protein_id, [])
    return max((r.end for r in regs), default=100)