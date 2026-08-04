"""Consensus-region figures.

Two views, both generalised away from the eight-tool / ribosomal-naming
assumptions of the original:

* :func:`plot_distribution` — a genome-browser-style stratified track per
  protein, one sub-track per evidence tier, with the protein's true length as
  the x-extent (the original hardcoded ``xlim(0, 250)`` truncated longer
  proteins).
* :func:`plot_positional_enrichment` — the decile histogram with the uniform
  null and the chi-square annotation, the visual companion to the positional
  test in :mod:`amyloscope.analysis.statistics`.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from ..core.consensus import ConsensusResult
from . import labels as _labels
from . import style as _style


def plot_distribution(result: ConsensusResult, figsize=None):
    import matplotlib.pyplot as plt

    config = result.config
    viz = config.viz
    _style.apply_style(viz)
    tr = _labels.translator(viz.language)

    proteins = config.proteins
    n = len(proteins)
    lane_step = 1.5  # vertical distance between protein lanes (room for labels)
    base_h = max(3.5, lane_step * n + 1.5)
    figsize = figsize or viz.figsize(14, base_h)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    max_len = max((_length(result, p.id) for p in proteins), default=100)
    axis_range = max_len + viz.x_padding + 5
    axis_width_in = figsize[0] * 0.80           # axes span, legend sits outside
    label_pt = viz.resolved_base_font * 0.6     # region labels are xx-small

    for i, spec in enumerate(proteins):
        y0 = i * lane_step
        length = _length(result, spec.id)
        ax.add_patch(
            Rectangle(
                (0, y0 - 0.45), length, 0.9,
                facecolor="#F7F7F7", edgecolor="#D0D0D0", linewidth=0.5, zorder=1,
            )
        )
        ax.plot([length, length], [y0 - 0.45, y0 + 0.45], color="#A0A0A0",
                linewidth=1, linestyle=":", alpha=0.6, zorder=2)
        ax.text(length + 2, y0, f"{length}", fontsize="small", va="center",
                color="#808080", style="italic")

    # Vertical offset per tier (weakest at the bottom track).
    order = _style.tier_order(config.consensus)
    offsets = {
        name: (idx - (len(order) - 1) / 2) * 0.16
        for idx, name in enumerate(order)
    }

    for protein_idx, spec in enumerate(proteins):
        y0 = protein_idx * lane_step
        regs = result.regions.get(spec.id, [])
        rows = _style.stagger_label_rows(
            [r.midpoint for r in regs], axis_range, axis_width_in, label_pt
        )
        for region, row in zip(regs, rows, strict=True):
            colour = _style.tier_color(region.tier, config.consensus)
            y = y0 + offsets.get(region.tier, 0.0)
            ax.add_patch(
                FancyBboxPatch(
                    (region.start, y - 0.075), region.length, 0.15,
                    boxstyle="round,pad=0.001,rounding_size=0.015",
                    facecolor=colour, alpha=0.9, edgecolor="none", zorder=10,
                )
            )
            ax.text(region.midpoint, y + 0.12 + row * 0.18,
                    f"{region.start}-{region.end}",
                    ha="center", va="bottom", fontsize="xx-small",
                    # fontweight="bold", color="#1a1a1a", zorder=20)
                    color="#1a1a1a", zorder=20)

    ax.set_xlim(-5, max_len + viz.x_padding)
    ax.set_ylim(-lane_step * 0.6, (n - 1) * lane_step + lane_step * 0.75)
    ax.set_xlabel(tr("axis_position"))
    ax.set_ylabel(tr("axis_protein"))
    ax.set_yticks([i * lane_step for i in range(n)])
    ax.set_yticklabels([p.label for p in proteins])
    ax.set_title(tr("title_consensus", name=config.name), pad=10, fontweight="bold")

    legend = [
        mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor=_style.tier_color(t.name, config.consensus),
            alpha=0.85, label=_tier_label(t, config),
        )
        for t in config.consensus.tiers_by_strength
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1),
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(True, axis="x", alpha=0.1, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_positional_enrichment(result: ConsensusResult, figsize=None):
    import matplotlib.pyplot as plt
    import scipy.stats as sc_stats
    import seaborn as sns

    config = result.config
    _style.apply_style(config.viz)
    tr = _labels.translator(config.viz.language)

    positions = []
    for protein_id, regs in result.regions.items():
        plen = _length(result, protein_id)
        positions += [r.normalized_position(plen) for r in regs]
    positions = np.array([p for p in positions if 0.0 <= p <= 1.0])

    figsize = figsize or config.viz.figsize(14, 8)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    if positions.size == 0:
        ax.text(0.5, 0.5, tr("no_regions"), ha="center", va="center",
                transform=ax.transAxes)
        return fig

    bins = np.linspace(0, 1, 11)
    counts, edges = np.histogram(positions, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    expected = len(positions) / 10
    chi2, p = sc_stats.chisquare(counts, [expected] * 10)

    # The chi-square test assumes an expected count of at least ~5 per cell.
    # With nine consensus regions pooled across two proteins the expectation is
    # 0.9 per decile, so the asymptotic p-value is not valid and reporting it
    # unqualified invites a referee to discard the whole panel. Flag it rather
    # than suppress it: the histogram is still a legitimate description of where
    # the regions fall, it is only the test statistic that is uninterpretable.
    underpowered = expected < 5.0

    colours = sns.color_palette("RdYlBu_r", n_colors=10)
    ax.bar(centers, counts, width=0.08, color=colours, edgecolor="#333333",
           linewidth=0.6, alpha=0.85)
    ax.axhline(expected, color="#CC0000", linestyle="--", linewidth=1.5,
               alpha=0.7, label=tr("expected_uniform"))

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    note = (f"$\\chi^2$ = {chi2:.2f}\n$p$ = {p:.3f}   {sig}\n"
            f"$n$ = {len(positions)} regions")
    if underpowered:
        note += f"\nexpected {expected:.1f}/bin \u2014 asymptotic $p$ unreliable"
    ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize="small", va="top",
            linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="#FFF8E1" if underpowered else "#F4F7F9",
                      edgecolor="#B08D2E" if underpowered else "#9AA7B0",
                      linewidth=0.9))

    ax.set_xlabel(tr("axis_norm_position"))
    ax.set_ylabel(tr("axis_n_regions"))
    ax.set_title(tr("title_enrichment"), pad=10)

    # Bin EDGES, not centres. The centres are 0.05, 0.15, ... 0.95, and
    # formatting them to one decimal produced "0.2" and "0.9" twice each, so two
    # pairs of adjacent bars carried identical labels and the axis could not be
    # read unambiguously. Edges also state what each bar covers, which is the
    # quantity a reader needs, and they place a tick at every decile boundary
    # rather than in the middle of a bar.
    ax.set_xticks(edges)
    ax.set_xticklabels([f"{e:.1f}" for e in edges])
    ax.set_xlim(-0.02, 1.02)
    # Counts are integers; a fractional y tick is meaningless.
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax.legend(loc="upper right", frameon=True, framealpha=0.95,
              edgecolor="#CCCCCC")
    sns.despine(ax=ax)
    ax.grid(True, axis="y", alpha=0.15, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _length(result: ConsensusResult, protein_id: str) -> int:
    spec = result.config.protein(protein_id)
    if spec.length:
        return spec.length
    regs = result.regions.get(protein_id, [])
    return max((r.end for r in regs), default=100)


def _tier_label(tier, config) -> str:
    from ..core.consensus import resolve_count

    n = len(config.enabled_tools)
    return f"{tier.name.capitalize()} (>={resolve_count(tier.min_fraction, n)}/{n})"