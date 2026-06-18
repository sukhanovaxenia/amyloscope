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
from . import style as _style


def plot_distribution(result: ConsensusResult, figsize=(14, 8)):
    import matplotlib.pyplot as plt

    config = result.config
    _style.apply_style(config.viz)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    proteins = config.proteins
    max_len = max((_length(result, p.id) for p in proteins), default=100)

    for i, spec in enumerate(proteins):
        length = _length(result, spec.id)
        ax.add_patch(
            Rectangle(
                (0, i - 0.4), length, 0.8,
                facecolor="#F7F7F7", edgecolor="#D0D0D0", linewidth=0.5, zorder=1,
            )
        )
        ax.plot([length, length], [i - 0.4, i + 0.4], color="#A0A0A0",
                linewidth=1, linestyle=":", alpha=0.6, zorder=2)
        ax.text(length + 2, i, f"{length}", fontsize=7, va="center",
                color="#808080", style="italic")

    # Vertical offset per tier (weakest at the bottom track).
    order = _style.tier_order(config.consensus)
    offsets = {
        name: (idx - (len(order) - 1) / 2) * 0.25
        for idx, name in enumerate(order)
    }

    for protein_idx, spec in enumerate(proteins):
        for region in result.regions.get(spec.id, []):
            colour = _style.tier_color(region.tier, config.consensus)
            y = protein_idx + offsets.get(region.tier, 0.0)
            ax.add_patch(
                FancyBboxPatch(
                    (region.start, y - 0.075), region.length, 0.15,
                    boxstyle="round,pad=0.001,rounding_size=0.015",
                    facecolor=colour, alpha=0.85, edgecolor="none", zorder=10,
                )
            )
            ax.text(region.midpoint, y, f"{region.start}-{region.end}",
                    ha="center", va="center", fontsize=5,
                    color="white" if region.tier == order[-1] else "black", zorder=20)

    ax.set_xlim(-5, max_len + config.viz.x_padding)
    ax.set_ylim(-0.6, len(proteins) - 0.4)
    ax.set_xlabel("Amino acid position", fontsize=11)
    ax.set_ylabel("Protein", fontsize=11)
    ax.set_yticks(range(len(proteins)))
    ax.set_yticklabels([p.label for p in proteins], fontsize=10)
    ax.set_title(
        f"Consensus aggregation-prone regions — {config.name}", fontsize=12, pad=10
    )

    legend = [
        mpatches.Rectangle(
            (0, 0), 1, 1,
            facecolor=_style.tier_color(t.name, config.consensus),
            alpha=0.85, label=_tier_label(t, config),
        )
        for t in config.consensus.tiers_by_strength
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, frameon=True, framealpha=0.9, edgecolor="#CCCCCC")
    ax.grid(True, axis="x", alpha=0.1, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_positional_enrichment(result: ConsensusResult, figsize=(14, 8)):
    import matplotlib.pyplot as plt
    import scipy.stats as sc_stats
    import seaborn as sns

    config = result.config
    _style.apply_style(config.viz)

    positions = []
    for protein_id, regs in result.regions.items():
        plen = _length(result, protein_id)
        positions += [r.normalized_position(plen) for r in regs]
    positions = np.array([p for p in positions if 0.0 <= p <= 1.0])

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    if positions.size == 0:
        ax.text(0.5, 0.5, "No consensus regions", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    bins = np.linspace(0, 1, 11)
    counts, edges = np.histogram(positions, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    expected = len(positions) / 10
    chi2, p = sc_stats.chisquare(counts, [expected] * 10)

    colours = sns.color_palette("RdYlBu_r", n_colors=10)
    ax.bar(centers, counts, width=0.08, color=colours, edgecolor="#333333",
           linewidth=0.6, alpha=0.85)
    ax.axhline(expected, color="#CC0000", linestyle="--", linewidth=1.5,
               alpha=0.7, label="Expected (uniform)")

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text(0.05, 0.95, f"chi2 = {chi2:.2f}\np = {p:.4f}\n{sig}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFFCC",
                      edgecolor="#666666", linewidth=0.8))

    ax.set_xlabel("Normalised position (0 = N-terminus, 1 = C-terminus)", fontsize=11)
    ax.set_ylabel("Number of regions", fontsize=11)
    ax.set_title("Positional enrichment of consensus regions", fontsize=12, pad=10)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{x:.1f}" for x in centers], fontsize=9)
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
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
