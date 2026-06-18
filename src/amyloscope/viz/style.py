"""Shared figure styling and colour resolution.

Centralises what the original modules repeated across files: the publication
rcParams block, the consensus-tier colour lookup, and the mapping from a
domain's abstract category to a fill colour.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from ..config import ConsensusConfig, VizConfig


def apply_style(viz: VizConfig) -> None:
    """Apply the package's publication rcParams."""
    sns.set_style("white")
    plt.rcParams.update(
        {
            "font.family": viz.font_family,
            "font.size": 9,
            "figure.dpi": 100,
            "savefig.dpi": viz.dpi,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def tier_color(
    tier_name: str, consensus: ConsensusConfig, default: str = "#999999"
) -> str:
    for tier in consensus.tiers:
        if tier.name == tier_name:
            return tier.color
    return default


def tier_order(consensus: ConsensusConfig) -> list[str]:
    """Tier names, weakest-first (suits stacked-track vertical ordering)."""
    return [t.name for t in sorted(consensus.tiers, key=lambda t: t.min_fraction)]


def domain_color(category: str, viz: VizConfig) -> str:
    """Resolve a domain category to a colour by case-insensitive substring.

    The first palette key that appears as a substring of ``category`` wins,
    so ``rna_binding_dsRBD`` and ``rna_binding`` both resolve to the same hue.
    """
    cat = (category or "default").lower()
    for key, colour in viz.domain_palette.items():
        if key.lower() in cat:
            return colour
    return viz.domain_palette.get("default", "#CCCCCC")


def save_figure(fig, path, viz: VizConfig) -> list[str]:
    """Save a figure as PNG (and SVG when configured); return written paths."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    fig.savefig(path, dpi=viz.dpi, bbox_inches="tight", facecolor="white")
    written.append(str(path))
    if viz.save_svg:
        svg = path.with_suffix(".svg")
        fig.savefig(svg, bbox_inches="tight", facecolor="white")
        written.append(str(svg))
    return written
