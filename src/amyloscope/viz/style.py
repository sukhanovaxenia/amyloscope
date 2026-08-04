"""Shared figure styling and colour resolution.

Centralises what the original modules repeated across files: the publication
rcParams block, the consensus-tier colour lookup, the mapping from a domain's
abstract category to a fill colour, and — new here — the per-tool colour
fallback and the colormap accessor, both of which had diverged between the
``tracks`` and ``detailed`` figure families.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from ..config import ConsensusConfig, VizConfig

#: Distinct fallback palette (Dark2/Set2 order) for tools that do not carry a
#: meaningful colour in the config (default "#333333"). Shared so that a given
#: predictor is the same colour in every figure of the paper.
TOOL_PALETTE = [
    "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E",
    "#E6AB02", "#A6761D", "#666666", "#1F78B4", "#B15928",
]


def get_cmap(name: str):
    """Colormap accessor valid on both old and new Matplotlib.

    ``matplotlib.cm.get_cmap`` was deprecated in 3.7 and removed in 3.9, so a
    figure script that calls it fails on any current environment — exactly the
    kind of silent reproducibility break that matters for an archived release.
    """
    try:
        import matplotlib as mpl

        return mpl.colormaps[name]
    except (AttributeError, KeyError):  # Matplotlib < 3.5
        import matplotlib.cm as cm

        return cm.get_cmap(name)


def tool_color(tool, idx: int) -> str:
    """Resolve a predictor's colour, falling back to the shared palette."""
    colour = getattr(tool, "color", None)
    if colour and colour != "#333333":
        return colour
    return TOOL_PALETTE[idx % len(TOOL_PALETTE)]


def apply_style(viz: VizConfig) -> None:
    """Apply the package's publication rcParams, scaled by the viz preset."""
    base = viz.resolved_base_font
    s = viz.resolved_scale
    sns.set_style("white")
    plt.rcParams.update(
        {
            # Keep the requested family first, but always fall back to a
            # Cyrillic-complete font so Russian labels render on any system.
            "font.family": "sans-serif",
            "font.sans-serif": [
                viz.font_family, "DejaVu Sans", "Liberation Sans", "Arial",
            ],
            "axes.unicode_minus": False,
            "font.size": base,
            "axes.titlesize": base * 1.33,
            "axes.labelsize": base * 1.22,
            "xtick.labelsize": base * 1.0,
            "ytick.labelsize": base * 1.0,
            "legend.fontsize": base * 1.0,
            "legend.title_fontsize": base * 1.22,
            "figure.titlesize": base * 1.33,
            "figure.dpi": 100,
            "savefig.dpi": viz.dpi,
            "axes.linewidth": 0.8 * s,
            "lines.linewidth": 1.2 * s,
            "patch.linewidth": 0.5 * s,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "text.color": "#333333",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "xtick.major.size": 3 * s,
            "ytick.major.size": 3 * s,
            "xtick.major.width": 0.8 * s,
            "ytick.major.width": 0.8 * s,
        }
    )


def stagger_label_rows(
    centers: list[float],
    axis_range: float,
    axis_width_in: float,
    font_pt: float,
    n_chars: int = 7,
    pad: float = 1.3,
) -> list[int]:
    """Assign each label a row so that horizontally-close labels stack vertically.

    Region range labels (e.g. ``163-171``) collide when adjacent regions sit a
    few residues apart. Estimating each label's width in data units from the
    font size and axis geometry, this greedily places every label on the lowest
    row whose previous label is at least one label-width away, returning a row
    index per input label (0 = closest to the track). Labels far enough apart
    all stay on row 0, so sparse panels are unchanged.
    """
    if not centers:
        return []
    label_w_in = n_chars * 0.6 * font_pt / 72.0
    min_gap = label_w_in / max(axis_width_in, 1e-6) * axis_range * pad
    order = sorted(range(len(centers)), key=lambda i: centers[i])
    rows = [0] * len(centers)
    last_x_in_row: list[float] = []
    for i in order:
        c = centers[i]
        for r, last in enumerate(last_x_in_row):
            if c - last >= min_gap:
                rows[i] = r
                last_x_in_row[r] = c
                break
        else:
            rows[i] = len(last_x_in_row)
            last_x_in_row.append(c)
    return rows


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


def tier_thresholds(consensus: ConsensusConfig, panel_size: int) -> list[tuple[str, int]]:
    """``[(tier_name, min_count), ...]`` strongest-first for a given panel size.

    The single place tiers are resolved to counts. Every figure must go through
    this rather than indexing ``tiers_by_strength`` positionally, which is what
    caused the strong tier to be resolved from the unanimous fraction.
    """
    from ..core.consensus import resolve_count

    return [
        (t.name, resolve_count(t.min_fraction, panel_size))
        for t in consensus.tiers_by_strength
    ]


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