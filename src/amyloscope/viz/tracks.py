"""Per-predictor score-track figure for a single protein.

Stacks each predictor's per-residue profile with its called APR intervals
shaded, plus a compact per-tool summary panel. The threshold guide line is
drawn only for predictors whose detection is an actual scalar cutoff
(``above``/``below``), since flag/presence-based calls have no scalar to plot.

Predictor colours are resolved through :func:`amyloscope.viz.style.tool_color`.
Reading ``tool.color`` directly gave every predictor without an explicit colour
in the config the same default grey, so ArchCandy, CrossBeta and TANGO were
indistinguishable here while being separately coloured in the detailed figures.
"""

from __future__ import annotations

import numpy as np
from matplotlib.gridspec import GridSpec

from ..core.regions import call_regions
from ..io.loader import ProteinTracks
from . import labels as _labels
from . import style as _style


def plot_tool_tracks(protein_tracks: ProteinTracks, config, figsize=None):
    import matplotlib.pyplot as plt

    _style.apply_style(config.viz)
    tr = _labels.translator(config.viz.language)
    # Enumerate over the full enabled panel so a predictor keeps the same
    # palette index (and therefore the same colour) whether or not every tool
    # produced a track for this protein.
    indexed = [
        (i, t) for i, t in enumerate(config.enabled_tools)
        if t.name in protein_tracks.tracks
    ]
    n = len(indexed)
    if n == 0:
        fig, ax = plt.subplots(figsize=config.viz.figsize(10, 2))
        ax.text(0.5, 0.5, tr("no_tracks"), ha="center", va="center")
        ax.axis("off")
        return fig

    figsize = figsize or config.viz.figsize(14, 1.4 * n)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n, 2, width_ratios=[5, 1], hspace=0.25, wspace=0.1)

    max_pos = protein_tracks.length or max(
        int(df["Number"].max()) for df in protein_tracks.tracks.values()
    )

    min_len = (
        getattr(config.consensus, "min_tool_region_length", None)
        or config.consensus.min_region_length
    )

    for row, (palette_idx, tool) in enumerate(indexed):
        ax = fig.add_subplot(gs[row, 0])
        df = protein_tracks.tracks[tool.name]
        colour = _style.tool_color(tool, palette_idx)
        y = df["Score"].to_numpy(dtype=float)
        ax.plot(df["Number"], y, color=colour, linewidth=1.2, alpha=0.9)
        # Fill to the track minimum rather than to zero: for a free-energy
        # profile that never crosses zero, filling to zero renders the whole
        # panel as a solid block and hides the structure.
        ax.fill_between(df["Number"], y, np.nanmin(y), alpha=0.25, color=colour)

        det = tool.detection
        if det.method in {"above", "below"} and det.threshold is not None:
            ax.axhline(det.threshold, color="#CC0000", linestyle="--", alpha=0.6,
                       linewidth=1, label=tr("threshold", value=f"{det.threshold:g}"))

        regions = call_regions(df, det, min_length=min_len)
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.15, color="#CC0000", zorder=-1)

        if det.method == "below":
            ax.invert_yaxis()  # lower (more negative) energy => higher propensity

        ax.set_ylabel(f"{tool.name}", fontsize="medium")
        ax.set_xlim(0, max_pos)
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
        ax.tick_params(axis="both", labelsize="small")
        if row < n - 1:
            # tick_params rather than set_xticklabels([]): the latter pins the
            # labels to the current locator, so any later autoscale silently
            # mislabels the shared axis.
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(tr("axis_position"))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize="x-small", frameon=True,
                      framealpha=0.9)

        # Summary panel.
        ax_s = fig.add_subplot(gs[row, 1])
        ax_s.axis("off")
        total_aa = sum(e - s + 1 for s, e in regions)
        avg = total_aa / len(regions) if regions else 0
        bg = "#FFE5E5" if len(regions) > 3 else "#FFF5E5" if regions else "#F5F5F5"
        summary = (
            f"{tool.name}\n{'-' * 12}\n"
            + tr("summary_regions", n=len(regions)) + "\n"
            + tr("summary_total_aa", n=total_aa) + "\n"
            + tr("summary_avg_len", value=f"{avg:.1f}")
        )
        ax_s.text(
            0.1, 0.5, summary,
            transform=ax_s.transAxes, fontsize="small", va="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.7),
        )

    fig.suptitle(
        tr("title_tracks", label=protein_tracks.spec.label),
        fontweight="bold", y=0.92,
    )
    return fig