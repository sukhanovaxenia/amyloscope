"""Per-predictor score-track figure for a single protein.

Stacks each predictor's per-residue profile with its called APR intervals
shaded, plus a compact per-tool summary panel. Generalised from the original
``create_tool_predictions_figure``: the threshold guide line is drawn only for
predictors whose detection is an actual scalar cutoff (``above``/``below``),
since flag/presence-based calls have no scalar to plot.
"""

from __future__ import annotations

from matplotlib.gridspec import GridSpec

from ..core.regions import call_regions
from ..io.loader import ProteinTracks
from . import style as _style


def plot_tool_tracks(protein_tracks: ProteinTracks, config, figsize=None):
    import matplotlib.pyplot as plt

    _style.apply_style(config.viz)
    tools = [t for t in config.enabled_tools if t.name in protein_tracks.tracks]
    n = len(tools)
    if n == 0:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, "No predictor tracks loaded", ha="center", va="center")
        ax.axis("off")
        return fig

    figsize = figsize or (14, 1.4 * n)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n, 2, width_ratios=[5, 1], hspace=0.25, wspace=0.1)

    max_pos = protein_tracks.length or max(
        int(df["Number"].max()) for df in protein_tracks.tracks.values()
    )

    for i, tool in enumerate(tools):
        ax = fig.add_subplot(gs[i, 0])
        df = protein_tracks.tracks[tool.name]
        ax.plot(df["Number"], df["Score"], color=tool.color, linewidth=1.2, alpha=0.9)
        ax.fill_between(df["Number"], df["Score"], alpha=0.25, color=tool.color)

        det = tool.detection
        if det.method in {"above", "below"} and det.threshold is not None:
            ax.axhline(det.threshold, color="#CC0000", linestyle="--", alpha=0.6,
                       linewidth=1, label=f"threshold {det.threshold:g}")
        if det.method == "below":
            ax.invert_yaxis()  # lower (more negative) energy => higher propensity

        regions = call_regions(df, det, min_length=config.consensus.min_region_length)
        for start, end in regions:
            ax.axvspan(start, end, alpha=0.15, color="#CC0000", zorder=-1)

        ax.set_ylabel(f"{tool.name}", fontsize=9)
        ax.set_xlim(0, max_pos)
        ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=8)
        if i < n - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Amino acid position", fontsize=10)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.9)

        # Summary panel.
        ax_s = fig.add_subplot(gs[i, 1])
        ax_s.axis("off")
        total_aa = sum(e - s + 1 for s, e in regions)
        avg = total_aa / len(regions) if regions else 0
        bg = "#FFE5E5" if len(regions) > 3 else "#FFF5E5" if regions else "#F5F5F5"
        ax_s.text(
            0.1, 0.5,
            f"{tool.name}\n{'-' * 12}\nRegions: {len(regions)}\n"
            f"Total aa: {total_aa}\nAvg len: {avg:.1f}",
            transform=ax_s.transAxes, fontsize=8, va="center", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, alpha=0.7),
        )

    fig.suptitle(
        f"Per-predictor amyloidogenic profiles — {protein_tracks.spec.label}",
        fontsize=12, fontweight="bold", y=0.92,
    )
    return fig
