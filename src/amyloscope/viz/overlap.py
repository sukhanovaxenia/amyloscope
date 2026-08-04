"""Domain-overlap figure.

The companion to :mod:`amyloscope.analysis.domains`.

Encoding
--------
Each category is drawn as the null distribution of APR residues — a box for the
interquartile range, whiskers for the central 95 % — with the observed count as
a filled marker on the same axis. An observation sitting inside its own box is
unremarkable; one outside the whisker is not. This replaces an earlier version
that drew observed counts as bars with the expected value marked by a vertical
rule on top of each bar, where the rule read as a tick or a rendering artefact
rather than as a reference value, and where the bar length encoded a quantity
(absolute residues) that is not the one being tested.

Putting the null on the axis rather than annotating it also removes the need to
explain it in the figure: the reference is drawn, so the caption can carry the
method and the panel carries only data.

Two modes
---------
``style="publication"`` renders the panels alone, with significance markers and
a two-entry legend, on the assumption that n, the null construction and the
p-values live in the caption where a journal expects them.
``style="diagnostic"`` adds the statistics column — the same content, in the
figure, for reading during analysis rather than for print.
"""

from __future__ import annotations

import numpy as np

from ..analysis.domains import DomainOverlapResult
from . import labels as _labels
from . import style as _style

#: Null summary drawn per category: box edges, whisker ends.
_BOX_Q = (25.0, 75.0)
_WHISKER_Q = (2.5, 97.5)


def _fonts(viz) -> dict[str, float]:
    base = getattr(viz, "resolved_base_font", 11.0)
    return {"title": base * 1.00, "label": base * 0.90, "tick": base * 0.78,
            "annot": base * 0.72, "star": base * 0.95}


def _hist_percentiles(hist, quantiles) -> list[float]:
    """Percentiles of a distribution stored as a count histogram.

    The null is kept as a bincount indexed by residue count, so the quantiles
    come from its cumulative distribution rather than from raw draws. The
    statistic is integer-valued, so nothing is lost by this representation.
    """
    counts = np.asarray(hist, float)
    total = counts.sum()
    if total <= 0:
        return [float("nan")] * len(quantiles)
    cdf = np.cumsum(counts) / total
    return [float(np.searchsorted(cdf, q / 100.0)) for q in quantiles]


def _stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def plot_domain_overlap(result: DomainOverlapResult, style: str = "publication",
                        figsize=None):
    """APR residues per domain category against the relocation null.

    Takes the object :func:`amyloscope.analysis.domains.compute_domain_overlap`
    returns, not the ConsensusResult the other figure builders take. That
    asymmetry is deliberate — the enrichment null is expensive and belongs to
    the analysis layer, not to a plotting call — but it is easy to get wrong, so
    the wrong type is rejected with a sentence rather than an AttributeError on
    whichever field happens to be read first.
    """
    import matplotlib.pyplot as plt

    if not hasattr(result, "enrichment"):
        raise TypeError(
            "plot_domain_overlap expects the DomainOverlapResult returned by "
            "amyloscope.analysis.domains.compute_domain_overlap(), not "
            f"{type(result).__name__}. In the pipeline this object is built in "
            "run() and must be passed through to _render_figures()."
        )
    if style not in {"publication", "diagnostic"}:
        raise ValueError("style must be 'publication' or 'diagnostic'")

    config = result.config
    viz = config.viz
    _style.apply_style(viz)
    tr = _labels.translator(viz.language)
    fonts = _fonts(viz)
    fs = getattr(viz, "resolved_figure_scale", 1.0)

    rows = list(result.enrichment)
    if not rows:
        fig, ax = plt.subplots(figsize=(8 * fs, 3 * fs), facecolor="white")
        ax.text(0.5, 0.5, tr("ov_no_regions"), ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    primary = result.primary_category
    null_hist = getattr(result, "null_hist", {}) or {}
    has_null = primary in null_hist
    diagnostic = style == "diagnostic"

    widths = [1.30]
    if has_null:
        widths.append(1.00)
    if diagnostic:
        widths.append(0.62)
    figsize = figsize or (sum(widths) * 4.6 * fs, 4.8 * fs)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    ax_n = fig.add_subplot(gs[0, 1]) if has_null else None
    ax_s = fig.add_subplot(gs[0, -1]) if diagnostic else None
    if ax_s is not None:
        ax_s.axis("off")

    # ---- null intervals with the observed value ---------------------------- #
    pos = np.arange(len(rows))
    for k, row in enumerate(rows):
        hist = null_hist.get(row.category)
        if hist is not None:
            q1, q3 = _hist_percentiles(hist, _BOX_Q)
            lo, hi = _hist_percentiles(hist, _WHISKER_Q)
            ax.plot([lo, hi], [k, k], color="#8A99A3", linewidth=1.1,
                    solid_capstyle="butt", zorder=2)
            for cap in (lo, hi):
                ax.plot([cap, cap], [k - 0.13, k + 0.13], color="#8A99A3",
                        linewidth=1.1, zorder=2)
            ax.add_patch(plt.Rectangle((q1, k - 0.20), max(q3 - q1, 0.001), 0.40,
                                       facecolor="#DCE3E8", edgecolor="#8A99A3",
                                       linewidth=0.8, zorder=3))
        colour = _style.domain_color(row.category, viz)
        ax.scatter(row.observed_aa, k, s=(230 if row.category == primary else 150) * fs,
                   color=colour, edgecolor="#1F2933",
                   linewidth=1.4 if row.category == primary else 0.9, zorder=5)
        p = row.p_greater if row.category == primary else row.p_holm
        star = _stars(p)
        if star:
            ax.text(row.observed_aa, k - 0.30, star, ha="center", va="bottom",
                    fontsize=fonts["star"], fontweight="bold", color="#1F2933")
        ax.text(row.observed_aa, k + 0.34, f"{row.fold:.2f}\u00d7", ha="center",
                va="top", fontsize=fonts["annot"],
                fontweight="bold" if row.category == primary else "normal",
                color="#1F2933")

    ax.set_yticks(pos)
    ax.set_yticklabels(
        [tr("ov_cat", cat=r.category, cov=f"{100 * r.coverage:.0f}") for r in rows],
        fontsize=fonts["tick"])
    ax.set_ylim(len(rows) - 0.5, -0.6)
    ax.set_xlabel(tr("ov_x_residues"), fontsize=fonts["label"])
    ax.set_title(tr("ov_title"), pad=10, fontsize=fonts["title"])
    ax.tick_params(axis="x", labelsize=fonts["tick"])
    ax.set_xlim(left=-max(r.observed_aa for r in rows) * 0.06)
    ax.margins(x=0.14)
    ax.grid(True, axis="x", alpha=0.18, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # Two entries only. The null is drawn, so it needs naming, not explaining.
    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9 * fs ** 0.5,
                   markerfacecolor="#7F8C99", markeredgecolor="#1F2933",
                   label=tr("ov_legend_observed")),
        plt.Rectangle((0, 0), 1, 1, facecolor="#DCE3E8", edgecolor="#8A99A3",
                      label=tr("ov_legend_null")),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=2, frameon=False, fontsize=fonts["annot"],
              handletextpad=0.6, columnspacing=1.6)

    # ---- null distribution for the pre-specified category ------------------ #
    if ax_n is not None:
        row = next(r for r in rows if r.category == primary)
        counts = np.asarray(null_hist[primary], float)
        ax_n.bar(np.arange(counts.size), counts, width=1.0, color="#DCE3E8",
                 edgecolor="#B8C4CC", linewidth=0.3, zorder=2)
        ax_n.axvline(row.observed_aa,
                     color=_style.domain_color(primary, viz), linewidth=2.4,
                     zorder=4)
        ax_n.annotate(tr("ov_legend_observed"),
                      xy=(row.observed_aa, counts.max() * 0.96),
                      xytext=(-8, 0), textcoords="offset points",
                      ha="right", va="top", fontsize=fonts["annot"],
                      color="#1F2933")
        ax_n.set_xlabel(tr("ov_x_null", cat=primary), fontsize=fonts["label"])
        ax_n.set_ylabel(tr("ov_y_null"), fontsize=fonts["label"])
        ax_n.set_title(tr("ov_null_title"), pad=10, fontsize=fonts["label"])
        ax_n.tick_params(labelsize=fonts["tick"])
        ax_n.grid(True, axis="y", alpha=0.15, linestyle=":", linewidth=0.5)
        ax_n.set_axisbelow(True)
        for sp in ("top", "right"):
            ax_n.spines[sp].set_visible(False)

    # ---- statistics column, diagnostic mode only --------------------------- #
    if ax_s is not None:
        note = [tr("ov_stat_head", n=result.total_apr_aa,
                   inside=result.total_in_domain_aa,
                   pct=f"{result.global_domain_fraction * 100:.0f}")]
        if primary:
            row = next(r for r in rows if r.category == primary)
            note += ["", tr("ov_stat_primary", cat=primary,
                            fold=f"{row.fold:.2f}", p=f"{row.p_greater:.4f}")]
        other = [r for r in rows if r.category != primary]
        if other:
            note += ["", tr("ov_stat_explore", n=len(other))]
        note += ["", tr("ov_stat_null", draws=f"{result.draws:,}")]
        ax_s.text(0.0, 0.98, "\n".join(note), transform=ax_s.transAxes, va="top",
                  ha="left", fontsize=fonts["annot"], linespacing=1.35,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#F4F7F9",
                            edgecolor="#9AA7B0", linewidth=0.9))

    fig.subplots_adjust(left=0.17, right=0.985, top=0.86,
                        bottom=0.24 if not diagnostic else 0.22)
    return fig


def caption(result: DomainOverlapResult, language: str = "en") -> str:
    """Figure caption carrying what the publication panels deliberately omit.

    The panels show data; n, the null construction and the p-values belong in
    the caption, and generating it here keeps the numbers in the text identical
    to the ones the figure was drawn from.
    """
    tr = _labels.translator(language)
    primary = result.primary_category
    parts = [tr("ov_cap_head", n=result.total_apr_aa,
                np=len([o for o in result.per_protein.values()
                        if o.total_apr_aa]),
                draws=f"{result.draws:,}")]
    if primary:
        row = next((r for r in result.enrichment if r.category == primary), None)
        if row is not None:
            parts.append(tr("ov_cap_primary", cat=primary,
                            obs=row.observed_aa, exp=f"{row.expected_aa:.1f}",
                            fold=f"{row.fold:.2f}", p=f"{row.p_greater:.3f}"))
    other = [r for r in result.enrichment if r.category != primary]
    if other:
        parts.append(tr("ov_cap_explore", cats=", ".join(r.category for r in other)))
    return " ".join(parts)