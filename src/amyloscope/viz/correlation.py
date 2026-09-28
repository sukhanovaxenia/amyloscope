"""Consensus versus an external per-protein measurement.

The panel-level figures answer "where are the aggregation-prone regions"; this
one answers "does the consensus track anything measured in the cell". It takes a
mapping of protein id to a single experimental value — a FRAP mobile fraction
here, but nothing in the module is specific to that — and plots it against the
consensus summary, with the rank correlation annotated.

Two design decisions carry the module.

**The summary is not unique, and the choice changes the answer.** A consensus
run can be reduced to a per-protein number in several defensible ways: the count
of called regions, the fraction of residues reaching the moderate tier, or the
fraction sitting in the sub-threshold band below it. These need not agree, and in
this dataset they do not — the region count and the sub-threshold band correlate
with mobility in opposite directions. Reporting only the summary that happens to
correlate would be selection on the outcome, so :func:`plot_measurement_correlation`
plots the chosen summary and, when the loaded tracks are supplied, shows the
correlation obtained from each alternative alongside it.

**The p-value must be exact.** With four or five proteins the asymptotic
Spearman p is not attainable: four proteins admit 24 orderings, so the smallest
two-sided p any arrangement can produce is 2/24 = 0.083, and a reported 0.05 is
an artefact of the normal approximation rather than a property of the data. The
module therefore enumerates the permutation null when the factorial is small
enough and samples it otherwise, and prints the floor next to the p so the
reader can see what the sample size permits. This is the same reasoning that
governs the Mann-Whitney floor in the FRAP statistics.

The figure is descriptive at these sample sizes and is labelled as such; it
supports a statement about ranking, not a hypothesis test.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Mapping, Optional, Sequence

import numpy as np

from ..core.consensus import ConsensusResult
from . import labels as _labels
from . import style as _style

#: Summaries the consensus can be reduced to, weakest-assumption first. The key
#: is the label-table suffix; the value is a short human-readable fallback.
SUMMARIES = ("region_count", "moderate_band", "weak_band")




def _wrap(text: str, width: int = 20) -> str:
    """Soft-wrap a category label so a long name does not run into the plot."""
    import textwrap

    return "\n".join(textwrap.wrap(text, width=width)) or text

#: Exhaustive enumeration is used up to this many permutations; beyond it the
#: null is sampled. 8! = 40320 sits comfortably inside a second.
_EXACT_LIMIT = 50_000


def _length(result: ConsensusResult, protein_id: str) -> int:
    spec = result.config.protein(protein_id)
    if spec.length:
        return spec.length
    regs = result.regions.get(protein_id, [])
    return max((r.end for r in regs), default=100)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho via Pearson on ranks, tie-aware."""
    from scipy import stats as sc_stats

    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    rx = sc_stats.rankdata(x)
    ry = sc_stats.rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def exact_spearman_p(x: Sequence[float], y: Sequence[float],
                     rng_seed: int = 0) -> tuple[float, float, float, bool]:
    """``(rho, p, p_floor, exact)`` from the permutation null.

    ``p_floor`` is the smallest two-sided p **this** design can return, taken
    from the null itself rather than from n!. The distinction is not academic.
    An untied n = 4 admits 24 orderings and a floor of about 2/24, but ties in
    the predictor summary collapse the null: with two proteins carrying zero
    consensus regions the ranks are [4, 3, 1.5, 1.5], several permutations
    reach the same extreme rho, and the floor rises to 0.20. A correlation of
    rho = -0.95 is then the strongest result the design can produce and still
    cannot fall below p = 0.20 — which is the honest reading, and the opposite
    of the p = 0.05 the normal approximation returns for the same numbers.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    rho = _spearman(x, y)
    if not np.isfinite(rho) or n < 3:
        return rho, float("nan"), float("nan"), False

    total = math.factorial(n)
    if total <= _EXACT_LIMIT:
        null = np.array([_spearman(x, [y[i] for i in perm])
                         for perm in permutations(range(n))])
        exact = True
    else:
        rng = np.random.default_rng(rng_seed)
        draws = 100_000
        null = np.array([_spearman(x, rng.permutation(y)) for _ in range(draws)])
        exact = False

    # +1 correction keeps the estimate conservative and non-zero.
    def _p(threshold: float) -> float:
        return float((np.sum(np.abs(null) >= threshold - 1e-12) + 1)
                     / (null.size + 1))

    p = _p(abs(rho))
    # The floor is the p attached to the most extreme rho the null contains,
    # computed the same way, so it is directly comparable with p above.
    floor = _p(float(np.nanmax(np.abs(null))))
    return rho, min(1.0, p), min(1.0, floor), exact


def _summaries_for(result: ConsensusResult, dataset, config,
                   protein_ids: Sequence[str]) -> dict[str, dict[str, float]]:
    """Per-protein value of every available consensus summary.

    ``region_count`` needs only the consensus result. The two residue-level
    bands need the loaded predictor tracks, because they are defined on the
    position-wise agreement profile rather than on the called regions; without a
    dataset they are omitted rather than approximated from region extents, which
    would silently conflate "residues reaching the tier" with "residues inside a
    region that reached it".
    """
    out: dict[str, dict[str, float]] = {
        "region_count": {pid: float(len(result.regions.get(pid, [])))
                         for pid in protein_ids}
    }
    if dataset is None:
        return out

    from .detailed import (_coverage_matrix, _panel_and_thresholds,
                           _per_tool_regions, _tool_order)

    moderate, weak = {}, {}
    for pt in dataset:
        pid = pt.spec.id
        if pid not in protein_ids or not pt.tracks:
            continue
        length = pt.length or _length(result, pid)
        per_tool = _per_tool_regions(pt, config)
        _, ks, _tiers = _panel_and_thresholds(config, per_tool)
        counts = _coverage_matrix(per_tool, _tool_order(config), length).sum(axis=0)
        moderate_k = ks[-1]
        moderate[pid] = 100.0 * float((counts >= moderate_k).sum()) / length
        weak[pid] = 100.0 * float(
            ((counts >= 2) & (counts < moderate_k)).sum()) / length
    if len(moderate) == len(protein_ids):
        out["moderate_band"] = moderate
        out["weak_band"] = weak
    return out


def plot_measurement_correlation(
    result: ConsensusResult,
    measurements: Mapping[str, float],
    dataset=None,
    config=None,
    summary: str = "region_count",
    measurement_label: Optional[str] = None,
    figsize=None,
):
    """Scatter a consensus summary against an external per-protein measurement.

    ``measurements`` maps protein id to one number (mobile fraction, ThT
    endpoint, aggregation index). Proteins absent from either side are dropped.
    Supplying ``dataset`` adds the comparison of alternative summaries.
    """
    import matplotlib.pyplot as plt

    config = config or result.config
    viz = config.viz
    _style.apply_style(viz)
    tr = _labels.translator(viz.language)

    ids = [p.id for p in config.proteins if p.id in measurements]
    labels = {p.id: p.label for p in config.proteins}
    fs = getattr(viz, "resolved_figure_scale", 1.0)

    if len(ids) < 3:
        fonts = _style.figure_fonts(viz, 8 * fs)
        fig, ax = plt.subplots(figsize=(8 * fs, 3 * fs), facecolor="white")
        ax.text(0.5, 0.5, tr("corr_too_few"), ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    all_summaries = _summaries_for(result, dataset, config, ids)
    if summary not in all_summaries:
        summary = "region_count"
    y = [float(measurements[i]) for i in ids]
    x = [all_summaries[summary][i] for i in ids]
    rho, p, floor, exact = exact_spearman_p(x, y)

    two = len(all_summaries) > 1

    # Three columns: scatter, bar comparison, and a dedicated statistics panel
    # with its axes switched off. The statistics used to sit inside the scatter
    # at transAxes (0.03, 0.97), where it covered the topmost point — with four
    # proteins one of them is always at the y-limit. Giving the text its own
    # column is the same pattern the per-predictor track figure uses for its
    # summary boxes, so the reader meets one convention rather than two.
    widths = [1.30, 1.05, 0.62] if two else [1.30, 0.62]
    figsize = figsize or (sum(widths) * 4.6 * fs, 5.0 * fs)
    # Sizes come from the shared width-relative scale, resolved against this
    # figure's actual width, so the type matches every other module once both
    # are placed at a common column width.
    fonts = _style.figure_fonts(viz, figsize[0])
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1]) if two else None
    ax_s = fig.add_subplot(gs[0, -1])
    ax_s.axis("off")

    # ---- scatter ---------------------------------------------------------- #
    cmap = _style.get_cmap("YlOrRd")
    order = np.argsort(x)
    shade = {i: cmap(0.35 + 0.5 * k / max(1, len(ids) - 1))
             for k, i in enumerate(order)}
    for k, pid in enumerate(ids):
        ax.scatter(x[k], y[k], s=150 * fs, color=shade[k], edgecolor="#333333",
                   linewidth=0.8, zorder=3)

    # Margins first, then label offsets chosen against the resulting limits: a
    # point on an axis limit gets its name pushed inward, so no label is clipped
    # regardless of where the extremes fall.
    ax.margins(x=0.16, y=0.16)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for k, pid in enumerate(ids):
        dx = -10 if x[k] > x0 + 0.72 * (x1 - x0) else 10
        dy = -14 if y[k] > y0 + 0.78 * (y1 - y0) else 8
        ax.annotate(labels.get(pid, pid), (x[k], y[k]),
                    xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left",
                    va="top" if dy < 0 else "bottom",
                    fontsize=fonts["point"], fontweight="bold")

    ax.set_xlabel(tr(f"corr_x_{summary}"), fontsize=fonts["label"])
    ax.set_ylabel(measurement_label or tr("corr_y_default"),
                  fontsize=fonts["label"])
    ax.set_title(tr("corr_title"), pad=10, #fontweight="bold",
                 fontsize=fonts["title"])
    ax.tick_params(labelsize=fonts["tick"])
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- alternative summaries -------------------------------------------- #
    if ax_b is not None:
        names, rhos, ps = [], [], []
        for key in SUMMARIES:
            if key not in all_summaries:
                continue
            r_k, p_k, _f, _e = exact_spearman_p(
                [all_summaries[key][i] for i in ids], y)
            names.append(_wrap(tr(f"corr_x_{key}")))
            rhos.append(0.0 if not np.isfinite(r_k) else r_k)
            ps.append(p_k)
        pos = np.arange(len(names))
        colours = ["#2E6F95" if r < 0 else "#B23A48" for r in rhos]
        ax_b.barh(pos, rhos, color=colours, edgecolor="#333333", linewidth=0.6,
                  alpha=0.9, height=0.55)
        ax_b.axvline(0, color="#333333", linewidth=0.9)
        ax_b.set_xlim(-1.15, 1.62)

        # Values right-aligned in a fixed column at the right edge. Anchoring
        # them to the bar tip meant a negative rho drew its text leftward, into
        # the category label on the same row — the collision in the previous
        # version. A fixed column also lets the numbers be read down the axis.
        for k, (r_k, p_k) in enumerate(zip(rhos, ps)):
            ax_b.text(1.56, k, f"{r_k:+.2f}   $p$ = {p_k:.3f}",
                      va="center", ha="right", fontsize=fonts["annot"],
                      color="#333333")
        ax_b.set_yticks(pos)
        ax_b.set_yticklabels(names, fontsize=fonts["tick"])
        ax_b.invert_yaxis()          # strongest summary on top, as listed
        ax_b.set_xlabel(tr("corr_rho_axis"), fontsize=fonts["label"])
        ax_b.set_title(tr("corr_alt_title"), pad=10, fontsize=fonts["label"])
        ax_b.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax_b.tick_params(axis="x", labelsize=fonts["tick"])
        ax_b.grid(True, axis="x", alpha=0.2, linestyle=":", linewidth=0.5)
        ax_b.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax_b.spines[sp].set_visible(False)

    # ---- statistics panel -------------------------------------------------- #
    note = "\n".join([
        tr("corr_stat", rho=f"{rho:+.2f}", p=f"{p:.3f}", n=len(ids)),
        "",
        tr("corr_floor" if exact else "corr_floor_mc", floor=f"{floor:.3f}"),
        "",
        tr("corr_descriptive"),
    ])
    ax_s.text(0.0, 0.98, note, transform=ax_s.transAxes, va="top", ha="left",
              fontsize=fonts["annot"], linespacing=1.35, wrap=True,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1",
                        edgecolor="#B08D2E", linewidth=0.9))

    # Explicit margins rather than tight_layout: the statistics column is an
    # axes with its frame switched off, which tight_layout cannot size from, and
    # it emits a compatibility warning while leaving the spacing wrong. Reserving
    # the margins directly is what the domain figure does for the same reason.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.16)
    return fig