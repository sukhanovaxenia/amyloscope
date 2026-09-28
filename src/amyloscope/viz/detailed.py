"""Optional, detailed per-protein consensus figures.

These reproduce the information-dense diagnostic panels from the legacy analysis
scripts: a per-tool agreement heatmap, the position-wise tool-count profile with
the tier thresholds drawn in, the classified consensus-region track, and an
agreement-statistics summary. They are opt-in (``viz.detailed: true``) because
they are diagnostic rather than publication figures, and one is emitted per
protein.

Tier counts come from :func:`amyloscope.viz.style.tier_thresholds`, which
resolves every tier by name. Positional indexing into ``tiers_by_strength``
(``tiers[0]`` for "strong") silently resolved the strong tier from the
unanimous fraction as soon as a third tier was added, which made the strong
band structurally empty in the statistics panel.
"""

from __future__ import annotations

import numpy as np

from ..core.consensus import _per_tool_regions, resolve_count  # noqa: F401
from . import style as _style

# Self-contained label catalogue (does not depend on viz.labels).
_LABELS = {
    "en": {
        "title": "Consensus analysis of amyloidogenic predictions — {name}",
        "heatmap": "Tool agreement heatmap (dark = predicted amyloidogenic)",
        "position": "Position",
        "n_tools": "Number of tools",
        "profile": "Position-wise tool agreement (peak agreement per residue)",
        "regions": "Consensus amyloidogenic regions",
        "stats_title": "Agreement statistics",
        "unanimous": "Unanimous ({k} tools)",
        "strong": "Strong (\u2265{k} tools)",
        "moderate": "Moderate (\u2265{k} tools)",
        "weak": "Weak (2\u2013{k} tools)",
        "single": "Single tool",
        "none": "No prediction",
        "region_label": "{start}\u2013{end}\n({k} tools)",
        "title_analysis": "Amyloidogenic predictions — {name} ({n} tools)",
        "tool_stats": "Tool statistics",
        "stat_regions": "Regions: {n}",
        "stat_aa": "AA count: {n}",
        "no_data": "{tool}: no track",
        "cross_title": "Consensus amyloidogenic regions across proteins",
        "proteins": "Proteins",
        "consensus_legend": "Consensus (\u2265{k} tools)",
        "covering_note": "region labels = predictors covering \u2265{cov:.0%} of "
                         "the region; bars = peak per-residue agreement",
    },
    "ru": {
        "title": "Консенсусный анализ предсказаний амилоидогенности — {name}",
        "heatmap": "Карта согласия предсказателей (тёмный = амилоидогенный)",
        "position": "Позиция",
        "n_tools": "Число предсказателей",
        "profile": "Согласие предсказателей по позициям",
        "regions": "Консенсусные амилоидогенные участки",
        "stats_title": "Статистика согласия",
        "unanimous": "Единогласно ({k})",
        "strong": "Сильно (\u2265{k})",
        "moderate": "Средне (\u2265{k})",
        "weak": "Слабо (2\u2013{k})",
        "single": "Один предсказатель",
        "none": "Нет предсказаний",
        "region_label": "{start}\u2013{end}\n({k} предсказателей)",
        "title_analysis": "Предсказания амилоидогенности — {name} ({n} предсказателей)",
        "tool_stats": "Статистика предсказателей",
        "stat_regions": "Участки: {n}",
        "stat_aa": "Число ак: {n}",
        "no_data": "{tool}: нет трека",
        "cross_title": "Консенсусные амилоидогенные участки по белкам",
        "proteins": "Белки",
        "consensus_legend": "Консенсус (\u2265{k})",
        "covering_note": "подписи участков = предсказатели, покрывающие "
                         "\u2265{cov:.0%} участка; столбцы = пиковое согласие",
    },
}


def _tr(viz):
    lang = getattr(viz, "language", "en")
    table = _LABELS.get(lang, _LABELS["en"])
    english = _LABELS["en"]

    def tr(key, **kw):
        s = table.get(key, english.get(key, key))
        return s.format(**kw) if kw else s

    return tr


def _tool_order(config) -> list[str]:
    return [t.name for t in config.enabled_tools]


def _tool_colour(tool, idx):
    """Backwards-compatible alias for the shared resolver in :mod:`style`."""
    return _style.tool_color(tool, idx)


def _coverage_matrix(per_tool, tools, length):
    """Binary tools x length matrix; 1 where a tool calls an APR."""
    mat = np.zeros((len(tools), length), dtype=float)
    for i, name in enumerate(tools):
        for start, end in per_tool.get(name, []):
            lo, hi = max(1, start), min(length, end)
            if hi >= lo:
                mat[i, lo - 1 : hi] = 1.0
    return mat


def _panel_and_thresholds(config, per_tool):
    """``(panel_size, ks, tiers)`` where ``ks`` is strongest-first tier counts.

    ``per_tool`` may be empty only when ``denominator == "configured"``; under
    the "loaded" denominator an empty mapping previously collapsed the panel to
    a single predictor, so it is rejected explicitly rather than silently
    producing a "\u22651 tools" consensus.
    """
    tiers = config.consensus.tiers_by_strength
    if config.consensus.denominator == "configured":
        panel_size = len(config.enabled_tools)
    else:
        if not per_tool:
            raise ValueError(
                "denominator='loaded' requires the per-tool region mapping; "
                "pass the loaded tracks rather than an empty dict"
            )
        panel_size = max(1, len(per_tool))
    ks = [k for _, k in _style.tier_thresholds(config.consensus, panel_size)]
    return panel_size, ks, tiers


def _tier_ramp(tiers):
    """Sequential colour per tier by rank (weak = light, strong = dark).

    Agreement strength is ordinal, so it is encoded on a sequential (YlOrRd)
    ramp rather than the config's categorical tier colours; this keeps the
    "more tools = hotter" reading monotonic in luminance, which a diverging
    red/blue/orange palette does not.
    """
    n = len(tiers)
    # cmap = _style.get_cmap("YlOrRd") # "#FDAE61" "#2166AC" "#B2182B"
    cmap = ["#FDAE61", "#2166AC", "#B2182B"]
    out = {}
    # tiers_by_strength is strongest-first; reverse so the weakest maps to light
    for i, t in enumerate(reversed(tiers)):
        # frac = 0.35 + 0.55 * (i / max(1, n - 1))
        # out[t.name] = cmap(frac)
        out[t.name] = cmap[i]
    return out


def _bar_colours(counts, ks, tiers):
    """Colour each residue by the strongest tier its agreement reaches."""
    ramp = _tier_ramp(tiers)
    moderate_k = ks[-1]
    out = []
    for c in counts:
        for tier, k in zip(tiers, ks, strict=True):
            if c >= k:
                out.append(ramp[tier.name])
                break
        else:
            out.append("#C9CDD4" if c >= 1 else "#EEEEEE")
    return out


def _agreement_bands(counts, ks, tiers, tr):
    """``[(label, n_residues), ...]`` for the statistics panel.

    Bands are the half-open intervals between consecutive tier thresholds, so
    the scheme holds for any number of tiers and no band can be empty by
    construction.
    """
    lines = []
    upper = int(max(ks)) + 1
    for tier, k in zip(tiers, ks, strict=True):
        n = int(((counts >= k) & (counts < upper)).sum())
        lines.append((tr(tier.name, k=k), n))
        upper = k
    moderate_k = ks[-1]
    lines.append((tr("weak", k=max(2, moderate_k - 1)),
                  int(((counts >= 2) & (counts < moderate_k)).sum())))
    lines.append((tr("single"), int((counts == 1).sum())))
    lines.append((tr("none"), int((counts == 0).sum())))
    return lines


def plot_consensus_analysis(protein_tracks, result, config, figsize=None):
    """Detailed per-protein consensus panel (heatmap + profile + regions + stats)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec

    viz = config.viz
    _style.apply_style(viz)
    tr = _tr(viz)

    spec = protein_tracks.spec
    length = protein_tracks.length or 1
    tools = _tool_order(config)
    per_tool = _per_tool_regions(protein_tracks, config)
    panel_size, ks, tiers = _panel_and_thresholds(config, per_tool)
    moderate_k = ks[-1]

    # position-wise number of tools calling an APR
    mat = _coverage_matrix(per_tool, tools, length)
    counts = mat.sum(axis=0).astype(int)
    positions = np.arange(1, length + 1)

    fs = getattr(viz, "resolved_figure_scale", 1.0)
    figsize = figsize or (14 * fs, 9 * fs)
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = GridSpec(
        3, 2, figure=fig, height_ratios=[2.1, 1.1, 0.9],
        width_ratios=[5, 1.15], hspace=0.45, wspace=0.04,
    )
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_prof = fig.add_subplot(gs[1, 0], sharex=ax_heat)
    ax_reg = fig.add_subplot(gs[2, 0], sharex=ax_heat)

    # ---- 1. agreement heatmap -------------------------------------------- #
    cmap = ListedColormap(["#3B5B80", "#8B1A2B"])
    ax_heat.imshow(
        mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
        extent=[0.5, length + 0.5, len(tools) - 0.5, -0.5],
    )
    ax_heat.set_yticks(range(len(tools)))
    ax_heat.set_yticklabels(
        [f"{t} ({len(per_tool.get(t, []))})" for t in tools], fontsize="small"
    )
    ax_heat.set_title(tr("heatmap"), fontsize="medium")
    ax_heat.tick_params(axis="x", labelbottom=False)
    for sp in ("top", "right"):
        ax_heat.spines[sp].set_visible(False)

    # ---- 2. position-wise agreement profile ------------------------------ #
    ramp = _tier_ramp(tiers)
    ax_prof.bar(positions, counts, width=1.0,
                color=_bar_colours(counts, ks, tiers), linewidth=0)
    # One line per tier, drawn weakest-first so the legend reads bottom-up.
    styles = [":", "--", "-.", (0, (3, 1, 1, 1))]
    seen_k = set()
    for i, (tier, k) in enumerate(zip(reversed(tiers), reversed(ks), strict=True)):
        if k in seen_k:      # identical counts would overplot silently
            continue
        seen_k.add(k)
        ax_prof.axhline(k, color="#B23A48", linestyle=styles[i % len(styles)],
                        linewidth=1, alpha=0.8, label=tr(tier.name, k=k))
    ax_prof.set_ylabel(tr("n_tools"), fontsize="small")
    ax_prof.set_ylim(0, panel_size + 0.5)
    ax_prof.set_title(tr("profile"), fontsize="medium")
    ax_prof.legend(loc="upper right", fontsize="x-small", ncol=3, framealpha=0.9)
    ax_prof.tick_params(axis="x", labelbottom=False)
    for sp in ("top", "right"):
        ax_prof.spines[sp].set_visible(False)

    # ---- 3. classified consensus-region track ---------------------------- #
    regs = result.regions.get(spec.id, [])
    tier_row = {t.name: i for i, t in enumerate(reversed(tiers))}  # weakest low
    for region in regs:
        y = tier_row.get(region.tier, 0)
        ax_reg.add_patch(
            plt.Rectangle((region.start, y - 0.32), region.length, 0.64,
                          facecolor=ramp.get(region.tier, "#888"),
                          edgecolor="black", linewidth=0.6, alpha=0.95)
        )
        ax_reg.text(region.midpoint, y,
                    tr("region_label", start=region.start, end=region.end,
                       k=region.n_tools),
                    ha="center", va="center", fontsize="xx-small",
                    fontweight="bold")
    ax_reg.set_yticks(list(tier_row.values()))
    ax_reg.set_yticklabels([t.name for t in reversed(tiers)], fontsize="small")
    ax_reg.set_ylim(-0.7, len(tiers) - 0.3)
    ax_reg.set_xlim(0.5, length + 0.5)
    ax_reg.set_xlabel(tr("position"))
    ax_reg.set_title(tr("regions"), fontsize="medium")
    ax_reg.text(0.5, -0.55, tr("covering_note", cov=config.consensus.coverage_fraction),
                transform=ax_reg.transAxes, ha="center", va="top",
                fontsize="xx-small", color="#666666")
    for sp in ("top", "right", "left"):
        ax_reg.spines[sp].set_visible(False)

    # ---- 4. agreement-statistics panel ----------------------------------- #
    total = length

    def pct(n):
        return f"{n} aa ({100 * n / total:.1f}%)" if total else f"{n} aa"

    lines = [(tr("stats_title"), None)]
    lines += [(label, pct(n)) for label, n in
              _agreement_bands(counts, ks, tiers, tr)]
    ax_stats.axis("off")
    y = 0.98
    for head, val in lines:
        if val is None:
            ax_stats.text(0.0, y, head, fontsize="small", fontweight="bold",
                          va="top", transform=ax_stats.transAxes)
            y -= 0.10
        else:
            ax_stats.text(0.0, y, head, fontsize="x-small", va="top",
                          transform=ax_stats.transAxes)
            ax_stats.text(0.04, y - 0.045, val, fontsize="x-small", va="top",
                          color="#333333", transform=ax_stats.transAxes)
            y -= 0.135

    fig.suptitle(tr("title", name=spec.label), fontsize="large",
                 fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.93, bottom=0.07)
    return fig


def plot_tool_analysis(protein_tracks, result, config, figsize=None):
    """Per-tool score tracks + statistics + agreement heatmap + region histogram.

    Reproduces the legacy ``*_analysis.png`` panel. Free-energy predictors
    (``below`` detection) have their y-axis inverted, matching the per-predictor
    ``tracks`` figure — previously the same PASTA 2.0 profile appeared with
    opposite polarity in the two figure families.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.gridspec import GridSpec

    viz = config.viz
    _style.apply_style(viz)
    tr = _tr(viz)
    spec = protein_tracks.spec
    length = protein_tracks.length or 1
    tools = _tool_order(config)
    per_tool = _per_tool_regions(protein_tracks, config)
    panel_size, ks, tiers = _panel_and_thresholds(config, per_tool)
    mat = _coverage_matrix(per_tool, tools, length)
    counts = mat.sum(axis=0).astype(int)
    positions = np.arange(1, length + 1)
    n = len(tools)

    fs = getattr(viz, "resolved_figure_scale", 1.0)
    figsize = figsize or (15 * fs, (0.9 * n + 4.0) * fs)
    fig = plt.figure(figsize=figsize, facecolor="white", constrained_layout=True)
    gs = GridSpec(
        n + 2, 2, figure=fig, width_ratios=[4.4, 1.0],
        height_ratios=[1] * n + [2.3, 1.5],
    )

    base_ax = None
    track_axes = []
    for i, tool in enumerate(config.enabled_tools):
        ax = fig.add_subplot(gs[i, 0], sharex=base_ax)
        base_ax = base_ax or ax
        track_axes.append(ax)
        colour = _style.tool_color(tool, i)
        df = protein_tracks.tracks.get(tool.name)
        if df is not None and len(df):
            x = df["Number"].to_numpy()
            y = df["Score"].to_numpy(dtype=float)
            ax.plot(x, y, color=colour, linewidth=1.0)
            ax.fill_between(x, y, np.nanmin(y), color=colour, alpha=0.12)
            if tool.detection.threshold is not None:
                ax.axhline(tool.detection.threshold, color="#CC0000",
                           linestyle="--", linewidth=0.7, alpha=0.8)
            for s, e in per_tool.get(tool.name, []):
                ax.axvspan(s, e, color="#E8888A", alpha=0.30, linewidth=0)
            if tool.detection.method == "below":
                # lower (more negative) energy => higher propensity
                ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, tr("no_data", tool=tool.name), ha="center",
                    va="center", transform=ax.transAxes, fontsize="xx-small",
                    color="#999999")
        ax.set_ylabel(tool.name, rotation=0, ha="right", va="center",
                      fontsize="xx-small")
        ax.set_xlim(1, length)
        ax.tick_params(axis="both", labelsize="xx-small")
        ax.tick_params(labelbottom=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    base_ax.set_xlim(1, length)

    # statistics sidebar (spans the track rows)
    ax_stats = fig.add_subplot(gs[0:n, 1])
    ax_stats.axis("off")
    slines = [tr("tool_stats"), "=" * 15, ""]
    for tool in config.enabled_tools:
        ivs = per_tool.get(tool.name, [])
        aa = sum(e - s + 1 for s, e in ivs)
        slines += [f"{tool.name}:", f"  {tr('stat_regions', n=len(ivs))}",
                   f"  {tr('stat_aa', n=aa)}", ""]
    ax_stats.text(0, 1, "\n".join(slines), fontfamily="monospace",
                  fontsize="xx-small", va="top", transform=ax_stats.transAxes)

    # agreement heatmap
    ax_heat = fig.add_subplot(gs[n, 0], sharex=base_ax)
    ax_heat.imshow(mat, aspect="auto", cmap=ListedColormap(["#3B5B80", "#8B1A2B"]),
                   vmin=0, vmax=1, interpolation="nearest",
                   extent=[0.5, length + 0.5, len(tools) - 0.5, -0.5])
    ax_heat.set_yticks(range(len(tools)))
    ax_heat.set_yticklabels(tools, fontsize="xx-small")
    ax_heat.set_title(tr("heatmap"), fontsize="small")
    ax_heat.tick_params(axis="x", labelbottom=False)

    # position-wise histogram with annotated consensus regions
    ax_hist = fig.add_subplot(gs[n + 1, 0], sharex=base_ax)
    ax_hist.bar(positions, counts, width=1.0,
                color=_bar_colours(counts, ks, tiers), linewidth=0)
    styles = [":", "--", "-."]
    for i, k in enumerate(reversed(ks)):
        ax_hist.axhline(k, color="#B23A48", linestyle=styles[i % len(styles)],
                        linewidth=1, alpha=0.8)
    ax_hist.set_ylim(0, panel_size + 1.2)
    ax_hist.set_ylabel(tr("n_tools"), fontsize="xx-small")
    for region in result.regions.get(spec.id, []):
        ax_hist.annotate(
            f"{region.start}-{region.end}\n({region.n_tools})",
            xy=(region.midpoint, min(region.n_tools, panel_size)),
            xytext=(region.midpoint, panel_size + 0.35), ha="center", va="bottom",
            fontsize="xx-small", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="#FFF3B0", ec="#C9A227", lw=0.5),
        )
    ax_hist.set_xlabel(tr("position"), fontsize="small")

    for a in (*track_axes, ax_heat):
        a.tick_params(axis="x", labelbottom=False)

    fig.suptitle(tr("title_analysis", name=spec.label, n=panel_size),
                 fontsize="large", fontweight="bold")
    return fig


def plot_cross_protein_overview(result, dataset, config, figsize=None):
    """Per-tool and consensus regions for every protein on one position axis."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    viz = config.viz
    _style.apply_style(viz)
    tr = _tr(viz)
    proteins = config.proteins
    tools = config.enabled_tools
    n_tools = len(tools)

    per_protein, max_len = {}, 1
    for pt in dataset:
        per_protein[pt.spec.id] = _per_tool_regions(pt, config)
        max_len = max(max_len, pt.length or 1)

    n = len(proteins)
    slot = 1.6
    row_h = 1.0 / (n_tools + 1)
    fs = getattr(viz, "resolved_figure_scale", 1.0)
    figsize = figsize or (16 * fs, max(6.0, 1.7 * n) * fs)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Union of the per-protein mappings so the "loaded" denominator resolves
    # against real tracks instead of an empty dict.
    union = {k: v for m in per_protein.values() for k, v in m.items()}
    _, ks, _ = _panel_and_thresholds(config, union)
    moderate_k = ks[-1]

    for i, spec in enumerate(proteins):
        base = i * slot
        ptool = per_protein.get(spec.id, {})
        band_h = n_tools * row_h
        for region in result.regions.get(spec.id, []):
            ax.add_patch(Rectangle(
                (region.start, base - 0.06), region.length, band_h + 0.12,
                facecolor="#F4A6A6", alpha=0.35, edgecolor="none", zorder=1))
            ax.text(region.midpoint, base + band_h / 2, str(region.n_tools),
                    ha="center", va="center", fontsize="xx-small",
                    fontweight="bold", zorder=6)
        for ti, tool in enumerate(tools):
            y = base + ti * row_h
            colour = _style.tool_color(tool, ti)
            for s, e in ptool.get(tool.name, []):
                ax.add_patch(Rectangle((s, y), max(1, e - s + 1), row_h * 0.85,
                             facecolor=colour, alpha=0.8, edgecolor="none",
                             zorder=3))

    ax.set_xlim(0, max_len + 5)
    ax.set_ylim(-0.4, (n - 1) * slot + n_tools * row_h + 0.4)
    ax.set_yticks([i * slot + (n_tools * row_h) / 2 for i in range(n)])
    ax.set_yticklabels([p.label for p in proteins])
    ax.set_xlabel(tr("position"))
    ax.set_ylabel(tr("proteins"))
    ax.set_title(tr("cross_title"), fontsize="large", fontweight="bold", pad=10)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, axis="x", alpha=0.12, linestyle=":", linewidth=0.5)

    handles = [Patch(facecolor=_style.tool_color(t, i), label=t.name)
               for i, t in enumerate(tools)]
    handles.append(Patch(facecolor="#F4A6A6", alpha=0.5,
                         label=tr("consensus_legend", k=moderate_k)))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1),
              fontsize="x-small", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return fig


def plot_all_proteins_agreement(result, dataset, config, figsize=None):
    """Stacked position-wise tool-agreement profiles, one row per protein."""
    import matplotlib.pyplot as plt

    viz = config.viz
    _style.apply_style(viz)
    tr = _tr(viz)
    proteins = [pt for pt in dataset if pt.tracks]
    if not proteins:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.text(0.5, 0.5, tr("no_data", tool="—"), ha="center", va="center")
        ax.axis("off")
        return fig

    n = len(proteins)
    fs = getattr(viz, "resolved_figure_scale", 1.0)
    figsize = figsize or (12 * fs, max(5.0, 2.5 * n) * fs)
    fig, axes = plt.subplots(n, 1, figsize=figsize, squeeze=False)
    axes = axes[:, 0]

    for ax, pt in zip(axes, proteins, strict=True):
        length = pt.length or 1
        per_tool = _per_tool_regions(pt, config)
        panel_size, ks, tiers = _panel_and_thresholds(config, per_tool)
        mat = _coverage_matrix(per_tool, _tool_order(config), length)
        counts = mat.sum(axis=0).astype(int)
        positions = np.arange(1, length + 1)
        ax.bar(positions, counts, width=1.0,
               color=_bar_colours(counts, ks, tiers), linewidth=0)
        styles = [":", "--", "-."]
        seen = set()
        for i, (tier, k) in enumerate(
            zip(reversed(tiers), reversed(ks), strict=True)
        ):
            if k in seen:
                continue
            seen.add(k)
            ax.axhline(k, color="#B23A48", linestyle=styles[i % len(styles)],
                       linewidth=0.8, alpha=0.7, label=tr(tier.name, k=k))
        ax.set_ylabel(pt.spec.label, fontweight="bold", fontsize=14)
        ax.set_ylim(0, panel_size + 0.5)
        ax.set_xlim(0, length)
        ax.tick_params(labelsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_title(tr("profile"), fontsize="large", fontweight="bold")
    axes[0].legend(loc="upper right", fontsize="xx-small", framealpha=0.9)
    axes[-1].set_xlabel(tr("position"), fontweight="bold", fontsize=14)
    fig.supylabel(tr("n_tools"), fontsize="small")
    fig.tight_layout()
    return fig