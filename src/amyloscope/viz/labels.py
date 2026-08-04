"""Localised figure text.

All user-visible strings in the figures are looked up here by key, so a single
``viz.language`` setting switches every label, title and legend between English
and Russian without touching the builders. Adding a language is a matter of
adding one nested dict; a missing key in a non-default language falls back to
English, and an unknown key falls back to the key itself (so a typo is visible
rather than silently blank).

Placeholders use ``str.format`` fields (e.g. ``{name}``, ``{value}``), filled by
the caller.

Note: :mod:`amyloscope.viz.detailed` still carries its own private catalogue.
That duplication is deliberate (the module is self-contained by design) but it
means a new string has to be added in two places — check both when editing.
"""

from __future__ import annotations

from collections.abc import Callable

DEFAULT_LANGUAGE = "en"

#: Consensus-tier display names. Kept separate from the ``LABELS`` templates so
#: a tier name can be localised without the caller having to interpolate an
#: English word into a Russian sentence.
TIER_NAMES: dict[str, dict[str, str]] = {
    "en": {"unanimous": "Unanimous", "strong": "Strong", "moderate": "Moderate"},
    "ru": {"unanimous": "Единогласный", "strong": "Сильный", "moderate": "Средний"},
}

LABELS: dict[str, dict[str, str]] = {
    "en": {
        # consensus distribution
        "axis_position": "Amino acid position",
        "axis_protein": "Protein",
        "title_consensus": "Consensus aggregation-prone regions — {name}",
        # positional enrichment
        "no_regions": "No consensus regions",
        "expected_uniform": "Expected (uniform)",
        "axis_norm_position": "Normalised position (0 = N-terminus, 1 = C-terminus)",
        "axis_n_regions": "Number of regions",
        "title_enrichment": "Positional enrichment of consensus regions",
        # tracks
        "no_tracks": "No predictor tracks loaded",
        "threshold": "threshold {value}",
        "summary_regions": "Regions: {n}",
        "summary_total_aa": "Total aa: {n}",
        "summary_avg_len": "Avg len: {value}",
        "title_tracks": "Per-predictor amyloidogenic profiles — {label}",
        # domains
        "axis_position_aa": "Position (aa)",
        "length_aa": "{length} aa",
        "legend_tier_consensus": "{tier} consensus",
        "legend_domain": "domain: {cat}",
        "legend_title": "Legend",
        "title_domains": "Domain architecture and consensus APR mapping — {name}",
        # consensus vs external measurement
        "corr_title": "Consensus prediction versus measured behaviour",
        "corr_alt_title": "Rank correlation by choice of summary",
        "corr_y_default": "Measured value",
        "corr_rho_axis": "Spearman \u03c1",
        "corr_x_region_count": "Consensus regions (count)",
        "corr_x_moderate_band": "Residues at moderate tier (%)",
        "corr_x_weak_band": "Residues below moderate tier (%)",
        "corr_stat": "Spearman $\\rho$ = {rho}\n$p$ = {p}   $n$ = {n} proteins",
        "corr_floor": "smallest attainable $p$\nat this $n$: {floor}\n(exact permutation)",
        "corr_floor_mc": "permutation resolution:\n{floor} (sampled null)",
        "corr_descriptive": "descriptive: ranking only,\nnot a hypothesis test",
        "corr_too_few": "Fewer than three proteins with a measurement",
        # domain overlap
        "ov_title": "APR residues by domain category",
        "ov_null_title": "Relocation null, pre-specified category",
        "ov_no_regions": "No consensus regions to place",
        "ov_x_residues": "Consensus APR residues",
        "ov_x_null": "APR residues in {cat} domains",
        "ov_y_null": "Placements",
        "ov_cat": "{cat}\n({cov} % of sequence)",
        "ov_expected": "expected under the null",
        "ov_observed": "observed",
        "ov_legend_observed": "observed",
        "ov_legend_null": "null distribution (IQR, 95 %)",
        "ov_cap_head": "Consensus APR residues by domain category. "
                       "Boxes give the interquartile range and whiskers the "
                       "central 95 % of a null in which all {n} APR residues "
                       "from {np} proteins are relocated as whole regions of "
                       "unchanged length ({draws} placements); markers give the "
                       "observed counts.",
        "ov_cap_primary": "Enrichment within {cat} domains was {fold}-fold "
                          "({obs} observed, {exp} expected; one-sided "
                          "permutation p = {p}), the pre-specified comparison.",
        "ov_cap_explore": "Remaining categories ({cats}) are exploratory and "
                          "Holm-corrected. * p < 0.05, ** p < 0.01, "
                          "*** p < 0.001.",
        "ov_stat_head": "{inside} of {n} APR residues\nlie in a domain ({pct} %)",
        "ov_stat_primary": "{cat}: {fold}x enriched\none-sided $p$ = {p}\n(pre-specified)",
        "ov_stat_explore": "{n} further categories\nexploratory, Holm-corrected",
        "ov_stat_null": "whole regions relocated,\nlengths preserved\n({draws} placements)",
    },
    "ru": {
        # consensus distribution
        "axis_position": "Позиция ак",
        "axis_protein": "Белок",
        "title_consensus": "Консенсусные предсказания участков, склонных "
                           "к агрегации — {name}",
        # positional enrichment
        "no_regions": "Нет консенсусных регионов",
        "expected_uniform": "Ожидаемое (равномерное)",
        "axis_norm_position": "Нормализованные позиции (0 = N-конец, 1 = C-конец)",
        "axis_n_regions": "Количество регионов",
        "title_enrichment": "Позиционное обогащение консенсусных областей",
        # tracks
        "no_tracks": "Нет загруженных треков предсказателей",
        "threshold": "порог {value}",
        "summary_regions": "Регионы: {n}",
        "summary_total_aa": "Общее кол-во ак: {n}",
        "summary_avg_len": "Средняя длина: {value}",
        "title_tracks": "Результаты предсказания амилоидогенности "
                        "(сравнительный анализ) — {label}",
        # domains
        "axis_position_aa": "Позиция (ак)",
        "length_aa": "{length} ак",
        "legend_tier_consensus": "{tier} консенсус",
        "legend_domain": "домен: {cat}",
        "legend_title": "Легенда",
        "title_domains": "Анализ доменной архитектуры и картирование "
                         "консенсусных участков агрегации (APR) — {name}",
        # consensus vs external measurement
        "corr_title": "Консенсусное предсказание и измеренное поведение",
        "corr_alt_title": "Ранговая корреляция по выбору сводной меры",
        "corr_y_default": "Измеренная величина",
        "corr_rho_axis": "\u03c1 Спирмена",
        "corr_x_region_count": "Консенсусные регионы (количество)",
        "corr_x_moderate_band": "Остатки среднего уровня (%)",
        "corr_x_weak_band": "Остатки ниже среднего уровня (%)",
        "corr_stat": "$\\rho$ Спирмена = {rho}\n$p$ = {p}   $n$ = {n} белков",
        "corr_floor": "минимально достижимое $p$\nпри данном $n$: {floor}\n(точная перестановка)",
        "corr_floor_mc": "разрешение перестановочного теста: {floor} "
                         "(выборочная нулевая модель)",
        "corr_descriptive": "описательно: только\nранжирование, не проверка\nгипотезы",
        "corr_too_few": "Менее трёх белков с измерением",
        # domain overlap
        "ov_title": "Остатки APR по категориям доменов",
        "ov_null_title": "Нулевая модель перемещения, заданная категория",
        "ov_no_regions": "Нет консенсусных регионов",
        "ov_x_residues": "Остатки консенсусных APR",
        "ov_x_null": "Остатки APR в доменах {cat}",
        "ov_y_null": "Размещения",
        "ov_cat": "{cat}\n({cov} % последовательности)",
        "ov_expected": "ожидаемое по нулевой модели",
        "ov_observed": "наблюдаемое",
        "ov_legend_observed": "наблюдаемое",
        "ov_legend_null": "нулевая модель (IQR, 95 %)",
        "ov_cap_head": "Остатки консенсусных APR по категориям доменов. "
                       "Прямоугольник — межквартильный размах, усы — central 95 % "
                       "нулевой модели, в которой все {n} остатков APR из {np} "
                       "белков перемещаются целыми регионами неизменной длины "
                       "({draws} размещений); маркеры — наблюдаемые значения.",
        "ov_cap_primary": "Обогащение в доменах {cat} составило {fold}x "
                          "({obs} наблюдаемых, {exp} ожидаемых; односторонний "
                          "перестановочный p = {p}) — заданное заранее сравнение.",
        "ov_cap_explore": "Остальные категории ({cats}) разведочные, с поправкой "
                          "Холма. * p < 0,05, ** p < 0,01, *** p < 0,001.",
        "ov_stat_head": "{inside} из {n} остатков APR\nв доменах ({pct} %)",
        "ov_stat_primary": "{cat}: обогащение {fold}x\nодносторонний $p$ = {p}\n(заданная заранее)",
        "ov_stat_explore": "ещё {n} категорий\nразведочно, поправка Холма",
        "ov_stat_null": "перемещение целых регионов,\nдлины сохранены\n({draws} размещений)",
    },
}

SUPPORTED_LANGUAGES = tuple(LABELS)


def tier_name(language: str, tier: str) -> str:
    """Localised display name for a consensus tier.

    Falls back to English, then to a capitalised form of the raw tier name, so
    a config that defines a custom tier still renders sensibly.
    """
    table = TIER_NAMES.get(language, TIER_NAMES[DEFAULT_LANGUAGE])
    english = TIER_NAMES[DEFAULT_LANGUAGE]
    return table.get(tier, english.get(tier, tier.capitalize()))


def translator(language: str) -> Callable[..., str]:
    """Return a ``tr(key, **fields)`` function bound to ``language``.

    Falls back to English for a missing key in another language, and to the key
    itself if it is unknown everywhere.
    """
    table = LABELS.get(language, LABELS[DEFAULT_LANGUAGE])
    english = LABELS[DEFAULT_LANGUAGE]

    def tr(key: str, **fields: object) -> str:
        template = table.get(key, english.get(key, key))
        return template.format(**fields) if fields else template

    return tr