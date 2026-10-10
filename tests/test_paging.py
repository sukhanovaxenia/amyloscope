"""Figure pagination: filenames, page views, and the size guard."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pytest

from amyloscope.viz import paging


# --------------------------------------------------------------------------- #
# chunk / page_name
# --------------------------------------------------------------------------- #
def test_small_input_is_one_page() -> None:
    assert paging.chunk(list(range(5)), 24) == [list(range(5))]


def test_exact_fit_is_one_page() -> None:
    items = list(range(24))
    assert paging.chunk(items, 24) == [items]


def test_large_input_splits() -> None:
    pages = paging.chunk(list(range(372)), 24)
    assert len(pages) == 16
    assert sum(len(p) for p in pages) == 372
    assert len(pages[-1]) == 372 - 15 * 24


def test_pagination_can_be_disabled() -> None:
    items = list(range(100))
    assert paging.chunk(items, 0) == [items]
    assert paging.chunk(items, None) == [items]


def test_single_page_keeps_the_original_filename() -> None:
    """A small run must not change shape just because pagination exists."""
    assert paging.page_name("consensus_distribution.png", 0, 1) == (
        "consensus_distribution.png"
    )


def test_multi_page_names_are_ordered_and_padded() -> None:
    names = [paging.page_name("domain_architecture.png", i, 16) for i in range(3)]
    assert names == [
        "domain_architecture_001.png",
        "domain_architecture_002.png",
        "domain_architecture_003.png",
    ]
    assert names == sorted(names)


def test_page_name_handles_a_stem_without_a_suffix() -> None:
    assert paging.page_name("figure", 1, 5) == "figure_002"


# --------------------------------------------------------------------------- #
# page_view
# --------------------------------------------------------------------------- #
@dataclass
class FakeSpec:
    id: str


@dataclass
class FakeConfig:
    proteins: list
    name: str = "panel"


@dataclass
class FakeResult:
    config: FakeConfig
    regions: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)
    raw_regions: dict = field(default_factory=dict)


def _result(n: int) -> FakeResult:
    specs = [FakeSpec(f"p{i}") for i in range(n)]
    return FakeResult(
        config=FakeConfig(proteins=specs),
        regions={s.id: [s.id] for s in specs},
        rejected={s.id: [] for s in specs},
        raw_regions={s.id: [s.id] for s in specs},
    )


def test_page_view_restricts_every_mapping() -> None:
    res = _result(10)
    page = paging.page_view(res, res.config.proteins[2:5])
    assert [s.id for s in page.config.proteins] == ["p2", "p3", "p4"]
    assert set(page.regions) == {"p2", "p3", "p4"}
    assert set(page.rejected) == {"p2", "p3", "p4"}
    assert set(page.raw_regions) == {"p2", "p3", "p4"}


def test_page_view_does_not_mutate_the_original() -> None:
    res = _result(10)
    paging.page_view(res, res.config.proteins[:2])
    assert len(res.config.proteins) == 10
    assert len(res.regions) == 10


def test_protein_pages_covers_every_protein_exactly_once() -> None:
    res = _result(100)
    seen: list[str] = []
    totals = set()
    for page, i, total in paging.protein_pages(res, per_page=24):
        seen.extend(s.id for s in page.config.proteins)
        totals.add(total)
    assert seen == [s.id for s in res.config.proteins]
    assert totals == {5}


# --------------------------------------------------------------------------- #
# The size guard
# --------------------------------------------------------------------------- #
class FakeFig:
    def __init__(self, w: float, h: float) -> None:
        self._size = (w, h)

    def get_size_inches(self):
        return self._size


def test_ordinary_figure_is_not_flagged() -> None:
    assert paging.oversized(FakeFig(14, 9), 300) is None


def test_the_real_domain_architecture_figure_is_flagged() -> None:
    """4,604 x 377,859 px at 300 dpi -- 1.74 gigapixels, past Pillow's guard."""
    size = paging.oversized(FakeFig(15.35, 1259.5), 300)
    assert size is not None
    assert size[1] > paging.MAX_PIXELS_PER_AXIS


def test_fit_dpi_brings_it_inside_the_budget() -> None:
    fig = FakeFig(15.35, 1259.5)
    dpi = paging.fit_dpi(fig, 300)
    assert dpi < 300
    assert paging.oversized(fig, dpi) is None


def test_fit_dpi_never_raises_resolution() -> None:
    assert paging.fit_dpi(FakeFig(8, 6), 300) == 300


def test_save_figure_warns_rather_than_writing_an_unopenable_file(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from amyloscope.config import VizConfig
    from amyloscope.viz.style import save_figure

    fig = plt.figure(figsize=(6, 300))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            written = save_figure(fig, tmp_path / "huge.png",
                                  VizConfig(dpi=300, save_svg=False))
        assert any("beyond the" in str(w.message) for w in caught)
        assert written
        from PIL import Image

        with Image.open(written[0]) as im:
            assert max(im.size) <= paging.MAX_PIXELS_PER_AXIS
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Cluster pagination
# --------------------------------------------------------------------------- #
@dataclass
class FakeClusterResult:
    clusters: dict = field(default_factory=dict)
    breadth: dict = field(default_factory=dict)

    def all_clusters(self):
        return [c for cs in self.clusters.values() for c in cs]


def test_cluster_pages_split_on_cluster_count_not_protein_count() -> None:
    """A ClusterResult is not a list, and its figure scales with CLUSTERS.

    Paginating it as if it were an iterable of clusters raised
    `TypeError: 'ClusterResult' object is not iterable`; paginating it by
    protein would still leave one page holding however many clusters those
    proteins happen to carry.
    """
    res = FakeClusterResult(clusters={f"p{i}": [f"c{i}_{j}" for j in range(10)]
                                      for i in range(10)})
    pages = list(paging.cluster_pages(res, 24))
    assert len(pages) == 5
    assert {total for _, _, total in pages} == {5}
    assert all(len(page.all_clusters()) <= 24 for page, _, _ in pages)


def test_cluster_pages_preserve_every_cluster_in_order() -> None:
    res = FakeClusterResult(clusters={"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]})
    seen = [c for page, _, _ in paging.cluster_pages(res, 2)
            for c in page.all_clusters()]
    assert seen == ["a1", "a2", "a3", "b1", "b2"]


def test_cluster_pages_keep_per_protein_grouping() -> None:
    res = FakeClusterResult(clusters={"a": ["a1", "a2"], "b": ["b1", "b2"]})
    first = next(iter(paging.cluster_pages(res, 3)))[0]
    assert set(first.clusters) == {"a", "b"}
    assert first.clusters["a"] == ["a1", "a2"]
    assert first.clusters["b"] == ["b1"]


def test_small_cluster_set_is_one_page_with_the_original_name() -> None:
    res = FakeClusterResult(clusters={"a": ["a1", "a2"]})
    pages = list(paging.cluster_pages(res, 24))
    assert len(pages) == 1
    assert paging.page_name("apr_clusters.png", 0, 1) == "apr_clusters.png"
