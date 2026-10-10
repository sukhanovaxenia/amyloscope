"""Pagination for the figures whose height grows with the protein count.

Several panel figures draw one lane per protein and size themselves as
``lane_step * n``. That is right for the handful of proteins these were written
for and produces an unopenable file for an orthologue sweep. Measured on a
372-protein run at the default 300 dpi:

    domain_architecture.png     4,604 x 377,859 px   34 MB   1.74 gigapixels
    apr_clusters.png            2,692 x 334,802 px   21 MB
    all_protein_agreement.png   3,549 x 278,978 px   20 MB
    all_proteins_consensus.png  4,778 x 189,698 px   13 MB
    consensus_distribution.png  4,171 x 167,829 px   13 MB
    per_protein_positional.png  3,759 x 103,137 px   10 MB

Pillow refuses anything past ``MAX_IMAGE_PIXELS`` (89 Mpx) as a decompression
bomb, so the largest cannot be opened by the imaging library this project
already depends on without an explicit override; cropping one OOM-killed a 7 GB
machine. Most viewers decline them too. The content is good -- these are useful
per-protein panels -- but a single image is the wrong container for 372 of them.

The approach here deliberately does NOT touch the plotting functions. A page is
a shallow *view* of the result carrying a subset of proteins, so each function
runs unchanged and simply sees fewer proteins. Nothing about a single-page run
changes: with ``per_page`` at or above the protein count there is exactly one
page and the filename keeps its original name.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")

#: Proteins per page. 24 lanes is about 36 inches at the usual 1.5-inch lane
#: step -- a tall page, but one that opens, prints and scrolls.
DEFAULT_PER_PAGE = 24

#: Refuse to render beyond this many pixels in either direction. Well under
#: Pillow's bomb guard and under the 65,500-pixel limit several viewers and the
#: JPEG/TIFF encoders impose.
MAX_PIXELS_PER_AXIS = 30_000


def chunk(items: Sequence[T], per_page: int) -> list[Sequence[T]]:
    """Split into pages of at most ``per_page``; one page when it already fits."""
    if per_page is None or per_page <= 0 or len(items) <= per_page:
        return [items]
    return [items[i : i + per_page] for i in range(0, len(items), per_page)]


def page_name(name: str, index: int, total: int) -> str:
    """``consensus_distribution.png`` -> ``consensus_distribution_003.png``.

    A single page keeps the original filename, so an existing run directory,
    a report that references the name, or a downstream script does not change
    shape just because pagination exists.
    """
    if total <= 1:
        return name
    stem, dot, suffix = name.rpartition(".")
    if not dot:                      # no extension: rpartition puts it all in `suffix`
        return f"{name}_{index + 1:03d}"
    return f"{stem}_{index + 1:03d}.{suffix}"


def page_view(result, specs):
    """A ``ConsensusResult`` restricted to ``specs``.

    ``PipelineConfig`` is a dataclass and ``config.protein()`` is a linear scan
    with no caching, so replacing the protein list is enough to give every
    plotting function a consistent smaller panel. The regions dictionaries are
    filtered to match; nothing is copied deeply.
    """
    ids = {s.id for s in specs}
    page_config = replace(result.config, proteins=list(specs))
    return type(result)(
        config=page_config,
        regions={k: v for k, v in result.regions.items() if k in ids},
        rejected={k: v for k, v in result.rejected.items() if k in ids},
        raw_regions={k: v for k, v in result.raw_regions.items() if k in ids},
    )


def protein_pages(result, per_page: int = DEFAULT_PER_PAGE):
    """Yield ``(page_result, index, total)`` for a protein-stacked figure."""
    pages = chunk(result.config.proteins, per_page)
    for i, specs in enumerate(pages):
        yield page_view(result, specs), i, len(pages)


def cluster_pages(result, per_page: int):
    """Split a ``ClusterResult`` into pages of at most ``per_page`` clusters.

    Clusters are paginated on their OWN count rather than the protein count,
    because the cluster figure draws one lane per cluster and a protein
    contributes however many it has: 372 proteins produced 2,654 lanes and a
    334,802-pixel figure. Per-protein grouping is preserved inside each page, so
    a page still reads protein by protein.

    Yields ``(page_result, index, total)``. ``plot_cluster_cores`` reads only
    ``all_clusters()``, so a page is the same type with a smaller mapping.
    """
    flat = [(pid, c) for pid, cs in result.clusters.items() for c in cs]
    pages = chunk(flat, per_page)
    for i, page in enumerate(pages):
        grouped: dict[str, list] = {}
        for pid, c in page:
            grouped.setdefault(pid, []).append(c)
        yield replace(result, clusters=grouped), i, len(pages)


def oversized(fig, dpi: float) -> tuple[int, int] | None:
    """Pixel dimensions if the figure exceeds the axis budget, else ``None``."""
    w_in, h_in = fig.get_size_inches()
    w, h = int(w_in * dpi), int(h_in * dpi)
    if w > MAX_PIXELS_PER_AXIS or h > MAX_PIXELS_PER_AXIS:
        return w, h
    return None


def fit_dpi(fig, dpi: float) -> float:
    """Largest dpi at or below ``dpi`` that keeps the figure inside the budget.

    A last-resort guard for a figure that is oversized for a reason pagination
    does not address -- a single very long chain, say. Lowering resolution is a
    poor substitute for paginating and the caller should warn; an unopenable
    file is worse.
    """
    w_in, h_in = fig.get_size_inches()
    longest = max(w_in, h_in)
    if longest <= 0:
        return dpi
    # 2 % of headroom: figures are written with bbox_inches="tight", which adds
    # the artists that overhang the canvas, so a dpi computed to land exactly on
    # the limit produces a file a little over it.
    return min(dpi, 0.98 * MAX_PIXELS_PER_AXIS / longest)
