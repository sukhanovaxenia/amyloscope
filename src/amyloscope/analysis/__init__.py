"""Statistical and domain-overlap analysis (rendering-independent)."""

from .domains import (
    DomainOverlapResult,
    compute_domain_overlap,
    format_overlap_report,
)
from .statistics import (
    ConsensusStatistics,
    compute_statistics,
    format_report,
)

__all__ = [
    "DomainOverlapResult",
    "compute_domain_overlap",
    "format_overlap_report",
    "ConsensusStatistics",
    "compute_statistics",
    "format_report",
]
