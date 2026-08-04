"""Figure generation. Importing this module requires matplotlib."""

from .consensus import plot_distribution, plot_positional_enrichment
from .correlation import plot_measurement_correlation
from .domains import plot_domain_architecture
from .tracks import plot_tool_tracks
from .overlap import plot_domain_overlap

__all__ = [
    "plot_distribution",
    "plot_positional_enrichment",
    "plot_measurement_correlation",
    "plot_domain_architecture",
    "plot_tool_tracks",
    "plot_domain_overlap"
]