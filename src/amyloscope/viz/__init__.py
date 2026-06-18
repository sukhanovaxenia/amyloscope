"""Figure generation. Importing this module requires matplotlib."""

from .consensus import plot_distribution, plot_positional_enrichment
from .domains import plot_domain_architecture
from .tracks import plot_tool_tracks

__all__ = [
    "plot_distribution",
    "plot_positional_enrichment",
    "plot_domain_architecture",
    "plot_tool_tracks",
]
