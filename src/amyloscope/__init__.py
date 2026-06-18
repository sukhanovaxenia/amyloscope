"""amyloscope — consensus meta-scoring, domain mapping and visualisation of
amyloidogenicity predictions.

A configuration-driven post-processing layer that ingests per-residue outputs
from multiple aggregation predictors, calls per-tool aggregation-prone regions,
builds fractional cross-predictor consensus, maps consensus regions onto an
arbitrary user-defined domain architecture, and renders publication figures.

Quick start
-----------
>>> from amyloscope import run_from_file
>>> artifacts = run_from_file("config.yaml")
>>> len(artifacts.consensus.all_regions())

Or compose the stages directly:

>>> from amyloscope import load_config, load_dataset, compute_consensus
>>> cfg = load_config("config.yaml")
>>> result = compute_consensus(load_dataset(cfg))
"""

from __future__ import annotations

__version__ = "0.1.0"

# Importing the adapters module registers the built-in predictor parsers.
from .analysis.domains import compute_domain_overlap, format_overlap_report
from .analysis.statistics import compute_statistics, format_report
from .config import (
    ConfigError,
    ConsensusConfig,
    ConsensusTier,
    DetectionStrategy,
    Domain,
    PipelineConfig,
    ProteinSpec,
    ToolSpec,
    VizConfig,
    load_config,
    with_overrides,
)
from .core.consensus import (
    ConsensusRegion,
    ConsensusResult,
    compute_consensus,
    consensus_for_protein,
)
from .io import adapters as _adapters  # noqa: F401  (registration side-effect)
from .io.adapters import available_adapters, register_adapter
from .io.loader import Dataset, ProteinTracks, load_dataset
from .pipeline import PipelineArtifacts, run, run_from_file

__all__ = [
    "__version__",
    # config
    "load_config",
    "with_overrides",
    "PipelineConfig",
    "ProteinSpec",
    "ToolSpec",
    "Domain",
    "DetectionStrategy",
    "ConsensusConfig",
    "ConsensusTier",
    "VizConfig",
    "ConfigError",
    # io
    "load_dataset",
    "Dataset",
    "ProteinTracks",
    "register_adapter",
    "available_adapters",
    # core
    "compute_consensus",
    "consensus_for_protein",
    "ConsensusResult",
    "ConsensusRegion",
    # analysis
    "compute_statistics",
    "format_report",
    "compute_domain_overlap",
    "format_overlap_report",
    # pipeline
    "run",
    "run_from_file",
    "PipelineArtifacts",
]
