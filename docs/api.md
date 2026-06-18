# API reference

Rendered from the source docstrings. The public surface is also re-exported from
the top-level `amyloscope` namespace (see `amyloscope.__all__`).

## Configuration

::: amyloscope.config
    options:
      members:
        - PipelineConfig
        - ProteinSpec
        - Domain
        - ToolSpec
        - DetectionStrategy
        - ConsensusConfig
        - ConsensusTier
        - VizConfig
        - load_config
        - with_overrides
        - ConfigError

## I/O

::: amyloscope.io.adapters
    options:
      members:
        - register_adapter
        - get_adapter
        - available_adapters

::: amyloscope.io.loader
    options:
      members:
        - load_dataset
        - Dataset
        - ProteinTracks

## Core

::: amyloscope.core.regions
    options:
      members:
        - call_regions

::: amyloscope.core.consensus
    options:
      members:
        - compute_consensus
        - consensus_for_protein
        - resolve_count
        - ConsensusResult
        - ConsensusRegion

## Analysis

::: amyloscope.analysis.statistics
    options:
      members:
        - compute_statistics
        - format_report

::: amyloscope.analysis.domains
    options:
      members:
        - compute_domain_overlap
        - format_overlap_report

## Pipeline

::: amyloscope.pipeline
    options:
      members:
        - run
        - run_from_file
        - PipelineArtifacts
