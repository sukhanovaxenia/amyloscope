"""Load all predictor tracks for a configured panel.

Produces the in-memory dataset consumed by the consensus and analysis layers,
and resolves each protein's length and reference sequence (from config when
curated, otherwise inferred from the loaded tracks).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pandas as pd

from ..config import PipelineConfig, ProteinSpec
from .adapters import AdapterError, get_adapter


@dataclass
class ProteinTracks:
    """All successfully-loaded predictor tracks for one protein."""

    spec: ProteinSpec
    tracks: dict[str, pd.DataFrame] = field(default_factory=dict)
    length: int = 0
    sequence: str = ""
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def loaded_tools(self) -> list[str]:
        return list(self.tracks)

    def track(self, tool: str) -> pd.DataFrame:
        return self.tracks[tool]


@dataclass
class Dataset:
    """Loaded tracks for the whole panel plus the originating config."""

    config: PipelineConfig
    proteins: dict[str, ProteinTracks] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.proteins.values())

    def __getitem__(self, protein_id: str) -> ProteinTracks:
        return self.proteins[protein_id]


def _resolve_length(spec: ProteinSpec, tracks: dict[str, pd.DataFrame]) -> int:
    """Curated length wins; otherwise take the max residue index observed."""
    if spec.length:
        return spec.length
    if not tracks:
        return 0
    return int(max(df["Number"].max() for df in tracks.values()))


def _resolve_sequence(
    spec: ProteinSpec, tracks: dict[str, pd.DataFrame], length: int
) -> str:
    """Prefer the configured sequence; else reconstruct from any track that
    carries residue identities (per-residue adapters such as Aggrescan).
    """
    if spec.sequence:
        return spec.sequence
    for df in tracks.values():
        residues = df.set_index("Number")["Residue"]
        if residues.replace("", pd.NA).notna().any():
            seq = ["X"] * length
            for pos, res in residues.items():
                if res and 1 <= pos <= length:
                    seq[pos - 1] = res
            return "".join(seq)
    return "X" * length


def load_dataset(config: PipelineConfig) -> Dataset:
    """Load every enabled predictor's track for every configured protein."""
    dataset = Dataset(config=config)
    for spec in config.proteins:
        pt = ProteinTracks(spec=spec)
        for tool in config.enabled_tools:
            path = tool.path_for(spec.id)
            try:
                parser = get_adapter(tool.adapter)
                pt.tracks[tool.name] = parser(path)
            except (AdapterError, FileNotFoundError, OSError, ValueError) as exc:
                pt.failures[tool.name] = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    f"[{spec.id}] failed to load '{tool.name}' from {path}: {exc}",
                    stacklevel=2,
                )
        pt.length = _resolve_length(spec, pt.tracks)
        pt.sequence = _resolve_sequence(spec, pt.tracks, pt.length)
        dataset.proteins[spec.id] = pt
    return dataset
