"""Load all predictor tracks for a configured panel.

Produces the in-memory dataset consumed by the consensus and analysis layers,
and resolves each protein's length and reference sequence (from config when
curated, otherwise inferred from the loaded tracks).
"""

from __future__ import annotations

import inspect
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


def _call_adapter(parser, path, sequence: str, tool) -> pd.DataFrame:
    """Invoke an adapter, forwarding only the options it declares.

    Adapters have a two-positional core signature ``(path, sequence)`` and may
    add keyword-only modelling choices. Forwarding by signature means a tool
    entry can carry ``options:`` without every adapter having to accept
    ``**kwargs``, and a key the adapter does not understand is reported against
    the config line that set it rather than silently ignored — the failure mode
    that left ``score_column`` documented but inert.
    """
    options = getattr(tool, "options", None) or {}
    if not options:
        return parser(path, sequence)
    accepted = set(inspect.signature(parser).parameters)
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise ValueError(
            f"tool '{tool.name}': adapter '{tool.adapter}' does not accept "
            f"option(s) {unknown}; it accepts {sorted(accepted - {'path', 'sequence'})}"
        )
    return parser(path, sequence, **options)


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


def _warn_shared_paths(config: PipelineConfig) -> None:
    """Flag tool paths that cannot vary by protein.

    A ``path`` without a ``{protein}`` placeholder resolves to the same file for
    every member of the panel, so one protein's track is silently attributed to
    all of them. Nothing downstream can detect this: the track parses, the
    consensus counts it, and the only symptom is that a predictor agrees with
    itself across proteins. The shipped example config had exactly this for
    amyloid_predict.
    """
    if len(config.proteins) < 2:
        return
    for tool in config.enabled_tools:
        if "{protein}" not in tool.path_template:
            warnings.warn(
                f"tool '{tool.name}': path '{tool.path_template}' has no "
                f"'{{protein}}' placeholder, so all {len(config.proteins)} "
                f"proteins would be read from the same file. If that is "
                f"intended (a single combined export), the adapter must select "
                f"the protein's record itself.",
                stacklevel=3,
            )


def load_dataset(config: PipelineConfig) -> Dataset:
    """Load every enabled predictor's track for every configured protein."""
    _warn_shared_paths(config)
    dataset = Dataset(config=config)
    for spec in config.proteins:
        pt = ProteinTracks(spec=spec)
        for tool in config.enabled_tools:
            path = tool.path_for(spec.id)
            try:
                parser = get_adapter(tool.adapter)
                pt.tracks[tool.name] = _call_adapter(parser, path, spec.sequence, tool)
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
