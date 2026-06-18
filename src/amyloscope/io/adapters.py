"""Predictor-output adapters.

Each adapter parses one predictor's native output into a *normalised track*: a
:class:`pandas.DataFrame` carrying at minimum the columns

    Number   1-based residue index (int)
    Residue  one-letter code (str; may be '' when the format omits it)
    Score    numeric per-residue score (float)

plus any auxiliary columns a tool's detection strategy reads (e.g. FoldAmyloid's
``Fold`` flag, APPNN's ``is_hotspot``, Aggrescan's ``Prediction``).

Adapters are looked up by the ``adapter`` key in each ``ToolSpec``. Registering a
new predictor is therefore a two-line change in user code plus a config entry;
no core module is touched. This is the extension seam that replaces the original
``if tool == ...`` parsing ladder in ``AmyloidAnalyzer.load_data``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

#: adapter key -> parser callable(path) -> normalised DataFrame
_REGISTRY: dict[str, Callable[[Path], pd.DataFrame]] = {}

REQUIRED_COLUMNS = ("Number", "Residue", "Score")


class AdapterError(RuntimeError):
    """Raised when a predictor output cannot be parsed into a normalised track."""


def register_adapter(key: str) -> Callable[[Callable], Callable]:
    """Decorator registering a parser under ``key`` (case-insensitive)."""

    def decorator(fn: Callable[[Path], pd.DataFrame]) -> Callable[[Path], pd.DataFrame]:
        _REGISTRY[key.lower()] = fn
        return fn

    return decorator


def get_adapter(key: str) -> Callable[[Path], pd.DataFrame]:
    try:
        return _REGISTRY[key.lower()]
    except KeyError as exc:
        raise AdapterError(
            f"no adapter registered for '{key}'. Registered: "
            f"{sorted(_REGISTRY)}"
        ) from exc


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def _finalise(df: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Coerce dtypes, enforce the required schema, sort by residue index."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise AdapterError(f"{source}: normalised track missing columns {missing}")
    df = df.copy()
    df["Number"] = pd.to_numeric(df["Number"], errors="coerce").astype("Int64")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Residue"] = df["Residue"].astype(str).str.strip()
    df = df.dropna(subset=["Number"]).astype({"Number": int})
    return df.sort_values("Number").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Adapters for the eight predictors carried over from the ribosomal pipeline
# --------------------------------------------------------------------------- #


@register_adapter("aggrescan")
def parse_aggrescan(path: Path) -> pd.DataFrame:
    """Aggrescan per-residue CSV. APR rows carry a non-null ``Prediction``."""
    df = pd.read_csv(path, sep=r",|;", header=1, engine="python")
    df.columns = ["Number", "Residue", "Score", "HSA", "NHSA", "a4vAHS", "Prediction"]
    return _finalise(df, path)


@register_adapter("appnn")
def parse_appnn(path: Path) -> pd.DataFrame:
    """APPNN TSV. Hotspot residues flagged by the boolean ``is_hotspot``."""
    df = pd.read_csv(path, sep="\t", header=0)
    df = df.drop(columns=[c for c in ("overall", "id") if c in df.columns])
    df.columns = ["Number", "Residue", "Score", "is_hotspot"]
    return _finalise(df, path)


@register_adapter("foldamyloid")
def parse_foldamyloid(path: Path) -> pd.DataFrame:
    """FoldAmyloid CSV. APR residues marked ``Fold == 'f'``."""
    df = pd.read_csv(path, sep=r",|;", header=0, engine="python")
    df = df.apply(lambda col: col.replace(r"\s+", "", regex=True))
    df.columns = ["Number", "Residue", "Fold", "Score"]
    df["Fold"] = df["Fold"].astype(str).str.strip()
    return _finalise(df, path)


@register_adapter("pasta2")
def parse_pasta2(path: Path) -> pd.DataFrame:
    """PASTA 2.0 free-energy profile (one value per line, lower = more stable).

    The native file is a single column of energies; the residue index is the
    line number. No residue identity is provided, so ``Residue`` is left empty
    and resolved from the protein sequence downstream when needed.
    """
    df = pd.read_csv(path, sep="\t", names=["Score"])
    df = df.reset_index(names="Number")
    df["Number"] = df["Number"] + 1  # 0-based line index -> 1-based residue
    df["Residue"] = ""
    return _finalise(df[["Number", "Residue", "Score"]], path)


@register_adapter("waltz")
def parse_waltz(path: Path) -> pd.DataFrame:
    """Waltz per-residue scores; 0 outside position-specific matrix hits."""
    df = pd.read_csv(path, sep="\t", names=["Number", "Score"])
    df["Residue"] = ""
    return _finalise(df[["Number", "Residue", "Score"]], path)


@register_adapter("aggreprot")
def parse_aggreprot(path: Path) -> pd.DataFrame:
    """AggreProt per-residue CSV."""
    df = pd.read_csv(path, sep=",", skiprows=1, header=0)
    df = df.drop(
        columns=[
            c
            for c in ("struct_position", "sasa", "transmembrane")
            if c in df.columns
        ]
    )
    df.columns = ["Number", "Residue", "Score"]
    return _finalise(df, path)


@register_adapter("crossbeta")
def parse_crossbeta(path: Path) -> pd.DataFrame:
    """CrossBeta JSON. The first record's ``AA_list`` holds per-residue scores."""
    with Path(path).open("rb") as handle:
        data = json.load(handle)
    records = next(iter(data.items()))[1][0]
    df = pd.DataFrame(records["AA_list"])
    df = df.drop(columns=[c for c in ("score_list",) if c in df.columns])
    df.columns = ["Number", "Residue", "Score"]
    return _finalise(df, path)


@register_adapter("archcandy")
def parse_archcandy(path: Path) -> pd.DataFrame:
    """ArchCandy region-list CSV expanded to per-residue rows.

    The native output reports beta-arch hit intervals with a score per interval.
    Each interval is expanded so that every covered residue becomes a row; the
    ``present`` detection strategy then treats all emitted residues as APR. This
    expansion also fixes a latent defect in the original per-tool branch, which
    iterated over a DataFrame object (yielding column labels rather than rows).
    """
    raw = pd.read_csv(path, sep=",", header=0)
    rows: dict[str, list] = {"Number": [], "Residue": [], "Score": []}
    for _, row in raw.iterrows():
        start, stop = int(row.iloc[3]), int(row.iloc[4])
        residues = list(str(row.iloc[1]))
        coords = list(range(start, stop + 1))
        score = row.iloc[5]
        # pad/truncate residue list to coordinate span defensively
        if len(residues) < len(coords):
            residues += [""] * (len(coords) - len(residues))
        rows["Number"].extend(coords)
        rows["Residue"].extend(residues[: len(coords)])
        rows["Score"].extend([score] * len(coords))
    df = pd.DataFrame(rows).drop_duplicates(subset="Number").sort_values("Number")
    return _finalise(df, path)
