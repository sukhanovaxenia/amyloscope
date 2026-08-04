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

import ast
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
def parse_aggrescan(path: Path, sequence: str) -> pd.DataFrame:
    """Aggrescan per-residue CSV. APR rows carry a non-null ``Prediction``."""
    df = pd.read_csv(path, sep=r",", header=0, engine="python", encoding='latin-1')
    df.columns = ["Number", "Residue", "Score", "HSA", "NHSA", "a4vAHS", "Prediction"]
    return _finalise(df, path)


@register_adapter("appnn")
def parse_appnn(path: Path, sequence: str) -> pd.DataFrame:
    """APPNN TSV. Hotspot residues flagged by the boolean ``is_hotspot``."""
    df = pd.read_csv(path, sep="\t", header=0)
    df = df.drop(columns=[c for c in ("overall", "id") if c in df.columns])
    df.columns = ["Number", "Residue", "Score", "is_hotspot"]
    return _finalise(df, path)


@register_adapter("foldamyloid")
def parse_foldamyloid(path: Path, sequence: str) -> pd.DataFrame:
    """FoldAmyloid CSV. APR residues marked ``Fold == 'f'``."""
    df = pd.read_csv(path, sep=r"\t|,", header=0, engine="python", comment="-")
    df = df.apply(lambda col: col.replace(r"\s+", "", regex=True))
    df.columns = ["Number", "Residue", "Fold", "Score"]
    df["Fold"] = df["Fold"].astype(str).str.strip()
    return _finalise(df, path)


@register_adapter("pasta2")
def parse_pasta2(path: Path, sequence: str) -> pd.DataFrame:
    """PASTA 2.0 free-energy profile (one value per line, lower = more stable).

    The native file is a single column of energies; the residue index is the
    line number. No residue identity is provided, so ``Residue`` is left empty
    and resolved from the protein sequence downstream when needed.
    """
    df = pd.read_csv(path, sep="\t", names=["Score"])
    df = df.reset_index(names="Number")
    df["Number"] = df["Number"] + 1  # 0-based line index -> 1-based residue
    df["Residue"] = [aa for aa in sequence]
    return _finalise(df[["Number", "Residue", "Score"]], path)


@register_adapter("waltz")
def parse_waltz(path: Path, sequence: str) -> pd.DataFrame:
    """Waltz per-residue scores; 0 outside position-specific matrix hits."""
    df = pd.read_csv(path, sep="\t", names=["Number", "Score"])
    df["Residue"] = [aa for aa in sequence]
    return _finalise(df[["Number", "Residue", "Score"]], path)


@register_adapter("aggreprot")
def parse_aggreprot(path: Path, sequence: str) -> pd.DataFrame:
    """AggreProt per-residue CSV."""
    df = pd.read_csv(path, sep=",", header=0)
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
def parse_crossbeta(path: Path, sequence: str) -> pd.DataFrame:
    """CrossBeta JSON. The first record's ``AA_list`` holds per-residue scores."""
    with Path(path).open("rb") as handle:
        data = json.load(handle)
    records = next(iter(data.items()))[1][0]
    df = pd.DataFrame(records["AA_list"])
    df = df.drop(columns=[c for c in ("score_list",) if c in df.columns])
    df.columns = ["Number", "Residue", "Score"]
    return _finalise(df, path)


@register_adapter("archcandy")
def parse_archcandy(path: Path, sequence: str) -> pd.DataFrame:
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


# --------------------------------------------------------------------------- #
# Local-binary adapters
#
# The ArchCandy and CrossBeta public web servers are intermittently offline, so
# both are now commonly run from their standalone distributions. The local
# builds emit different layouts from the web outputs handled above, so they get
# their own adapters rather than overloading the web parsers. The web adapters
# are retained for historical files.
# --------------------------------------------------------------------------- #


def _parse_span(text: str) -> tuple[int, int]:
    """Parse a zero-padded ``start-stop`` ArchCandy position (e.g. ``02-23``)."""
    a, _, b = str(text).strip().partition("-")
    return int(a), int(b)


@register_adapter("archcandy_local")
def parse_archcandy_local(path: Path, sequence: str) -> pd.DataFrame:
    """Standalone-ArchCandy candidate list expanded to a per-residue track.

    The local build writes one row per predicted beta-arch candidate with
    columns ``Number, Diagram, Score, Arc_type, Position`` (the file carries a
    UTF-8 BOM, and ``Position`` is a zero-padded ``start-stop`` interval). Unlike
    the web output, every candidate carries a score, so this adapter preserves
    it: each candidate interval is expanded to per-residue rows and, where
    candidates overlap, the **maximum** covering score is kept — matching the
    "highest score above threshold" track in ArchCandy's own plots.

    Residue identities are left empty because the ``Diagram`` column encodes the
    arch geometry rather than a plain sequence; the loader reconstructs the
    sequence from a per-residue track or the configured sequence.

    Detection: use ``present`` to treat every reported candidate residue as an
    APR (web-server parity), or ``above`` on ``Score`` with ArchCandy's
    amyloidogenicity threshold of 0.560 (Ahmed et al., 2015, *Sci. Rep.*) to
    keep only high-confidence arches.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Score", "Position"}
    missing = required - set(raw.columns)
    if missing:
        raise AdapterError(
            f"{path}: ArchCandy local output missing columns {sorted(missing)} "
            f"(have {list(raw.columns)})"
        )
    per_residue: dict[int, float] = {}
    for _, row in raw.iterrows():
        try:
            start, stop = _parse_span(row["Position"])
        except (ValueError, AttributeError):
            continue  # skip a malformed position rather than abort the file
        score = float(row["Score"])
        for pos in range(start, stop + 1):
            per_residue[pos] = max(per_residue.get(pos, 0.0), score)
    if not per_residue:
        raise AdapterError(f"{path}: no parseable ArchCandy candidates")
    df = pd.DataFrame(
        {
            "Number": sorted(per_residue),
            "Residue": [""] * len(per_residue),
            "Score": [per_residue[p] for p in sorted(per_residue)],
        }
    )
    return _finalise(df, path)


@register_adapter("crossbeta_local")
def parse_crossbeta_local(path: Path, sequence: str) -> pd.DataFrame:
    """Standalone-CrossBeta output parsed to a scored, AR-flagged track.

    The local build writes a single ``;``-delimited record with columns
    ``Query_name, Sequence_length, Average_protein_prediction, AR_position,
    Amino_acids_score``. ``Amino_acids_score`` is a Python-literal list of
    single-key ``{residue: score}`` dicts in sequence order, and ``AR_position``
    is a Python-literal list of ``[start, stop]`` aggregation-region intervals
    that CrossBeta calls with its own internal (window-based) logic. Both fields
    use single quotes, so they are read with :func:`ast.literal_eval` rather than
    a JSON parser.

    The normalised track carries the raw per-residue ``Score`` plus a boolean
    ``in_AR`` column flagging residues inside CrossBeta's called regions.
    CrossBeta's region boundaries do not coincide with a fixed cut on the raw
    score (adjacent residues astride a boundary can both exceed 0.5), so the
    faithful APR signal is the tool's own call.

    Detection: use ``flag`` on ``in_AR`` to honour CrossBeta's called regions
    (recommended), or ``above`` on ``Score`` with a user-chosen threshold for a
    custom per-residue cut.
    """
    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    raw.columns = raw.columns.str.replace(",", "")
    needed = {"Amino_acids_score", "AR_position"}
    missing = needed - set(raw.columns)
    if missing:
        raise AdapterError(
            f"{path}: CrossBeta local output missing columns {sorted(missing)} "
            f"(have {list(raw.columns)})"
        )
    if raw.empty:
        raise AdapterError(f"{path}: CrossBeta local output has no data row")
    record = raw.iloc[0]
    try:
        residue_scores = ast.literal_eval(record["Amino_acids_score"])
        ar_intervals = ast.literal_eval(record["AR_position"])
    except (ValueError, SyntaxError) as exc:
        raise AdapterError(
            f"{path}: could not parse CrossBeta literal fields: {exc}"
        ) from exc

    ar_residues: set[int] = set()
    for interval in ar_intervals:
        start, stop = int(interval[0]), int(interval[1])
        ar_residues.update(range(start, stop + 1))

    numbers, residues, scores, in_ar = [], [], [], []
    for idx, entry in enumerate(residue_scores, start=1):
        (residue, score), = entry.items()  # each entry is a single-key dict
        numbers.append(idx)
        residues.append(residue)
        scores.append(float(score))
        in_ar.append(idx in ar_residues)

    df = pd.DataFrame(
        {"Number": numbers, "Residue": residues, "Score": scores, "in_AR": in_ar}
    )
    return _finalise(df, path)

@register_adapter("tango")
def parse_tango(path: Path, sequence: str) -> pd.DataFrame:
    """TANGO per-residue output (https://tango.crg.es/products#tango).
 
    TANGO reports, per residue, the percentage population of each conformational
    state — ``Beta`` (intramolecular beta), ``Turn``, ``Helix`` — plus the
    cross-beta ``Aggregation`` propensity and its concentration-stabilised
    variant (``Conc-Stab_Aggregation``). The file is tab-separated with
    zero-padded residue indices and space-padded fields.
 
    The aggregation-relevant signal is the ``Aggregation`` column, which becomes
    ``Score``; the concentration-stabilised aggregation and the beta-structure
    column are preserved as ``Aggregation_conc`` and ``Beta`` for alternative
    detection targets.
 
    Detection: ``above`` on ``Score`` with a 5% threshold. Combined with the
    default ``min_region_length`` of 5, this reproduces TANGO's standard
    aggregation-nucleating-region criterion — at least five consecutive residues
    each scoring above 5% (Fernandez-Escamilla et al., 2004, *Nat. Biotechnol.*).
    """
    raw = pd.read_csv(path, sep="\t", skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    required = {"res", "aa", "Aggregation"}
    missing = required - set(raw.columns)
    if missing:
        raise AdapterError(
            f"{path}: TANGO output missing columns {sorted(missing)} "
            f"(have {list(raw.columns)})"
        )
    out = pd.DataFrame(
        {
            "Number": raw["res"],
            "Residue": raw["aa"],
            "Score": pd.to_numeric(raw["Aggregation"], errors="coerce"),
        }
    )
    if "Conc-Stab_Aggregation" in raw.columns:
        out["Aggregation_conc"] = pd.to_numeric(
            raw["Conc-Stab_Aggregation"], errors="coerce"
        )
    if "Beta" in raw.columns:
        out["Beta"] = pd.to_numeric(raw["Beta"], errors="coerce")
    return _finalise(out, path)