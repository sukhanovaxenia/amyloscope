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
import io
import json
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd

#: adapter key -> parser callable(path) -> normalised DataFrame
_REGISTRY: dict[str, Callable[[Path], pd.DataFrame]] = {}

REQUIRED_COLUMNS = ("Number", "Residue", "Score")
SCORE_COLUMNS_AMY_PRED = ("6aa_Avg_Score", "10aa_Avg_Score", "15aa_Avg_Score", "Avg_Score")


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
# Shared readers
#
# Predictor exports are not a stable format. The same web server, saved on
# different days or through a spreadsheet, yields comma- and semicolon-separated
# files, a UTF-8 BOM or none, CRLF or LF, and column headers that differ between
# releases (Aggrescan writes `a4v` in one export and `Score` in another,
# `Prediction` in one and `Disorder` in another). Positional column assignment
# hides all of that: a semicolon file parses as a single column and then dies on
# the rename, and the loader records it as a load failure, so the tool silently
# leaves the panel while the consensus denominator still counts it.
#
# Resolving columns by NAME, with explicit alias sets, converts a format change
# from a silent dropout into either a correct parse or a named error.
# --------------------------------------------------------------------------- #

_DELIMITERS = (",", ";", "\t", "|")


def _read_table(path: Path, **read_kwargs) -> pd.DataFrame:
    """Read a delimited predictor export, sniffing separator and encoding."""
    path = Path(path)
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 decodes any byte string
        raise AdapterError(f"{path}: could not decode as UTF-8 or Latin-1")

    header = next(
        (
            ln
            for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ),
        "",
    )
    delimiter = max(_DELIMITERS, key=header.count)
    if header.count(delimiter) == 0:
        raise AdapterError(
            f"{path}: header line has no recognised delimiter "
            f"(tried {_DELIMITERS!r}): {header[:120]!r}"
        )
    read_kwargs.setdefault("comment", "#")
    df = pd.read_csv(io.StringIO(text), sep=delimiter, **read_kwargs)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def _pick(df: pd.DataFrame, aliases: tuple[str, ...], *, source: Path, what: str) -> str:
    """Resolve one logical column from a set of case-insensitive aliases."""
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    raise AdapterError(
        f"{source}: no {what} column found (looked for {list(aliases)}); "
        f"file has {list(df.columns)}"
    )


def _optional(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _residues_for(sequence: str | None, n: int, *, source: Path, tool: str) -> list[str]:
    """Residue identities for a track whose native format omits them.

    PASTA 2.0 writes a bare column of pairing free energies and WALTZ a
    position/score pair; neither carries the amino acid, so identity has to come
    from the configured sequence. When no sequence is configured the honest
    answer is an empty identity, not a crash: the consensus is positional and
    does not need the letter, and the loader reconstructs identities from any
    other track that does carry them. Previously this raised TypeError inside a
    list comprehension, which surfaced as 'NoneType object is not iterable' with
    no mention of the protein or the tool.
    """
    if not sequence:
        return [""] * n
    if len(sequence) != n:
        raise AdapterError(
            f"{source}: {tool} reports {n} positions but the configured "
            f"sequence has {len(sequence)} residues"
        )
    return list(sequence)


def _clean_flag(series: pd.Series) -> pd.Series:
    """Coerce a tool's hot-spot marker column to a strict boolean.

    Aggrescan marks hot-spot residues with ``1`` and leaves the field empty
    otherwise, but its exports also carry stray non-breaking spaces (0xCA in the
    Mac Roman files) in that field — on residue 1 of RPL27, for instance. Under a
    ``notnull`` detection rule that whitespace is indistinguishable from a
    prediction, and the protein acquires a phantom single-residue APR at its
    N-terminus. Anything that is not a positive number or an explicit truth word
    is therefore treated as absent.
    """
    text = series.astype(str).str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    truthy = text.str.lower().isin({"true", "yes", "y", "f", "hs", "hotspot"})
    return (numeric.fillna(0) > 0) | truthy


# --------------------------------------------------------------------------- #
# Adapters for the eight predictors carried over from the ribosomal pipeline
# --------------------------------------------------------------------------- #


@register_adapter("aggrescan")
def parse_aggrescan(path: Path, sequence: str) -> pd.DataFrame:
    """AGGRESCAN per-residue export → normalised track.

    AGGRESCAN (Conchillo-Sole et al., 2007, *BMC Bioinformatics*) scores each
    residue by ``a4v``: the intrinsic aggregation-propensity value ``a3v``
    averaged over a sequence-length-dependent window, so the score already
    encodes local context rather than single-residue hydrophobicity. Hot Spots
    are the contiguous stretches AGGRESCAN itself calls; the tool marks their
    residues and leaves every other residue blank.

    Emitted columns: ``Number, Residue, Score`` (= a4v) plus, where present,
    ``HSA``, ``NHSA``, ``a4vAHS`` and a strict boolean ``is_hotspot``.

    ``is_hotspot`` comes from the tool's own marker column when the file has
    one, and otherwise from ``NHSA > 0``, which is the same call: NHSA
    (Normalized Hot Spot Area) is zero outside a Hot Spot and positive inside
    one. ``HSA`` is NOT interchangeable with it — AGGRESCAN assigns shared
    area before applying its run-length requirement, so an isolated residue
    can carry ``HSA > 0`` while ``NHSA = 0`` (Aβ42 Y10 does exactly that).

    Detection: use ``flag`` on ``is_hotspot`` to honour AGGRESCAN's own Hot Spot
    calls. ``notnull`` on ``Prediction`` is NOT equivalent — see
    :func:`_clean_flag` — and ``above`` on ``Score`` re-derives regions with a
    cutoff AGGRESCAN did not calibrate, which on a 334-residue protein
    invented a whole Hot Spot and widened three others.
    """
    df = _read_table(path)
    number = _pick(df, ("Number", "Position", "Num", "Res_number"), source=path, what="residue index")
    residue = _pick(df, ("AA", "Residue", "Res", "amino_acid"), source=path, what="residue identity")
    score = _pick(df, ("a4v", "Score", "a3v", "a4vAHS"), source=path, what="a4v score")
    out = pd.DataFrame(
        {"Number": df[number], "Residue": df[residue], "Score": df[score]}
    )
    for logical, aliases in (
        ("HSA", ("HSA",)),
        ("NHSA", ("NHSA",)),
        ("a4vAHS", ("a4vAHS",)),
    ):
        col = _optional(df, aliases)
        if col is not None and col != score:
            out[logical] = pd.to_numeric(df[col], errors="coerce")
    flag_col = _optional(df, ("Prediction", "Disorder", "HotSpot", "Hot_Spot", "HS"))
    if flag_col is not None:
        out["is_hotspot"] = _clean_flag(df[flag_col])
    elif "NHSA" in out:
        # Hand-scraped exports often carry the six data columns and no marker.
        # NHSA is that marker, numerically: recovering it here keeps those files
        # usable instead of letting them arrive with no Hot Spots at all.
        out["is_hotspot"] = out["NHSA"] > 0
    if "is_hotspot" in out:
        # Retained so configs written against the original adapter still resolve,
        # but normalised: the raw column's blanks and stray whitespace are gone.
        out["Prediction"] = out["is_hotspot"].map({True: 1, False: pd.NA})
    return _finalise(out, path)


@register_adapter("appnn")
def parse_appnn(path: Path, sequence: str) -> pd.DataFrame:
    """APPNN TSV. Hotspot residues flagged by the boolean ``is_hotspot``."""
    df = pd.read_csv(path, sep="\t", header=0)
    df = df.drop(columns=[c for c in ("overall", "id") if c in df.columns])
    df.columns = ["Number", "Residue", "Score", "is_hotspot"]
    return _finalise(df, path)


@register_adapter("foldamyloid")
def parse_foldamyloid(path: Path, sequence: str) -> pd.DataFrame:
    """FoldAmyloid per-residue export → normalised track.

    FoldAmyloid (Garbuzynskiy, Lobanov & Galzitskaya, 2010, *Bioinformatics*)
    scores a residue by its *expected number of contacts* within 8 A, averaged
    over a 5-residue frame. The mechanistic claim is packing-based rather than
    hydrophobicity-based: amyloidogenic segments are those predicted to be
    densely contacting, because a cross-beta spine requires tight side-chain
    interdigitation (a steric zipper). Residues whose averaged profile exceeds
    the calibrated 21.4 threshold are flagged ``f`` in the ``Fold`` column.

    Emitted columns: ``Number, Residue, Score`` (= expected contacts), ``Fold``
    and a boolean ``is_amyloidogenic``.

    The native export carries a UTF-8 BOM, CRLF line endings and the headers
    ``Num, Res, Fold, Value``. The previous implementation read it with
    ``comment="-"``, which truncates any line at the first hyphen: harmless while
    the profile stays positive, but it would silently drop the tail of every row
    of a file that used ``-`` as its negative marker, as some FoldAmyloid builds
    do.
    """
    df = _read_table(path)
    number = _pick(df, ("Num", "Number", "Position", "N"), source=path, what="residue index")
    residue = _pick(df, ("Res", "Residue", "AA"), source=path, what="residue identity")
    score = _pick(df, ("Value", "Score", "Contacts"), source=path, what="profile value")
    fold = _optional(df, ("Fold", "Flag", "Amyloid"))
    out = pd.DataFrame(
        {"Number": df[number], "Residue": df[residue], "Score": df[score]}
    )
    if fold is not None:
        marker = df[fold].fillna("").astype(str).str.strip().str.lower()
        out["Fold"] = marker
        out["is_amyloidogenic"] = marker.isin({"f", "1", "true", "yes"})
    return _finalise(out, path)


@register_adapter("pasta2")
def parse_pasta2(path: Path, sequence: str) -> pd.DataFrame:
    """PASTA 2.0 free-energy profile (one value per line, lower = more stable).

    The native file is a single column of energies; the residue index is the
    line number. No residue identity is provided, so ``Residue`` is left empty
    and resolved from the protein sequence downstream when needed.
    """
    df = pd.read_csv(path, sep="\t", names=["Score"], comment="#")
    df = df.reset_index(names="Number")
    df["Number"] = df["Number"] + 1  # 0-based line index -> 1-based residue
    df["Residue"] = _residues_for(sequence, len(df), source=Path(path), tool="PASTA 2.0")
    return _finalise(df[["Number", "Residue", "Score"]], path)


@register_adapter("waltz")
def parse_waltz(path: Path, sequence: str) -> pd.DataFrame:
    """Waltz per-residue scores; 0 outside position-specific matrix hits."""
    df = pd.read_csv(path, sep="\t", names=["Number", "Score"], comment="#")
    df["Residue"] = _residues_for(sequence, len(df), source=Path(path), tool="WALTZ")
    return _finalise(df[["Number", "Residue", "Score"]], path)


@register_adapter("aggreprot")
def parse_aggreprot(path: Path, sequence: str) -> pd.DataFrame:
    """AggreProt per-residue CSV → normalised track.

    AggreProt is a deep-network ensemble predicting aggregation-prone regions
    from sequence alone, and it reports two structural covariates alongside the
    propensity: relative solvent accessibility and a transmembrane flag. Both
    are kept rather than dropped, because they bear directly on whether a
    predicted APR is available to pair. Burial is the masking variable the
    ribosome-protection hypothesis turns on, and TM segments are a known
    false-positive source for any hydrophobicity-weighted model.

    Column *order* differs between exports -- the live service writes
    ``position, struct_position, amino_acid, aggregation, sasa, transmembrane``
    while older tables write ``position, residue, score, ...`` -- so columns are
    resolved by name. The previous implementation dropped three columns and
    renamed whatever was left by position, which produced a correct track only
    as long as the surviving three happened to stay in that order.
    """
    df = _read_table(path)
    number = _pick(df, ("position", "Number", "residue_number"), source=path, what="residue index")
    residue = _pick(df, ("amino_acid", "residue", "AA"), source=path, what="residue identity")
    score = _pick(df, ("aggregation", "score", "Score"), source=path, what="aggregation score")
    out = pd.DataFrame(
        {"Number": df[number], "Residue": df[residue], "Score": df[score]}
    )
    for logical, aliases in (
        ("sasa", ("sasa",)),
        ("transmembrane", ("transmembrane",)),
        ("struct_position", ("struct_position",)),
    ):
        col = _optional(df, aliases)
        if col is not None:
            out[logical] = df[col]
    return _finalise(out, path)


@register_adapter("crossbeta")
def parse_crossbeta(path: Path, sequence: str) -> pd.DataFrame:
    """CrossBeta JSON. The first record's ``AA_list`` holds per-residue scores."""
    with Path(path).open("rb") as handle:
        data = json.load(handle)
    # The service returns either a bare list of records or a dict keyed by query
    # name; which one depends on whether the job carried one sequence or several.
    # The previous implementation indexed data[0] unconditionally and raised
    # KeyError: 0 on the dict form, so a multi-query export could not be read at
    # all while the single-query form worked.
    if isinstance(data, dict):
        if not data:
            raise AdapterError(f"{path}: Cross-Beta JSON is empty")
        records = next(iter(data.values()))
    else:
        records = data
    if isinstance(records, list):
        if not records:
            raise AdapterError(f"{path}: Cross-Beta record list is empty")
        records = records[0]
    if "AA_list" not in records:
        raise AdapterError(
            f"{path}: Cross-Beta record has no 'AA_list' (keys: {sorted(records)})"
        )
    df = pd.DataFrame(records["AA_list"])
    df = df.drop(columns=[c for c in ("score_list",) if c in df.columns])
    number = _pick(
        df, ("Number", "position", "Num", "index"), source=path, what="residue index"
    )
    residue = _pick(df, ("Residue", "amino_acid", "AA"), source=path, what="residue identity")
    score = _pick(
        df,
        ("Score", "mean_confidence", "score", "prediction"),
        source=path,
        what="confidence",
    )
    positions = pd.to_numeric(df[number], errors="coerce")
    # The 2.0 API's `index` is 0-BASED; the older export's `Number` is 1-based.
    # Detected rather than configured: a silent off-by-one here shifts every
    # Cross-Beta call by one residue relative to the rest of the panel, which no
    # downstream check could catch.
    if str(number).lower() == "index" or positions.min() == 0:
        positions = positions + 1
    out = pd.DataFrame(
        {"Number": positions, "Residue": df[residue], "Score": df[score]}
    )
    # AR_list is Cross-Beta's own aggregation-region call, 1-based inclusive.
    # It embeds the model's 15-residue window, so it is not reproducible by
    # thresholding the smoothed confidence.
    ar = records.get("AR_list") if isinstance(records, dict) else None
    if ar:
        flags = [False] * len(out)
        for interval in ar:
            for pos in range(int(interval[0]), int(interval[1]) + 1):
                if 1 <= pos <= len(flags):
                    flags[pos - 1] = True
        out["in_AR"] = flags
    return _finalise(out, path)


#: ArchCandy threshold reference points. These are NOT interchangeable.
#:   0.40 — the ArchCandy 2.0 web default (a recall-oriented submission filter)
#:   0.57 — ArchCandy 2.0's significance boundary; below it, 0.40-0.57 is
#:          documented as "ambiguous" and under 0.40 as "non-significant"
#:   0.56 — the published threshold for ArchCandy 1.0 (Ahmed et al., 2015,
#:          Alzheimers Dement. 11:681), which is the only downloadable build
ARCHCANDY_WEB_DEFAULT = 0.40
ARCHCANDY_2_SIGNIFICANT = 0.57
ARCHCANDY_1_PUBLISHED = 0.56


def _archcandy_spans(raw: pd.DataFrame, source: Path) -> list[tuple[int, int, float]]:
    """``(start, stop, score)`` from either ArchCandy layout.

    The web ``Table`` view and the standalone build report the same thing — one
    row per beta-arch candidate with a score and a span — in two column
    vocabularies:

        local 1.0 : Number, Digram, Score, Arc_type, Position   (Position = "02-23")
        web   2.0 : ID, Sequence, Arch, Start, Stop, Score

    Resolving both here means a file cannot be mis-parsed by choosing the wrong
    adapter name, which was previously a silent failure: the old web parser read
    columns by POSITION (``row.iloc[3]``, ``iloc[4]``, ``iloc[5]``) and would
    happily return nonsense for the other layout.
    """
    score_col = _pick(raw, ("Score",), source=source, what="arch score")
    start_col = _optional(raw, ("Start",))
    stop_col = _optional(raw, ("Stop", "End"))
    position_col = _optional(raw, ("Position", "Span", "Range"))

    spans: list[tuple[int, int, float]] = []
    for _, row in raw.iterrows():
        try:
            score = float(row[score_col])
        except (TypeError, ValueError):
            continue
        if start_col and stop_col:
            try:
                start, stop = int(row[start_col]), int(row[stop_col])
            except (TypeError, ValueError):
                continue
        elif position_col:
            text = str(row[position_col]).strip()
            head, _, tail = text.partition("-")
            try:
                start, stop = int(head), int(tail)
            except ValueError:
                # Excel turns "24-01" into a date. Those rows are unrecoverable
                # from the file alone, so they are skipped loudly rather than
                # guessed at -- regenerate the export instead of repairing it.
                warnings.warn(
                    f"{source}: unparseable ArchCandy position {text!r} "
                    f"(an Excel round-trip mangles spans into dates); row skipped",
                    stacklevel=3,
                )
                continue
        else:
            raise AdapterError(
                f"{source}: ArchCandy output has neither Start/Stop nor Position; "
                f"columns are {list(raw.columns)}"
            )
        if stop < start:
            start, stop = stop, start
        spans.append((start, stop, score))
    if not spans:
        raise AdapterError(f"{source}: no parseable ArchCandy candidates")
    return spans


def _archcandy_track(
    path: Path, sequence: str, *, score_mode: str
) -> pd.DataFrame:
    if score_mode not in ("highest", "cumulative"):
        raise AdapterError("score_mode must be 'highest' or 'cumulative'")
    raw = _read_table(path)
    spans = _archcandy_spans(raw, Path(path))
    length = len(sequence) if sequence else max(stop for _, stop, _ in spans)
    scores = [0.0] * length
    counts = [0] * length
    for start, stop, score in spans:
        for pos in range(max(1, start), min(length, stop) + 1):
            idx = pos - 1
            scores[idx] = max(scores[idx], score) if score_mode == "highest" else scores[idx] + score
            counts[idx] += 1
    out = pd.DataFrame(
        {
            "Number": range(1, length + 1),
            "Residue": list(sequence) if sequence else [""] * length,
            "Score": scores,
            "arch_count": counts,
        }
    )
    # Residues covered by no arch are not predictions of zero amyloidogenicity;
    # they are absent from the tool's output. Keeping them at 0.0 matches the
    # `present`/`above` detection strategies, but arch_count distinguishes them.
    return _finalise(out, path)


@register_adapter("archcandy")
@register_adapter("archcandy2")
def parse_archcandy(
    path: Path,
    sequence: str,
    *,
    score_mode: str = "highest",
) -> pd.DataFrame:
    """ArchCandy region list (either layout) expanded to a per-residue track.

    ArchCandy predicts **beta-arches** — beta-strand/loop/beta-strand motifs
    that stack in parallel and in register into a beta-arcade, the structural
    core of most naturally occurring and disease-related amyloid fibrils
    (Kajava et al., 2010, *FASEB J.* 24:1311). It reports scored *segments*, so
    the per-residue track is a projection and the projection rule matters:

    ``highest`` (default)
        A residue takes the best single arch covering it, keeping scores on
        ArchCandy's own [0, 1] scale and comparable to its published bands.
    ``cumulative``
        Overlapping arches are summed, reproducing the web UI's cumulative tab.
        ArchCandy emits many overlapping candidates — fourteen on Abeta42 — so a
        residue can reach 6.681 where the best single arch is 0.723. No
        documented threshold applies to such a value.

    **Thresholds are version-specific and must not be mixed.** 0.40 is the
    ArchCandy 2.0 *web default*; 0.57 is 2.0's significance boundary (0.40-0.57
    is documented as ambiguous, below 0.40 as non-significant); 0.56 is the
    published threshold for ArchCandy 1.0, which is the only downloadable build.
    A detection rule set below 0.40 selects predictions the tool's own benchmark
    calls non-significant.

    ``arch_count`` carries how many arches cover each residue, which separates a
    lone high-confidence arch from a pile of weak overlapping ones — a
    distinction ``highest`` deliberately drops from the score.
    """
    return _archcandy_track(path, sequence, score_mode=score_mode)


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
def parse_archcandy_local(
    path: Path,
    sequence: str,
    *,
    score_mode: str = "highest",
) -> pd.DataFrame:
    """Standalone ArchCandy 1.0 output — same content, different column names.

    Kept as a separate registered name so existing configs keep working, but it
    is the same parser: ``Number, Digram, Score, Arc_type, Position`` and the
    web ``ID, Sequence, Arch, Start, Stop, Score`` are one table in two
    vocabularies, and resolving both in one place removes the possibility of
    choosing the wrong adapter for a file.

    Note the threshold difference when configuring detection: the published
    cutoff for the 1.0 build is 0.56 (Ahmed et al., 2015), not the 0.40 web
    default and not 2.0's 0.57 significance boundary.
    """
    return _archcandy_track(path, sequence, score_mode=score_mode)


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

@register_adapter("archcandy2")
def parse_archcandy2(
    path: Path,
    sequence: str,
    *,
    score_mode: str = "highest",
) -> pd.DataFrame:
    """ArchCandy 2.0 web CSV (``ID,Sequence,Arch,Start,Stop,Score``) -> track.

    ArchCandy predicts beta-arches -- beta-strand/loop/beta-strand motifs that
    stack in parallel and in register into a beta-arcade, the structural core of
    most amyloid fibrils (Kajava et al., 2010, *FASEB J.* 24:1311). It reports
    scored SEGMENTS with a topology string, so the per-residue track is a
    projection, and which projection is used changes the numbers:

    ``highest`` (default)
        A residue takes the best single arch covering it. Scores stay on
        ArchCandy's own [0, 1] scale and remain comparable to its published
        calibration.
    ``cumulative``
        Overlapping arches are summed, reproducing the web UI's cumulative tab.
        ArchCandy emits many overlapping candidates -- 14 on Abeta42 alone --
        so a residue reaches 6.681 where the best single arch is 0.723. Such a
        value is no longer an ArchCandy score and no documented threshold
        applies to it.

    **Threshold calibration, from ArchCandy 2.0's own documentation:** a
    prediction is non-significant below 0.40, ambiguous from 0.40 to 0.57, and
    significant above 0.57; 0.40 is the web default. A ``detection`` rule set
    below 0.40 is therefore selecting predictions the authors' benchmark treats
    as non-significant -- which is a statement about the evidence, not a milder
    version of the same claim.

    Version 2.0's layout is NOT the standalone 1.0 layout
    (``Number,Digram,Score,Arc_type,Position``); use ``archcandy_local`` for that.

    Detection: ``above`` on ``Score`` at 0.57 (significant) or 0.40 (web
    default). The auxiliary ``arch_count`` column carries how many arches cover
    each residue, which distinguishes a lone high-confidence arch from a
    consensus of weak overlapping ones -- a difference ``highest`` deliberately
    discards from the score.
    """
    if score_mode not in ("highest", "cumulative"):
        raise AdapterError("score_mode must be 'highest' or 'cumulative'")
    raw = _read_table(path)
    for column in ("Start", "Stop", "Score"):
        if column not in raw.columns:
            raise AdapterError(
                f"{path}: ArchCandy 2.0 CSV missing {column!r}; has "
                f"{list(raw.columns)}. A file with 'Number,Digram,Score,"
                f"Arc_type,Position' is the standalone 1.0 layout -- use the "
                f"'archcandy_local' adapter for it."
            )
    length = len(sequence) if sequence else int(raw["Stop"].max())
    scores = [0.0] * length
    counts = [0] * length
    for _, row in raw.iterrows():
        try:
            start, stop, score = int(row["Start"]), int(row["Stop"]), float(row["Score"])
        except (TypeError, ValueError):
            continue
        for pos in range(max(1, start), min(length, stop) + 1):
            idx = pos - 1
            scores[idx] = max(scores[idx], score) if score_mode == "highest" else scores[idx] + score
            counts[idx] += 1
    out = pd.DataFrame(
        {
            "Number": range(1, length + 1),
            "Residue": list(sequence) if sequence else [""] * length,
            "Score": scores,
            "arch_count": counts,
        }
    )
    return _finalise(out, path)


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


@register_adapter("amyloid_predict")
def parse_amyloid_predict(
    path: Path,
    sequence: str,
    *,
    score_column: str = "Avg_Score",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """amyloid_predict multi-probe export → normalised track.

    The tool reports the same residue averaged over 6-, 10- and 15-residue
    probes plus its own ``Avg_Score`` summary. These are not interchangeable:
    the 15-aa probe smears an APR boundary by up to seven residues on each side
    and inflates the termini, while the 6-aa probe localises sharply. Which one
    becomes ``Score`` is therefore a modelling choice and is set per tool entry
    via ``options: {score_column: ...}``, with the remaining probe widths kept
    as auxiliary columns so a boundary-sensitivity check needs no re-parse.

    The default is the tool's own ``Avg_Score``, matching the detection column
    the shipped config already thresholds. Previously the adapter defaulted to
    ``6aa_Avg_Score`` while the config's detection rule read ``Avg_Score``, so
    the track that was plotted and the track that called APRs were different
    quantities.
    """
    df = _read_table(path)
    if score_column not in df.columns:
        raise AdapterError(
            f"{path}: score_column {score_column!r} not present; "
            f"available: {[c for c in df.columns]}"
        )
    if len(df) != len(sequence):
        raise AdapterError(
            f"{path}: {len(df)} residues in score file, sequence has {len(sequence)}"
        )
    index = _pick(df, ("Residue", "Number", "Position"), source=path, what="residue index")
    scores = pd.to_numeric(df[score_column], errors="coerce")
    out = pd.DataFrame(
        {
            "Number": pd.to_numeric(df[index], errors="coerce").astype("Int64"),
            "Residue": list(sequence),
            "Score": scores,
            "APR": (scores >= threshold).astype(int),
        }
    )
    for col in SCORE_COLUMNS_AMY_PRED:  # keep the other probe-length views
        if col in df.columns and col != score_column:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    return _finalise(out, path)


@register_adapter("amylodeep")
def parse_amylodeep(
    path: Path,
    sequence: str,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """AmyloDeep window probabilities projected onto residues.

    AmyloDeep (Davtyan et al., 2025, bioRxiv 2025.09.16.676495) is an ensemble of
    five heads over protein-language-model embeddings -- ESM2-150M fine-tuned,
    ESM2-650M with a trained classifier, UniRep with a trained classifier, and an
    SVM and an XGBoost over mean-pooled ESM2-650M features -- averaged after
    per-head calibration. That makes it the only member of this panel whose
    features are learned rather than biophysical, which is the reason to carry it:
    a consensus of nine mechanistically related models and one unrelated one tells
    you something a consensus of ten related models cannot.

    It is also the reason to keep it labelled. Its agreement with the biophysical
    panel is a finding, not a vote to be absorbed silently, so it is registered as
    opt-in and its breadth should be read in ``analysis/clusters.py`` alongside
    Cross-Beta's before it is allowed to move a tier.

    **The model scores WINDOWS, not residues.** Three table shapes reach this
    adapter and only two of them may pass. AmyloDeep 0.4 writes a residue table by
    default (``residue_number`` 1-based, with the window statistics it was
    projected from alongside) and a window table under ``--resolution window``
    (``window_start_0based``, as 0.3 emitted); ``aggressor-wrappers`` writes a
    third, its own ``Number, Residue, Score`` projection. The two residue-grain
    tables load; a window table is refused, identified by its marker column where
    the header carries one and by its row count otherwise.

    That projection is the same problem AmyloGram has, and the same caution
    applies with a wider window: under ``max`` one positive 10-mer paints all ten
    of its residues, so a boundary can exceed the evidence by up to *w*-1 = 9
    residues -- nearly twice AmyloGram's 5, and in the same family as
    Cross-Beta's 15-residue artefact. Read ``window_size`` and ``aggregate`` from
    the file rather than assuming either: AmyloDeep 0.4 defaults to 6 in both the
    CLI and ``predict_ensemble_rolling``, 0.3's CLI defaulted to 10, and a table's
    provenance is what decides how far its shoulders can be trusted.

    Emitted columns: ``Number, Residue, Score`` (= ensemble probability), ``APR``,
    plus whichever of ``window_size``, ``aggregate``, ``coverage_depth``,
    ``window_min/mean/max/support/std``, ``window_at_start`` and ``heads_used``
    the source table carries. The spread columns are worth keeping even though the
    consensus reads only ``Score``: a residue at 0.35 whose covering windows ran
    0.05 to 0.95 is a boundary residue, and ``Score`` alone cannot distinguish it
    from a residue every window agreed was mid-range.

    Detection: ``above`` on ``Score``. The ensemble is calibrated per head before
    averaging, so 0.5 is a meaningful operating point rather than an arbitrary
    cut -- but it is calibrated on the paper's own 2 366-sequence set, so it
    should still be checked against the validation set this panel is tuned on.
    """
    df = _read_table(path)
    lowered = {str(c).strip().lower() for c in df.columns}

    # Refuse a WINDOW table before looking for a residue index, so the error names
    # the actual problem. AmyloDeep 0.4 labels its two grains: a window table
    # carries window_index/window_start_0based and a residue table carries
    # residue_number, so the grain is usually readable off the header. The row
    # count stays as the fallback for upstream tables that carry neither, and it
    # is the only signal available for those -- a window table cannot have one row
    # per residue, but a projected table must.
    window_grain = {"window_index", "window_start_0based"} & lowered
    residue_grain = {"residue_number", "number"} & lowered
    if (window_grain and not residue_grain) or not (
        bool(sequence) and len(df) == len(sequence)
    ):
        detail = (
            f"its {sorted(window_grain)} column marks it as one"
            if window_grain
            else f"{len(df)} rows for a {len(sequence)}-residue sequence"
        )
        raise AdapterError(
            f"{path}: {detail}, so this is a WINDOW table, not a per-residue one. "
            f"AmyloDeep window scores must be projected onto residues before they "
            f"enter a consensus -- run amylodeep with --resolution residue (or the "
            f"default), or put the table through aggressor-wrappers' amylodeep "
            f"parser, either of which records the window size and the aggregation "
            f"rule it used. Assigning each window's probability to its start "
            f"residue alone would mislocate every region by about half a window."
        )

    number = _pick(
        df,
        ("Number", "residue_number", "Position", "position", "Num"),
        source=path,
        what="residue index",
    )
    score = _pick(
        df,
        ("Score", "probability", "Probability", "amylodeep_score"),
        source=path,
        what="probability",
    )
    residue = _optional(df, ("Residue", "residue", "AA", "Res"))
    values = pd.to_numeric(df[score], errors="coerce")

    positions = pd.to_numeric(df[number], errors="coerce")
    # An upstream projected table and AmyloDeep's own residue_number are both
    # 1-based; only a position column carried over from a window axis starts at 0.
    # The first position supplies the base, nothing more.
    base = int(positions.min()) if positions.notna().any() else 1

    out = pd.DataFrame(
        {
            "Number": positions.astype(int) + (1 if base == 0 else 0),
            "Residue": df[residue] if residue else list(sequence)[: len(df)],
            "Score": values,
            "APR": (values >= threshold).astype(int),
        }
    )
    # Provenance and window-spread columns, where the source table carries them.
    # window_size and aggregate decide how far a boundary can be trusted;
    # coverage_depth marks the termini, where a value rests on fewer windows; and
    # window_min/window_max separate a genuine mid-range residue from one sitting
    # on the boundary between a silent and a hot window, which the aggregated
    # Score alone cannot distinguish. heads_used flags a run that dropped the
    # XGBoost head and averaged four.
    for col in (
        "window_size",
        "coverage_depth",
        "window_min",
        "window_mean",
        "window_max",
        "window_support",
        "window_std",
        "window_at_start",
        "heads_used",
    ):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
    if "aggregate" in df.columns:
        out["aggregate"] = df["aggregate"].astype(str)
    return _finalise(out, path)


@register_adapter("amylogram")
def parse_amylogram(
    path: Path,
    sequence: str,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """AmyloGram hexapeptide probabilities projected onto residues.

    AmyloGram (Burdukiewicz et al., 2017, *PeerJ*) is an n-gram model over a
    reduced amino-acid alphabet, trained on the **hexapeptides** of AmyLoad. It
    therefore returns one probability per query sequence, not per residue, and a
    whole-protein query is outside the regime it was fitted for. The projection
    onto residues — score every overlapping 6-mer, then aggregate the windows
    covering each position — happens upstream in ``aggressor-wrappers``, which
    writes the ``Number, Residue, Score`` table this adapter reads.

    Keeping the projection upstream and the table on disk is deliberate: the
    aggregation rule materially changes APR extent. Under ``max`` a single
    positive hexamer paints all six of its residues, so boundaries are broader
    than the evidence by up to *w*-1 = 5 residues; under ``mean`` they contract
    onto the peptide core. On Abeta42 the two rules give 14-25 and 17-22
    respectively, and only the second isolates the KLVFFA nucleating segment.
    The file on disk records which rule produced the numbers the consensus
    counted.

    Detection: ``above`` on ``Score``. The model emits a probability, so 0.5 is
    the natural cut, but it is the operating point of a *hexapeptide*
    classifier and should be chosen against the validation set the panel is
    tuned on.
    """
    df = _read_table(path)
    number = _pick(df, ("Number", "Position", "Num"), source=path, what="residue index")
    score = _pick(
        df, ("Score", "Probability", "amylogram_score"), source=path, what="probability"
    )
    residue = _optional(df, ("Residue", "AA", "Res"))
    values = pd.to_numeric(df[score], errors="coerce")
    out = pd.DataFrame(
        {
            "Number": df[number],
            "Residue": df[residue] if residue else list(sequence)[: len(df)],
            "Score": values,
            "APR": (values >= threshold).astype(int),
        }
    )
    # Per-width projections (w6_Score, w8_Score, ...) and the coverage depth are
    # carried through as auxiliary tracks. Depth matters for reading the figure:
    # a terminal residue is covered by fewer windows than an interior one, so
    # every aggregation rule except `max` is evaluated on a smaller sample there
    # and the first and last w-1 positions are not directly comparable with the
    # middle of the chain.
    for col in df.columns:
        if col == score:
            continue
        if col == "coverage_depth" or (col.startswith("w") and col.endswith("_Score")):
            out[col] = pd.to_numeric(df[col], errors="coerce")
    if sequence and len(out) != len(sequence):
        raise AdapterError(
            f"{path}: {len(out)} projected residues, sequence has {len(sequence)}. "
            f"AmyloGram's projection must cover the whole chain; a length "
            f"mismatch means the window table and the sequence disagree."
        )
    return _finalise(out, path)