"""Per-predictor aggregation-prone region (APR) calling.

Converts a normalised track into contiguous APR intervals using the declarative
:class:`~amyloscope.config.DetectionStrategy` attached to each tool. This
replaces the original ``identify_amyloid_regions`` ladder, where the masking
rule was chosen by matching the literal tool name. Decoupling the rule from the
name is what lets an arbitrary predictor be added by configuration alone.

Threshold semantics are INCLUSIVE on both sides: ``above`` calls a residue when
``score >= threshold`` and ``below`` when ``score <= threshold``. State this in
the Methods; the difference is not cosmetic for predictors whose scores pile up
exactly on the cutoff (Waltz, TANGO).
"""

from __future__ import annotations

import numbers as _numbers

import pandas as pd

from ..config import DetectionStrategy

Interval = tuple[int, int]

#: Values accepted as "true" when a flag column carries no explicit
#: ``flag_true_values`` match. Covers the three encodings seen in the wild:
#: real booleans, 0/1 integers, and single-letter string flags.
_DEFAULT_TRUTHY = {"true", "t", "yes", "y", "1"}


def _normalise_flag(value) -> str:
    """Canonical lowercase token for a flag cell, dtype-agnostic.

    ``numpy.bool_(True) is True`` evaluates to False, so identity tests against
    the Python singleton silently fail on a real boolean column; and a 0/1
    integer column never matches the string ``"true"``. Both are normalised
    here so the same ``flag_true_values`` works across all three encodings.
    """
    if isinstance(value, (bool,)) or type(value).__name__ == "bool_":
        return "true" if bool(value) else "false"
    if isinstance(value, _numbers.Number) and not isinstance(value, bool):
        if value == int(value):
            return str(int(value))
        return str(value)
    return str(value).strip().lower()


def _apr_mask(df: pd.DataFrame, strategy: DetectionStrategy) -> pd.Series:
    """Boolean per-residue 'lies in an APR' mask for one track."""
    method = strategy.method
    if method == "present":
        return pd.Series(True, index=df.index)

    col = strategy.column
    if col not in df.columns:
        raise KeyError(
            f"detection column '{col}' absent from track "
            f"(have {list(df.columns)})"
        )
    series = df[col]

    if method in {"above", "below"}:
        if strategy.threshold is None:
            raise ValueError(
                f"detection method '{method}' on column '{col}' requires a "
                f"numeric threshold, but none was configured"
            )
        numeric = pd.to_numeric(series, errors="coerce")
        if method == "above":
            return numeric.ge(strategy.threshold).fillna(False)
        return numeric.le(strategy.threshold).fillna(False)

    if method == "nonzero":
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return numeric != 0
    if method == "notnull":
        return series.notna() & (series.astype(str).str.strip() != "")
    if method == "flag":
        truthy = {_normalise_flag(v) for v in strategy.flag_true_values}
        # A configured value of literal boolean true means "whatever this file
        # uses for true", so expand it to the standard synonyms; a specific
        # sentinel such as FoldAmyloid's 'f' is matched exactly and never
        # expanded, so it cannot accidentally swallow other tokens.
        if "true" in truthy:
            truthy |= _DEFAULT_TRUTHY
        norm = series.map(_normalise_flag)
        return norm.isin(truthy)
    raise ValueError(f"unsupported detection method '{method}'")


def call_regions(
    df: pd.DataFrame, strategy: DetectionStrategy, min_length: int = 5
) -> list[Interval]:
    """Return APR intervals ``[(start, end), ...]`` for one predictor track.

    Contiguity is defined on the ``Number`` column so that sparse, hit-only
    formats (e.g. ArchCandy interval lists) are grouped correctly even when the
    track does not span every residue.

    ``min_length`` discards short hits *at the predictor level*. This is a
    distinct decision from the minimum length of a consensus region and must be
    reported as such: at the default of 5, a 4-residue Aggrescan or TANGO
    hotspot never reaches the consensus stage at all, which lowers apparent
    agreement everywhere those tools are the short-hit callers.
    """
    if df.empty:
        return []
    if "Number" not in df.columns:
        raise KeyError(f"track lacks a 'Number' column (have {list(df.columns)})")

    mask = _apr_mask(df, strategy).to_numpy()
    positions = pd.to_numeric(df["Number"], errors="coerce").to_numpy()

    order = positions.argsort(kind="stable")  # tolerate unsorted input
    mask, positions = mask[order], positions[order]

    regions: list[Interval] = []
    start = None
    prev = None
    for is_apr, pos in zip(mask, positions, strict=True):
        if pos != pos:  # NaN position
            continue
        pos = int(pos)
        if is_apr:
            if start is None:
                start = pos
            elif prev is not None and pos != prev + 1:
                _emit(regions, start, prev, min_length)
                start = pos
            prev = pos
        else:
            if start is not None:
                _emit(regions, start, prev, min_length)
                start = None
                prev = None
    if start is not None:
        _emit(regions, start, prev, min_length)
    return regions


def _emit(regions: list[Interval], start: int, end: int, min_length: int) -> None:
    if end - start + 1 >= min_length:
        regions.append((max(1, int(start)), int(end)))