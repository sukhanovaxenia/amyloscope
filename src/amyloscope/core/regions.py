"""Per-predictor aggregation-prone region (APR) calling.

Converts a normalised track into contiguous APR intervals using the declarative
:class:`~amyloscope.config.DetectionStrategy` attached to each tool. This
replaces the original ``identify_amyloid_regions`` ladder, where the masking
rule was chosen by matching the literal tool name. Decoupling the rule from the
name is what lets an arbitrary predictor be added by configuration alone.
"""

from __future__ import annotations

import pandas as pd

from ..config import DetectionStrategy

Interval = tuple[int, int]


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

    if method == "above":
        return series >= strategy.threshold
    if method == "below":
        return series <= strategy.threshold
    if method == "nonzero":
        return series.fillna(0) != 0
    if method == "notnull":
        return series.notna() & (series.astype(str).str.strip() != "")
    if method == "flag":
        # Accept literal booleans and string flags alike (e.g. FoldAmyloid
        # 'f', APPNN True). Aggrescan's "Prediction not null" case is expressed
        # by listing the non-null sentinel(s) in flag_true_values, or by using
        # the dedicated 'nonzero'/'present' methods where appropriate.
        truthy = {str(v).strip().lower() for v in strategy.flag_true_values}
        bool_hits = series.apply(lambda v: v is True)
        str_hits = series.astype(str).str.strip().str.lower().isin(truthy)
        return bool_hits | str_hits
    raise ValueError(f"unsupported detection method '{method}'")


def call_regions(
    df: pd.DataFrame, strategy: DetectionStrategy, min_length: int = 5
) -> list[Interval]:
    """Return APR intervals ``[(start, end), ...]`` for one predictor track.

    Contiguity is defined on the ``Number`` column so that sparse, hit-only
    formats (e.g. ArchCandy interval lists) are grouped correctly even when the
    track does not span every residue.
    """
    if df.empty:
        return []
    mask = _apr_mask(df, strategy).to_numpy()
    numbers = df["Number"].to_numpy()

    regions: list[Interval] = []
    start = None
    prev = None
    for is_apr, pos in zip(mask, numbers, strict=True):
        pos = int(pos)
        if is_apr:
            if start is None:
                start = pos
            elif prev is not None and pos != prev + 1:
                # gap inside a hit-only track closes the current interval
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
        regions.append((int(start), int(end)))
