"""Sequence-level scoring used to rank candidate mutations.

Two intrinsic per-residue profiles are defined, smoothed over a short window:

* an **amyloid** propensity composed from normalised Kyte–Doolittle
  hydrophobicity and Chou–Fasman beta-propensity, minus electrostatic and
  beta-breaker (Pro/Gly) penalties — the Chiti–Dobson/Zyggregator logic;
* a **condensate** propensity from sticker/cation-pi valence.

Because both profiles carry the gatekeeper penalties explicitly, scoring a
mutation over a *window* (rather than at the single mutated residue) reproduces
context effects emergently: raising hydrophobicity in an APR core increases the
window score, while removing a flanking charge or proline also increases it by
lifting a penalty. This is what lets one uniform "score all 19 substitutions and
rank by signed effect" rule express both core-strengthening and
gatekeeper-engineering moves without special-casing.

The numbers are a transparent linear model, not a fitted predictor: the engine
ranks candidates for downstream prediction, so monotonicity with the established
axes matters more than absolute calibration. All weights are config-exposed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .scales import (
    BETA_BREAKERS,
    CHARGE,
    CHOU_FASMAN_BETA,
    KYTE_DOOLITTLE,
    STICKER_VALENCE,
)

# Min/max of the raw scales, for 0–1 normalisation.
_KD_LO, _KD_HI = min(KYTE_DOOLITTLE.values()), max(KYTE_DOOLITTLE.values())
_CF_LO, _CF_HI = min(CHOU_FASMAN_BETA.values()), max(CHOU_FASMAN_BETA.values())


def _norm(value: float, lo: float, hi: float) -> float:
    return (value - lo) / (hi - lo) if hi > lo else 0.0


@dataclass(frozen=True)
class EffectWeights:
    """Weights for the intrinsic amyloid propensity and scoring window."""

    hydrophobicity: float = 1.0
    beta: float = 1.0
    charge_penalty: float = 0.7
    breaker_penalty: float = 1.2
    window: int = 7  # smoothing window for the per-residue profile
    flank: int = 3   # residues of context added around a target region

    extra: dict = field(default_factory=dict)


def _residue_amyloid(aa: str, w: EffectWeights) -> float:
    h = _norm(KYTE_DOOLITTLE.get(aa, 0.0), _KD_LO, _KD_HI)
    b = _norm(CHOU_FASMAN_BETA.get(aa, 0.0), _CF_LO, _CF_HI)
    score = w.hydrophobicity * h + w.beta * b
    score -= w.charge_penalty * abs(CHARGE.get(aa, 0.0))
    if aa in BETA_BREAKERS:
        score -= w.breaker_penalty
    return score


def _residue_condensate(aa: str) -> float:
    return STICKER_VALENCE.get(aa, 0.0)


def residue_propensity(aa: str, mode: str, weights: EffectWeights) -> float:
    """Unsmoothed per-residue propensity for the requested grammar."""
    return _residue_condensate(aa) if mode == "condensate" else _residue_amyloid(
        aa, weights
    )


def profile(seq: str, mode: str, weights: EffectWeights) -> list[float]:
    """Window-smoothed per-residue propensity along ``seq`` (1-based order)."""
    raw = [residue_propensity(aa, mode, weights) for aa in seq]
    n = len(raw)
    half = max(0, weights.window // 2)
    smoothed = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        smoothed.append(sum(raw[lo:hi]) / (hi - lo))
    return smoothed


def aggregation_load(seq: str, mode: str, weights: EffectWeights) -> float:
    """Global score: summed positive smoothed propensity ("aggregation load").

    For the amyloid grammar this is the integrated supra-zero propensity; for
    the condensate grammar, total sticker valence. Increases mean more
    aggregation/phase-separation-prone sequence.
    """
    return float(sum(max(0.0, v) for v in profile(seq, mode, weights)))


def _region_mean(
    seq: str, start: int, stop: int, mode: str, weights: EffectWeights,
    override: tuple[int, str] | None = None,
) -> float:
    """Mean smoothed propensity over [start-flank, stop+flank] (1-based, incl.).

    Computed locally — only residues whose smoothing windows touch the region
    are read — so the cost is independent of total sequence length, which keeps a
    whole-sequence scan tractable. ``override`` substitutes ``(position, aa)``
    in place without copying the sequence, used to score a candidate mutation.

    The flank is what makes the score gatekeeper-aware: a charge or proline just
    outside the core still falls inside the scored window.
    """
    n = len(seq)
    half = max(0, weights.window // 2)
    lo = max(0, start - 1 - weights.flank)
    hi = min(n, stop + weights.flank)
    if lo >= hi:
        return 0.0
    span_lo = max(0, lo - half)
    span_hi = min(n, hi + half)
    raw = []
    for i in range(span_lo, span_hi):
        aa = override[1] if (override and i == override[0] - 1) else seq[i]
        raw.append(residue_propensity(aa, mode, weights))
    total = 0.0
    for i in range(lo, hi):
        wlo, whi = max(0, i - half), min(n, i + half + 1)
        window = raw[wlo - span_lo : whi - span_lo]
        total += sum(window) / len(window)
    return total / (hi - lo)


def region_score(
    seq: str, start: int, stop: int, mode: str, weights: EffectWeights
) -> float:
    """Windowed, gatekeeper-aware propensity of a region (see :func:`_region_mean`)."""
    return _region_mean(seq, start, stop, mode, weights)


def mutation_region_effect(
    wt: str, pos: int, new_aa: str, start: int, stop: int,
    mode: str, weights: EffectWeights,
) -> float:
    """Δ region score for substituting position ``pos`` (1-based) with ``new_aa``."""
    return _region_mean(
        wt, start, stop, mode, weights, override=(pos, new_aa)
    ) - _region_mean(wt, start, stop, mode, weights)


# --------------------------------------------------------------------------- #
# Charge patterning ("distant" sequence context)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChargePatterning:
    ncpr: float   # net charge per residue
    fcr: float    # fraction of charged residues
    scd: float    # sequence charge decoration (Sawle & Ghosh 2015)


def charge_patterning(seq: str) -> ChargePatterning:
    """Non-local charge descriptors governing IDR compaction and LLPS.

    SCD (Das & Pappu 2013, *PNAS*; Sawle & Ghosh 2015, *J. Chem. Phys.*) rewards
    segregated like-charges and is the standard sequence-only proxy for the
    "distant" electrostatic context that point scales miss.
    """
    n = len(seq)
    if n == 0:
        return ChargePatterning(0.0, 0.0, 0.0)
    q = [CHARGE.get(aa, 0.0) for aa in seq]
    pos = sum(1 for x in q if x > 0)
    neg = sum(1 for x in q if x < 0)
    ncpr = (pos - neg) / n
    fcr = (pos + neg) / n
    scd = 0.0
    for j in range(1, n):
        qj = q[j]
        if qj == 0.0:
            continue
        for i in range(j):
            qi = q[i]
            if qi != 0.0:
                scd += qi * qj * math.sqrt(j - i)
    return ChargePatterning(ncpr, fcr, scd / n)
