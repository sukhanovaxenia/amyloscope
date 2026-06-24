"""Directional substitution proposals.

Given a target position and the region it belongs to, every substitution vector
is expressed as the same operation — score all nineteen alternative residues by
their windowed effect on the chosen grammar, then select by the sign and
magnitude that the vector calls for:

* **activating** — the substitutions that most *raise* the region propensity
  (strengthen an APR core, or remove a flanking gatekeeper);
* **inhibiting** — those that most *lower* it (install a beta-breaker or charge,
  strip a sticker);
* **neutral / conservative** — iso-physicochemical swaps whose effect is closest
  to zero, the negative-control arm.

Keeping the three on one scoring substrate is what makes the directions
comparable and the controls meaningful: a "neutral" mutation is neutral *on the
same axis* the activating and inhibiting arms move along.
"""

from __future__ import annotations

from dataclasses import dataclass

from .effect import EffectWeights, mutation_region_effect
from .scales import AMINO_ACIDS, conservative_partners


@dataclass(frozen=True)
class Substitution:
    position: int       # 1-based
    wt: str
    mut: str
    effect: float       # Δ region score (signed; grammar-specific)

    @property
    def label(self) -> str:
        # HGVS-like protein notation, e.g. p.F42I.
        return f"p.{self.wt}{self.position}{self.mut}"


def _scored_alternatives(
    seq: str, pos: int, start: int, stop: int, mode: str, weights: EffectWeights,
) -> list[Substitution]:
    wt = seq[pos - 1]
    out = []
    for aa in AMINO_ACIDS:
        if aa == wt:
            continue
        eff = mutation_region_effect(seq, pos, aa, start, stop, mode, weights)
        out.append(Substitution(pos, wt, aa, eff))
    return out


def propose(
    seq: str, pos: int, start: int, stop: int, mode: str, vector: str,
    weights: EffectWeights, top_k: int = 3, min_abs_effect: float = 0.0,
) -> list[Substitution]:
    """Ranked substitutions at ``pos`` for one grammar/vector.

    ``start``/``stop`` bound the region whose windowed propensity is scored
    (1-based, inclusive). ``top_k`` caps the returned candidates.
    """
    alts = _scored_alternatives(seq, pos, start, stop, mode, weights)

    if vector == "activating":
        alts.sort(key=lambda s: s.effect, reverse=True)
        ranked = [s for s in alts if s.effect > min_abs_effect]
    elif vector == "inhibiting":
        alts.sort(key=lambda s: s.effect)
        ranked = [s for s in alts if s.effect < -min_abs_effect]
    elif vector == "neutral":
        partners = set(conservative_partners(seq[pos - 1], mode))
        ranked = sorted(
            (s for s in alts if s.mut in partners), key=lambda s: abs(s.effect)
        )
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"unknown substitution vector: {vector}")

    return ranked[:top_k]
