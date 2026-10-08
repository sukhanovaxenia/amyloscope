"""Terminal bias of consensus APRs, and what survives a phylogenetic objection.

``positional_permutation`` answers one question: is the pooled decile profile of
normalised midpoints further from uniform than relocation would give? That is the
right test for a handful of unrelated proteins and the wrong one for an
orthologue sweep, for two reasons this module addresses.

**A pooled p-value over related proteins is not what it looks like.** Several
hundred orthologues of one protein are not several hundred independent
observations. An APR that arose once and was retained is counted in every
descendant, so with 2,654 regions from 152 taxa the pooled permutation p is
bounded only by the draw count and says almost nothing about how many times the
arrangement arose. The quantity that does survive the objection is the fraction
of INDIVIDUAL proteins whose own placement is shifted against their own null:
"every one of 91 FXR1 proteins is N-shifted relative to its own null" is a claim
phylogeny cannot explain away, and it needs no pooled p at all. Both are
computed here, and :func:`format_terminal_report` prints the per-protein count
first.

**"N-terminal" and "near a terminus" are different hypotheses.** A decile
histogram shows the first, and a protein whose APRs cluster at BOTH ends looks
unremarkable in it -- the deciles are high at either end and the mean sits near
0.5. ``mean_normalised_position`` tests N-versus-C asymmetry; ``mean_terminal_
distance`` tests proximity to the nearer end, whichever it is. They answer
different questions and can disagree, which is informative rather than awkward.

Two confounds are reported rather than silently absorbed:

*Length.* Normalised position divides by chain length, so a family whose members
differ in the length of a terminal extension will show a shift that is partly
architecture and partly arithmetic. The Spearman correlation between a protein's
mean normalised APR position and its length says how much of the first this
could be.

*Truncation.* A fragment spreads its APRs over the full 0-1 axis and manufactures
terminal bias out of nothing. This module does not know which entries are
fragments -- pass only complete sequences, or pass ``min_length`` -- but it
reports the length range it was given so the hazard is visible in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .relocation import relocate_starts

#: Pooled draws. Resolves a p to 1/(draws+1); the pooled p is reported with that
#: floor stated, because "p = 0.0005" at 2,000 draws means "nothing in the null
#: reached the observed value", not a measured probability.
DEFAULT_DRAWS = 2_000

#: Per-protein draws. Smaller because this runs once per protein, and the
#: per-protein quantity that matters is the DIRECTION, not a precise p.
DEFAULT_PER_PROTEIN_DRAWS = 300


def _normalised_midpoint(start: int, length: int, protein_length: int) -> float:
    """Midpoint on 0 (N-terminus) to 1 (C-terminus), anchored on residue 1."""
    if protein_length <= 1:
        return 0.0
    mid = start + (length - 1) / 2
    return (mid - 1.0) / (protein_length - 1.0)


def _terminal_distance(start: int, length: int, protein_length: int) -> float:
    """Distance from the region midpoint to the NEARER terminus, normalised.

    0 at either end, 0.5 at the centre. Deliberately not signed: this is the
    "near an end, whichever end" statistic, and folding the axis is what makes
    it blind to the N/C asymmetry that ``_normalised_midpoint`` measures.
    """
    return min(
        p := _normalised_midpoint(start, length, protein_length), 1.0 - p
    )


@dataclass
class ProteinTerminalBias:
    """One protein's placement against its own relocation null."""

    protein: str
    n_regions: int
    length: int
    observed_position: float
    null_position: float
    observed_distance: float
    null_distance: float
    p_position: float

    @property
    def position_shift(self) -> float:
        """Negative = shifted towards the N-terminus."""
        return self.observed_position - self.null_position

    @property
    def distance_shift(self) -> float:
        """Negative = closer to a terminus than relocation gives."""
        return self.observed_distance - self.null_distance

    @property
    def is_n_shifted(self) -> bool:
        return self.position_shift < 0


@dataclass
class TerminalBiasResult:
    """Pooled and per-protein terminal bias."""

    n_proteins: int
    n_regions: int
    length_range: tuple[int, int]
    draws: int
    per_protein_draws: int

    pooled_position: float = float("nan")
    pooled_null_position: float = float("nan")
    pooled_p: float = float("nan")
    pooled_distance: float = float("nan")
    pooled_null_distance: float = float("nan")
    pooled_distance_p: float = float("nan")

    n_tertile: float = float("nan")
    mid_tertile: float = float("nan")
    c_tertile: float = float("nan")

    per_protein: list[ProteinTerminalBias] = field(default_factory=list)
    length_correlation: float = float("nan")
    length_correlation_p: float = float("nan")

    @property
    def pooled_shift(self) -> float:
        return self.pooled_position - self.pooled_null_position

    @property
    def n_shifted(self) -> int:
        return sum(1 for e in self.per_protein if e.is_n_shifted)

    @property
    def fraction_n_shifted(self) -> float:
        return self.n_shifted / len(self.per_protein) if self.per_protein else float("nan")

    @property
    def p_floor(self) -> float:
        """The smallest p this draw count can express."""
        return 1.0 / (self.draws + 1)


def _empirical_p(null: np.ndarray, observed: float) -> float:
    """Two-sided permutation p, Phipson & Smyth +1 corrected."""
    centre = null.mean()
    extreme = np.abs(null - centre) >= abs(observed - centre) - 1e-12
    return float((extreme.sum() + 1) / (null.size + 1))


def terminal_bias(
    regions_by_protein: dict[str, list[tuple[int, int]]],
    lengths: dict[str, int],
    *,
    draws: int = DEFAULT_DRAWS,
    per_protein_draws: int = DEFAULT_PER_PROTEIN_DRAWS,
    seed: int = 0,
    min_length: int | None = None,
) -> TerminalBiasResult:
    """Terminal bias, pooled and per protein.

    ``regions_by_protein`` maps protein id -> ``(start, end)`` pairs, 1-based
    inclusive. ``lengths`` maps protein id -> chain length. ``min_length`` drops
    short entries before anything is computed, which is the crude guard against
    fragments; prefer filtering on a completeness criterion upstream.
    """
    rng = np.random.default_rng(seed)
    items = {
        p: regs for p, regs in regions_by_protein.items()
        if regs and (min_length is None or lengths.get(p, 0) >= min_length)
    }
    if not items:
        return TerminalBiasResult(
            n_proteins=0, n_regions=0, length_range=(0, 0),
            draws=draws, per_protein_draws=per_protein_draws,
        )

    widths = {p: np.array([e - s + 1 for s, e in regs], dtype=int)
              for p, regs in items.items()}
    starts = {p: np.array([s for s, _ in regs], dtype=int)
              for p, regs in items.items()}

    def positions(spec: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        return np.array([
            _normalised_midpoint(int(s), int(w), lengths[p])
            for p, (ss, ws) in spec.items()
            for s, w in zip(ss, ws)
        ])

    def distances(spec: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        return np.array([
            _terminal_distance(int(s), int(w), lengths[p])
            for p, (ss, ws) in spec.items()
            for s, w in zip(ss, ws)
        ])

    observed_spec = {p: (starts[p], widths[p]) for p in items}
    obs_pos = positions(observed_spec)
    obs_dist = distances(observed_spec)

    null_pos = np.empty(draws)
    null_dist = np.empty(draws)
    for i in range(draws):
        spec = {
            p: (relocate_starts(widths[p], lengths[p], rng), widths[p])
            for p in items
        }
        null_pos[i] = positions(spec).mean()
        null_dist[i] = distances(spec).mean()

    result = TerminalBiasResult(
        n_proteins=len(items),
        n_regions=int(obs_pos.size),
        length_range=(min(lengths[p] for p in items), max(lengths[p] for p in items)),
        draws=draws,
        per_protein_draws=per_protein_draws,
        pooled_position=float(obs_pos.mean()),
        pooled_null_position=float(null_pos.mean()),
        pooled_p=_empirical_p(null_pos, float(obs_pos.mean())),
        pooled_distance=float(obs_dist.mean()),
        pooled_null_distance=float(null_dist.mean()),
        pooled_distance_p=_empirical_p(null_dist, float(obs_dist.mean())),
        n_tertile=float((obs_pos < 1 / 3).mean()),
        mid_tertile=float(((obs_pos >= 1 / 3) & (obs_pos <= 2 / 3)).mean()),
        c_tertile=float((obs_pos > 2 / 3).mean()),
    )

    # Per protein, against its OWN null. This is the replication count.
    for p in items:
        w, L = widths[p], lengths[p]
        o_pos = np.mean([_normalised_midpoint(int(s), int(x), L)
                         for s, x in zip(starts[p], w)])
        o_dist = np.mean([_terminal_distance(int(s), int(x), L)
                          for s, x in zip(starts[p], w)])
        np_pos = np.empty(per_protein_draws)
        np_dist = np.empty(per_protein_draws)
        for i in range(per_protein_draws):
            st = relocate_starts(w, L, rng)
            np_pos[i] = np.mean([_normalised_midpoint(int(s), int(x), L)
                                 for s, x in zip(st, w)])
            np_dist[i] = np.mean([_terminal_distance(int(s), int(x), L)
                                  for s, x in zip(st, w)])
        result.per_protein.append(ProteinTerminalBias(
            protein=p,
            n_regions=int(w.size),
            length=L,
            observed_position=float(o_pos),
            null_position=float(np_pos.mean()),
            observed_distance=float(o_dist),
            null_distance=float(np_dist.mean()),
            p_position=_empirical_p(np_pos, float(o_pos)),
        ))

    # Length confound: does a protein's mean APR position track its length?
    # Skipped when every chain is the same length -- the correlation is then
    # undefined, and scipy says so with a warning rather than a value.
    chain_lengths = [e.length for e in result.per_protein]
    if len(result.per_protein) >= 5 and len(set(chain_lengths)) > 1:
        from scipy import stats as _stats

        rho, pval = _stats.spearmanr(
            chain_lengths, [e.observed_position for e in result.per_protein]
        )
        result.length_correlation = float(rho)
        result.length_correlation_p = float(pval)
    return result


def format_terminal_report(result: TerminalBiasResult) -> str:
    """Plain-text report. The per-protein count is stated before the pooled p."""
    if result.n_proteins == 0:
        return "No regions to test for terminal bias."

    direction = "N-terminus" if result.pooled_shift < 0 else "C-terminus"
    lines = [
        "=" * 70,
        "TERMINAL BIAS OF CONSENSUS APRs",
        "=" * 70,
        f"{result.n_regions} region(s) in {result.n_proteins} protein(s); "
        f"chain lengths {result.length_range[0]}-{result.length_range[1]} aa",
        f"Null: whole regions relocated uniformly within their own protein, "
        f"lengths preserved",
        "",
        "REPLICATION ACROSS PROTEINS  (the number phylogeny cannot explain away)",
        "-" * 70,
        f"  {result.n_shifted} of {len(result.per_protein)} proteins "
        f"({result.fraction_n_shifted:.1%}) are shifted towards the N-terminus",
        f"  relative to their OWN null. Several hundred orthologues are not",
        f"  several hundred independent observations, so this count -- not the",
        f"  pooled p below -- is what belongs in a Methods section.",
        "",
        "POOLED",
        "-" * 70,
        f"  mean normalised position   observed {result.pooled_position:.4f}  "
        f"null {result.pooled_null_position:.4f}  "
        f"shift {result.pooled_shift:+.4f} (towards the {direction})",
        f"  mean distance to nearer end observed {result.pooled_distance:.4f}  "
        f"null {result.pooled_null_distance:.4f}  "
        f"shift {result.pooled_distance - result.pooled_null_distance:+.4f}",
        "",
        f"  p(position) = {result.pooled_p:.2g}"
        + (f"   [floor {result.p_floor:.2g} at {result.draws:,} draws]"
           if result.pooled_p <= result.p_floor + 1e-12 else ""),
        f"  p(distance) = {result.pooled_distance_p:.2g}",
        "",
        f"  tertiles: N {result.n_tertile:.1%}   mid {result.mid_tertile:.1%}   "
        f"C {result.c_tertile:.1%}   (uniform would be ~33% each)",
        "",
        "CONFOUNDS",
        "-" * 70,
    ]
    if np.isfinite(result.length_correlation):
        lines.append(
            f"  Spearman(chain length, mean APR position) = "
            f"{result.length_correlation:+.3f} (p = {result.length_correlation_p:.3g})"
        )
        if abs(result.length_correlation) > 0.3:
            lines.append(
                "  Non-trivial: normalised position divides by length, so part of "
                "the shift\n  may be the length distribution rather than the "
                "architecture. Report the\n  absolute-coordinate profile as well."
            )
        else:
            lines.append(
                "  Weak, so the shift is not an artefact of the length distribution."
            )
    lines += [
        f"  Chain lengths span {result.length_range[0]}-{result.length_range[1]} aa. "
        f"A truncated\n  sequence spreads its APRs over the whole 0-1 axis and "
        f"manufactures terminal\n  bias; confirm these are complete before reading "
        f"the shift as architecture.",
        "",
        "=" * 70,
    ]
    return "\n".join(lines)
