"""Per-residue biophysical scales and groupings for directed mutagenesis.

The directional rules in :mod:`amyloscope.mutate.rules` reduce to scoring
substitutions against a small set of physicochemical axes that the experimental
literature has repeatedly shown to govern aggregation:

* **Hydrophobicity** — Kyte & Doolittle (1982, *J. Mol. Biol.*) hydropathy.
* **Beta-sheet propensity** — Chou & Fasman (1978, *Adv. Enzymol.*) Pβ.
* **Charge** — formal side-chain charge at neutral pH (His treated as weakly
  positive).
* **Aromatic / cation-π valence** — the "sticker" strength that drives
  low-complexity-domain phase separation (Wang et al. 2018, *Cell*; Martin et
  al. 2020, *Science*; Bremer et al. 2022, *Nat. Chem.*), where Tyr outweighs
  Phe and Arg outweighs Lys.

From hydrophobicity, beta-propensity and charge an intrinsic amyloid
aggregation propensity is composed in :mod:`amyloscope.mutate.effect` in the
spirit of the Chiti–Dobson rate model (Chiti et al. 2003, *Nature*; Chiti &
Dobson 2006, *Annu. Rev. Biochem.*) and Zyggregator (Pawar et al. 2005, *J.
Mol. Biol.*). Keeping the primary scales explicit here — rather than a single
opaque aggregation number — lets a user audit or override any axis from config.

All scales are plain ``dict[str, float]`` over the 20 standard one-letter codes,
so they are trivially overridable.
"""

from __future__ import annotations

# Kyte & Doolittle (1982) hydropathy index.
KYTE_DOOLITTLE: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Chou & Fasman (1978) beta-sheet conformational parameter (Pβ).
CHOU_FASMAN_BETA: dict[str, float] = {
    "A": 0.83, "R": 0.93, "N": 0.89, "D": 0.54, "C": 1.19,
    "Q": 1.10, "E": 0.37, "G": 0.75, "H": 0.87, "I": 1.60,
    "L": 1.30, "K": 0.74, "M": 1.05, "F": 1.38, "P": 0.55,
    "S": 0.75, "T": 1.19, "W": 1.37, "Y": 1.47, "V": 1.70,
}

# Formal side-chain charge at ~pH 7 (His given a small positive weight).
CHARGE: dict[str, float] = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
    "A": 0.0, "N": 0.0, "C": 0.0, "Q": 0.0, "G": 0.0,
    "I": 0.0, "L": 0.0, "M": 0.0, "F": 0.0, "P": 0.0,
    "S": 0.0, "T": 0.0, "W": 0.0, "Y": 0.0, "V": 0.0,
}

# Phase-separation "sticker" valence: aromatic pi plus cation-pi donors.
# Encodes the experimentally observed ordering Tyr > Trp > Phe for aromatics and
# Arg >> Lys for cation-pi; spacers contribute zero.
STICKER_VALENCE: dict[str, float] = {
    "Y": 1.00, "W": 0.90, "F": 0.85, "H": 0.25,
    "R": 0.70, "K": 0.25,
    "A": 0.0, "N": 0.0, "C": 0.0, "D": 0.0, "E": 0.0,
    "G": 0.0, "I": 0.0, "L": 0.0, "M": 0.0, "P": 0.0,
    "Q": 0.0, "S": 0.0, "T": 0.0, "V": 0.0,
}

AMINO_ACIDS: tuple[str, ...] = tuple(KYTE_DOOLITTLE)

# Functional groupings used by the rule engine.
AROMATIC = frozenset("FYWH")
ALIPHATIC = frozenset("AVLIM")
POSITIVE = frozenset("KRH")
NEGATIVE = frozenset("DE")
POLAR_UNCHARGED = frozenset("STNQ")
# Classic aggregation gatekeepers: charge (electrostatic) and the beta-breakers.
BETA_BREAKERS = frozenset("PG")
CHARGED_GATEKEEPERS = frozenset("DEKR")
GATEKEEPERS = BETA_BREAKERS | CHARGED_GATEKEEPERS

# Iso-physicochemical groups for conservative (negative-control) substitutions.
# Mode-specific because some swaps that are near-silent for amyloid are not for
# phase separation: Phe<->Tyr and Lys<->Arg change sticker/cation-pi valence, so
# they are excluded from the condensate conservative set.
CONSERVATIVE_GROUPS_AMYLOID: tuple[frozenset[str], ...] = (
    frozenset("ILVM"), frozenset("FY"), frozenset("DE"),
    frozenset("KR"), frozenset("ST"), frozenset("NQ"),
)
CONSERVATIVE_GROUPS_CONDENSATE: tuple[frozenset[str], ...] = (
    frozenset("ILV"), frozenset("DE"), frozenset("ST"),
    frozenset("NQ"), frozenset("AG"),
)


def conservative_partners(residue: str, mode: str) -> list[str]:
    """Iso-physicochemical alternatives to ``residue`` for the given mode."""
    groups = (
        CONSERVATIVE_GROUPS_CONDENSATE
        if mode == "condensate"
        else CONSERVATIVE_GROUPS_AMYLOID
    )
    for group in groups:
        if residue in group:
            return sorted(group - {residue})
    return []
