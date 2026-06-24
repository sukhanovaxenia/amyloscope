"""Compare two consensus runs to quantify a mutation's effect.

Once mutant sequences have been scored by the external predictor panel and
post-processed with ``amyloscope run``, this diffs the wild-type and mutant
``consensus_regions.tsv`` files: which consensus APRs were gained, lost, or
retained, and the net change in aggregation-prone residues. This is the
authoritative read-out the intrinsic load delta only approximates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_REGION_RE = re.compile(r"^(?P<pid>.+):(?P<start>\d+)-(?P<stop>\d+)$")


@dataclass
class IntervalSet:
    intervals: list[tuple[int, int]]

    @property
    def residues(self) -> set[int]:
        out: set[int] = set()
        for a, b in self.intervals:
            out.update(range(a, b + 1))
        return out


def _load(tsv: str | Path, protein: str | None) -> IntervalSet:
    df = pd.read_csv(tsv, sep="\t")
    intervals: list[tuple[int, int]] = []
    for value in df["Protein"].astype(str):
        m = _REGION_RE.match(value)
        if not m:
            continue
        if protein and m["pid"] != protein:
            continue
        intervals.append((int(m["start"]), int(m["stop"])))
    return IntervalSet(sorted(intervals))


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


@dataclass
class ConsensusDiff:
    gained: list[tuple[int, int]]
    lost: list[tuple[int, int]]
    retained: list[tuple[int, int]]
    wt_residues: int
    mutant_residues: int

    @property
    def residue_delta(self) -> int:
        return self.mutant_residues - self.wt_residues

    def report(self) -> str:
        lines = [
            "Consensus APR comparison (wild type -> mutant)",
            "=" * 46,
            f"APR residues:   {self.wt_residues} -> {self.mutant_residues} "
            f"({self.residue_delta:+d})",
            f"Regions gained: {len(self.gained)}  "
            f"{[f'{a}-{b}' for a, b in self.gained]}",
            f"Regions lost:   {len(self.lost)}  "
            f"{[f'{a}-{b}' for a, b in self.lost]}",
            f"Regions kept:   {len(self.retained)}",
        ]
        verdict = (
            "increased" if self.residue_delta > 0
            else "decreased" if self.residue_delta < 0
            else "unchanged"
        )
        lines.append(f"Net effect:     aggregation-prone coverage {verdict}.")
        return "\n".join(lines)


def compare_consensus(
    wt_tsv: str | Path, mutant_tsv: str | Path, protein: str | None = None
) -> ConsensusDiff:
    """Diff wild-type vs mutant ``consensus_regions.tsv`` outputs."""
    wt = _load(wt_tsv, protein)
    mut = _load(mutant_tsv, protein)
    gained = [b for b in mut.intervals if not any(_overlaps(a, b) for a in wt.intervals)]
    lost = [a for a in wt.intervals if not any(_overlaps(a, b) for b in mut.intervals)]
    retained = [a for a in wt.intervals if any(_overlaps(a, b) for b in mut.intervals)]
    return ConsensusDiff(
        gained=gained, lost=lost, retained=retained,
        wt_residues=len(wt.residues), mutant_residues=len(mut.residues),
    )
