"""Declared and measured non-independence between panel members.

A fractional consensus treats every enabled predictor as one vote and reads
agreement as converging evidence. That reading is only valid while the models
are mechanistically independent. Some published tools are built ON other tools,
and when both are enabled the panel counts one line of evidence twice.

The inflation is not uniform noise, which is what makes it worth a warning
rather than a footnote: a derived tool agrees with its parent precisely where
the parent calls, so the duplicated vote lands on the residues that already have
the highest agreement. It therefore pushes regions UP the tiers — moderate
towards strong, strong towards unanimous — rather than blurring them. Reported
tiers become optimistic exactly where the conclusions are drawn.

Known derivations
-----------------
``amyloid_predict``
    Re-expresses TANGO and Waltz, so it duplicates two narrow callers at once.
``st_arch``
    A machine-learning predictor trained on beta-arch structures **annotated by
    ArchCandy**; its labels come from ArchCandy's own output, so the two
    necessarily overlap in what they call.
``tapass``
    A meta-predictor combining ArchCandy, TANGO and PASTA.

None of these is excluded. They are legitimate to run — for comparison, or when
the derived model is the one being evaluated — and the honest handling is to let
them run and say what the consensus then means. ``effective_panel_size`` gives
the count of independent lines of evidence, which is the denominator the tier
fractions *ought* to use.

Declared dependency is a priori. :func:`pairwise_overlap` measures the same
thing after the fact from the APR masks actually produced, so a claimed
independence that does not survive contact with the data is visible too.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from ..core.consensus import _per_tool_regions

#: child adapter/tool key -> (parents, reason)
DERIVATIONS: dict[str, tuple[tuple[str, ...], str]] = {
    "amyloid_predict": (
        ("tango", "waltz"),
        "re-expresses TANGO and Waltz rather than modelling independently",
    ),
    "st_arch": (
        ("archcandy", "archcandy_local", "archcandy2"),
        "trained on beta-arch structures annotated by ArchCandy, so its labels "
        "derive from ArchCandy's own calls",
    ),
    "starch": (
        ("archcandy", "archcandy_local", "archcandy2"),
        "trained on beta-arch structures annotated by ArchCandy",
    ),
    "tapass": (
        ("archcandy", "archcandy_local", "archcandy2", "tango", "pasta2", "pasta"),
        "a meta-predictor combining ArchCandy, TANGO and PASTA",
    ),
}


def _key(text: str) -> str:
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


#: What kind of model each predictor is. Independence is not only a question of
#: derivation: nine models that all score the same biophysics are closer to one
#: another than any of them is to a model trained on sequence embeddings, even
#: with no shared code or labels. A fractional consensus cannot see that -- every
#: enabled tool is one vote -- so the composition is reported next to the tier
#: counts, and a tier carried entirely by one family is readable as such.
#:
#: These are families, not derivations: membership implies correlated inductive
#: bias, not a duplicated vote, and nothing here is deducted from the panel size.
MECHANISM_FAMILIES: dict[str, str] = {
    # Pairing energy / beta-propensity from physical potentials.
    "pasta": "biophysical",
    "pasta2": "biophysical",
    "tango": "biophysical",
    "foldamyloid": "biophysical",
    "aggrescan": "biophysical",
    # Position-specific matrices fitted to experimental peptide sets.
    "waltz": "matrix",
    # Explicit structural hypothesis (beta-arch / beta-arcade geometry).
    "archcandy": "structural",
    "archcandy2": "structural",
    "archcandy_local": "structural",
    # Classical ML on hand-built sequence features.
    "appnn": "shallow_ml",
    "amylogram": "shallow_ml",
    "aggreprot": "deep_cnn",
    "crossbeta": "shallow_ml",
    "crossbeta_local": "shallow_ml",
    # Learned protein-language-model representations.
    "amylodeep": "plm_ensemble",
}

#: One line per family, for the report.
FAMILY_NOTES: dict[str, str] = {
    "biophysical": "energy or propensity from physical potentials",
    "matrix": "position-specific matrices fitted to peptide sets",
    "structural": "an explicit structural hypothesis",
    "shallow_ml": "classical ML on hand-built sequence features",
    "deep_cnn": "convolutional network over sequence",
    "plm_ensemble": "protein-language-model embeddings",
    "unclassified": "mechanism not declared here",
}


@dataclass
class Dependency:
    """One enabled tool that derives from other enabled tools."""

    child: str
    parents_present: list[str]
    reason: str

    def message(self) -> str:
        return (
            f"'{self.child}' is not independent of "
            f"{', '.join(repr(p) for p in self.parents_present)}: {self.reason}. "
            f"Consensus tiers count each tool as one vote, and a derived tool "
            f"agrees with its parent exactly where the parent calls — so the "
            f"duplicated vote lands on the residues that already have the "
            f"highest agreement and pushes them UP a tier. Treat tiers "
            f"involving these tools as optimistic, or drop one of them."
        )


@dataclass
class IndependenceReport:
    dependencies: list[Dependency] = field(default_factory=list)
    panel_size: int = 0
    #: {family: [tool names]} for the enabled panel. A family holding a single
    #: tool is the interesting case: its agreement with the rest is evidence
    #: from a different direction, and its disagreement is not outvoted so much
    #: as outnumbered.
    families: dict[str, list[str]] = field(default_factory=dict)

    @property
    def solitary_families(self) -> list[str]:
        """Families represented by exactly one enabled tool."""
        return sorted(name for name, tools in self.families.items() if len(tools) == 1)

    @property
    def effective_panel_size(self) -> int:
        """Independent lines of evidence: the panel minus each derived member.

        A derived tool is subtracted whole. That over-corrects slightly — it is
        not a perfect copy of its parents — but the alternative, subtracting a
        fraction, would need a shared-variance estimate the panel cannot supply,
        and understating independence is the safer error for a claim about
        convergent evidence.
        """
        return self.panel_size - len(self.dependencies)

    def composition_text(self) -> str:
        """The panel by mechanism, and which families stand alone.

        Reported whether or not a derivation was found, because the two questions
        are different: derivation asks whether a vote is duplicated, composition
        asks how many genuinely different ways of looking at the sequence the
        panel contains. A 10-tool panel in which nine tools share one family is
        closer to a 2-tool panel than the tier counts suggest.
        """
        if not self.families:
            return ""
        lines = ["Panel composition by mechanism:"]
        for family in sorted(self.families, key=lambda f: (-len(self.families[f]), f)):
            tools = ", ".join(sorted(self.families[family]))
            note = FAMILY_NOTES.get(family, FAMILY_NOTES["unclassified"])
            lines.append(f"  {family:14s} {len(self.families[family])}  ({note}): {tools}")
        solitary = self.solitary_families
        if solitary and len(self.families) > 1:
            lines += [
                "",
                f"  Represented by a single tool: {', '.join(solitary)}. Agreement "
                f"between a solitary family and the rest of the panel is evidence "
                f"from a different direction and is worth more than one vote; "
                f"disagreement is not outvoted so much as outnumbered, so a tier "
                f"that rests on the majority family alone should say so.",
            ]
        return "\n".join(lines)

    def as_text(self) -> str:
        composition = self.composition_text()
        if not self.dependencies:
            head = (
                f"Independence: no declared derivations among the "
                f"{self.panel_size} enabled predictors."
            )
            return f"{head}\n\n{composition}" if composition else head
        lines = [
            f"Independence: {len(self.dependencies)} of {self.panel_size} enabled "
            f"predictors derive from others in the panel.",
            f"Effective independent evidence: {self.effective_panel_size} of "
            f"{self.panel_size}.",
            "",
        ]
        for dependency in self.dependencies:
            lines.append(f"  {dependency.message()}")
        if composition:
            lines += ["", composition]
        return "\n".join(lines)


def check_independence(config, *, warn: bool = True) -> IndependenceReport:
    """Flag enabled tools that derive from other enabled tools."""
    enabled = list(config.enabled_tools)
    present: dict[str, str] = {}
    for tool in enabled:
        present[_key(tool.name)] = tool.name
        present[_key(tool.adapter)] = tool.name

    report = IndependenceReport(panel_size=len(enabled))
    for tool in enabled:
        family = MECHANISM_FAMILIES.get(
            _key(tool.adapter), MECHANISM_FAMILIES.get(_key(tool.name), "unclassified")
        )
        report.families.setdefault(family, []).append(tool.name)
    for tool in enabled:
        for candidate in (_key(tool.name), _key(tool.adapter)):
            if candidate not in DERIVATIONS:
                continue
            parents, reason = DERIVATIONS[candidate]
            found = sorted({present[p] for p in parents if p in present})
            if found:
                report.dependencies.append(
                    Dependency(child=tool.name, parents_present=found, reason=reason)
                )
            break

    if warn:
        for dependency in report.dependencies:
            warnings.warn(dependency.message(), stacklevel=3)
    return report


def pairwise_overlap(dataset, config) -> list[dict]:
    """Measured Jaccard overlap between every pair of predictors' APR masks.

    The declared table above is a priori. This is the same question asked of the
    run: two tools that share a training signal show it here as an overlap far
    above the panel's typical pair, whether or not anyone declared the link.
    """
    residues: dict[str, set[tuple[str, int]]] = {}
    for protein_tracks in dataset:
        protein = protein_tracks.spec.id
        for tool, intervals in _per_tool_regions(protein_tracks, config).items():
            bucket = residues.setdefault(tool, set())
            for start, stop in intervals:
                bucket.update((protein, p) for p in range(start, stop + 1))

    # Containment is only interpretable against breadth. A narrow caller is
    # ALMOST ALWAYS largely contained in a broad one -- Waltz calls ~4 % of the
    # panel and ArchCandy ~28 %, so Waltz sitting inside ArchCandy is arithmetic,
    # not shared training. The signature of derivation is high containment
    # between tools of SIMILAR breadth, or containment well above other pairs at
    # the same breadth ratio. `breadth_ratio` is emitted so the two can be told
    # apart instead of reading the containment column on its own.
    names = sorted(residues)
    rows: list[dict] = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            union = residues[a] | residues[b]
            if not union:
                continue
            intersection = residues[a] & residues[b]
            rows.append(
                {
                    "tool_a": a,
                    "tool_b": b,
                    "jaccard": round(len(intersection) / len(union), 4),
                    "shared_aa": len(intersection),
                    # Containment: how much of the SMALLER tool sits inside the
                    # larger. A derived narrow tool can have modest Jaccard
                    # (because the parent is broader) while being almost wholly
                    # contained -- which is the signature to look for.
                    "containment_min": round(
                        len(intersection) / min(len(residues[a]), len(residues[b]))
                        if min(len(residues[a]), len(residues[b])) else 0.0,
                        4,
                    ),
                    # 1.0 = equal breadth. Near 1 WITH high containment is the
                    # pattern that warrants suspicion; a small ratio explains
                    # high containment without any shared signal.
                    "breadth_ratio": round(
                        min(len(residues[a]), len(residues[b]))
                        / max(len(residues[a]), len(residues[b]))
                        if max(len(residues[a]), len(residues[b])) else 0.0,
                        4,
                    ),
                }
            )
    return sorted(rows, key=lambda r: -r["containment_min"])
