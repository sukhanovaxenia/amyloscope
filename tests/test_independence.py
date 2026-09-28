"""Declared derivations and measured overlap between panel members."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import pytest

from amyloscope.analysis.independence import (
    DERIVATIONS,
    check_independence,
    Dependency,
    IndependenceReport,
)


@dataclass
class _Tool:
    name: str
    adapter: str
    enabled: bool = True


@dataclass
class _Cfg:
    tools: list = field(default_factory=list)

    @property
    def enabled_tools(self):
        return [t for t in self.tools if t.enabled]


def test_known_derivations_name_their_parents():
    assert "tango" in DERIVATIONS["amyloid_predict"][0]
    assert "waltz" in DERIVATIONS["amyloid_predict"][0]
    assert "archcandy" in DERIVATIONS["st_arch"][0]
    assert "archcandy" in DERIVATIONS["tapass"][0]


def test_derived_tool_with_its_parent_present_is_flagged():
    cfg = _Cfg([
        _Tool("ArchCandy", "archcandy_local"),
        _Tool("ST-ARCH", "st_arch"),
        _Tool("Waltz", "waltz"),
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = check_independence(cfg)
    assert [d.child for d in report.dependencies] == ["ST-ARCH"]
    assert "ArchCandy" in report.dependencies[0].parents_present
    assert any("not independent" in str(w.message) for w in caught)


def test_derived_tool_alone_is_not_flagged():
    """Running ST-ARCH without ArchCandy is a legitimate independent panel."""
    cfg = _Cfg([_Tool("ST-ARCH", "st_arch"), _Tool("Waltz", "waltz")])
    report = check_independence(cfg, warn=False)
    assert report.dependencies == []
    assert report.effective_panel_size == report.panel_size == 2


def test_amyloid_predict_flags_both_parents():
    cfg = _Cfg([
        _Tool("TANGO", "tango"),
        _Tool("Waltz", "waltz"),
        _Tool("amyloid_predict", "amyloid_predict"),
    ])
    report = check_independence(cfg, warn=False)
    assert len(report.dependencies) == 1
    assert report.dependencies[0].parents_present == ["TANGO", "Waltz"]


def test_effective_panel_size_subtracts_each_derived_member():
    """The denominator the tiers ought to use, not the one they do use."""
    report = IndependenceReport(
        panel_size=11,
        dependencies=[
            Dependency("ST-ARCH", ["ArchCandy"], "x"),
            Dependency("amyloid_predict", ["TANGO", "Waltz"], "y"),
        ],
    )
    assert report.effective_panel_size == 9
    assert "9 of 11" in report.as_text()


def test_the_warning_explains_the_direction_of_the_bias():
    """It must say tiers go UP, not merely that the tools are correlated.

    A derived tool agrees with its parent where the parent calls, so the
    duplicate vote concentrates on the highest-agreement residues.
    """
    message = Dependency("ST-ARCH", ["ArchCandy"], "trained on its output").message()
    assert "UP a tier" in message
    assert "optimistic" in message


def test_nine_predictor_ribosomal_panel_has_no_declared_derivation():
    cfg = _Cfg([
        _Tool(n, a) for n, a in [
            ("Aggrescan", "aggrescan"), ("APPNN", "appnn"),
            ("FoldAmyloid", "foldamyloid"), ("PASTA2", "pasta2"),
            ("Waltz", "waltz"), ("AggreProt", "aggreprot"),
            ("ArchCandy", "archcandy_local"), ("CrossBeta", "crossbeta_local"),
            ("TANGO", "tango"),
        ]
    ])
    report = check_independence(cfg, warn=False)
    assert report.dependencies == []
    assert report.effective_panel_size == 9


# --------------------------------------------------------------------------- #
# Mechanism families. Derivation asks whether a vote is duplicated; composition
# asks how many genuinely different ways of reading the sequence the panel holds.
# A 10-tool panel where nine share one family is closer to a 2-tool panel than
# the tier counts suggest, and nothing in a fractional consensus can see that.
# --------------------------------------------------------------------------- #

from amyloscope.analysis.independence import (  # noqa: E402
    FAMILY_NOTES,
    MECHANISM_FAMILIES,
)


def _panel(*names):
    """Reuses this module's _Tool/_Cfg rather than redefining them.

    The first version of these tests declared its own _Tool and _Cfg at the
    bottom of the file, which shadowed the ones above and broke four unrelated
    tests -- the hazard of appending to a test module instead of reading it.
    """
    return _Cfg([_Tool(name, name.lower()) for name in names])


def test_every_declared_family_has_a_note():
    """A family name in a report with no explanation is not attribution."""
    for family in set(MECHANISM_FAMILIES.values()):
        assert family in FAMILY_NOTES


def test_amylodeep_is_its_own_family():
    """The point of carrying it: its features are learned, everything else's are not."""
    assert MECHANISM_FAMILIES["amylodeep"] == "plm_ensemble"
    others = {v for k, v in MECHANISM_FAMILIES.items() if k != "amylodeep"}
    assert "plm_ensemble" not in others


def test_composition_counts_the_enabled_panel():
    report = check_independence(
        _panel("tango", "pasta2", "foldamyloid", "amylodeep"), warn=False
    )
    assert report.families["biophysical"] == ["foldamyloid", "pasta2", "tango"] or sorted(
        report.families["biophysical"]
    ) == ["foldamyloid", "pasta2", "tango"]
    assert report.families["plm_ensemble"] == ["amylodeep"]
    assert report.solitary_families == ["plm_ensemble"]


def test_a_solitary_family_is_called_out_in_the_text():
    text = check_independence(_panel("tango", "pasta2", "amylodeep"), warn=False).as_text()
    assert "plm_ensemble" in text
    assert "outnumbered" in text


def test_a_single_family_panel_says_nothing_about_solitude():
    """With one family there is nothing to contrast, so the note is suppressed."""
    report = check_independence(_panel("tango", "pasta2", "foldamyloid"), warn=False)
    assert report.solitary_families == []
    assert "outnumbered" not in report.as_text()


def test_composition_is_reported_even_with_no_derivations():
    text = check_independence(_panel("tango", "amylodeep"), warn=False).as_text()
    assert "no declared derivations" in text
    assert "Panel composition by mechanism" in text


def test_families_do_not_reduce_the_effective_panel_size():
    """A shared family is correlated bias, not a duplicated vote."""
    report = check_independence(
        _panel("tango", "pasta2", "foldamyloid", "aggrescan"), warn=False
    )
    assert report.effective_panel_size == report.panel_size == 4


def test_an_unknown_tool_is_reported_rather_than_dropped():
    report = check_independence(_panel("tango", "some_new_tool"), warn=False)
    assert report.families["unclassified"] == ["some_new_tool"]
