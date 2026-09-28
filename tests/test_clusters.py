"""APR clusters: core/extent separation and the shoulder-provenance verdict."""

from __future__ import annotations

import pytest

from amyloscope.analysis.clusters import (
    AMBIGUOUS,
    EXTENSION_SUPPORTED,
    NO_SHOULDER,
    WINDOW_ARTIFACT,
    ToolBreadth,
    classify_breadth,
    _runs,
)


def _breadth(**fractions):
    return {t: ToolBreadth(tool=t, called_aa=int(f * 1000), total_aa=1000)
            for t, f in fractions.items()}


def test_breadth_bands_follow_the_panel_that_was_run():
    """Narrow/broad are terciles of the observed distribution, not a fixed list."""
    breadth = _breadth(Waltz=0.04, TANGO=0.07, AggreProt=0.14,
                       FoldAmyloid=0.16, PASTA2=0.19, APPNN=0.22,
                       ArchCandy=0.27, Aggrescan=0.28, CrossBeta=0.42)
    narrow, broad = classify_breadth(breadth)
    assert narrow == ["AggreProt", "TANGO", "Waltz"]
    assert broad == ["Aggrescan", "ArchCandy", "CrossBeta"]
    assert set(narrow) & set(broad) == set()


def test_degenerate_panel_refuses_to_invent_a_split():
    """Tools that all call the same fraction cannot be ranked by breadth.

    A synthetic fixture where every predictor shares one window set collapses
    the terciles onto a single value. Returning no classification makes every
    verdict AMBIGUOUS, which is the honest reading; splitting on a tie would
    label arbitrary tools 'broad' and make the verdicts look informative.
    """
    breadth = _breadth(a=0.28, b=0.28, c=0.28, d=0.28)
    assert classify_breadth(breadth) == ([], [])


def test_explicit_cut_points_override_the_terciles():
    breadth = _breadth(Waltz=0.04, CrossBeta=0.42, APPNN=0.22)
    narrow, broad = classify_breadth(breadth, narrow_max=0.10, broad_min=0.30)
    assert narrow == ["Waltz"]
    assert broad == ["CrossBeta"]
    assert "APPNN" not in narrow + broad  # intermediate


def test_runs_groups_contiguous_positions():
    assert _runs({3, 4, 5, 9, 10}) == [(3, 5), (9, 10)]
    assert _runs(set()) == []
    assert _runs({7}) == [(7, 7)]


def test_verdict_vocabulary_is_explicit():
    """The four outcomes are distinct strings, not booleans.

    A boolean 'merge?' would collapse 'no shoulder exists' together with
    'a shoulder exists but only broad callers support it', which are opposite
    situations for interpretation.
    """
    assert len({NO_SHOULDER, EXTENSION_SUPPORTED, WINDOW_ARTIFACT, AMBIGUOUS}) == 4
