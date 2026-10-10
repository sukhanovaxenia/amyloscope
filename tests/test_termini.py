"""Terminal bias: the two statistics, the replication count, the confounds."""

from __future__ import annotations

import numpy as np
import pytest

from amyloscope.analysis.termini import (
    ProteinTerminalBias,
    format_terminal_report,
    terminal_bias,
)


def _uniformish(n_proteins: int, length: int, starts: list[int]) -> tuple[dict, dict]:
    regions = {f"p{i}": [(s, s + 7) for s in starts] for i in range(n_proteins)}
    lengths = {f"p{i}": length for i in range(n_proteins)}
    return regions, lengths


def test_n_terminal_placement_is_detected() -> None:
    regions, lengths = _uniformish(20, 600, [20, 60, 100, 140])
    res = terminal_bias(regions, lengths, draws=300, per_protein_draws=100, seed=0)
    assert res.pooled_shift < 0
    assert res.fraction_n_shifted == pytest.approx(1.0)
    assert res.n_tertile == pytest.approx(1.0)
    assert res.c_tertile == pytest.approx(0.0)


def test_c_terminal_placement_is_detected() -> None:
    regions, lengths = _uniformish(20, 600, [450, 500, 540, 580])
    res = terminal_bias(regions, lengths, draws=300, per_protein_draws=100, seed=0)
    assert res.pooled_shift > 0
    assert res.fraction_n_shifted == pytest.approx(0.0)
    assert res.c_tertile == pytest.approx(1.0)


def test_central_placement_shows_no_shift() -> None:
    regions, lengths = _uniformish(20, 600, [280, 300, 320])
    res = terminal_bias(regions, lengths, draws=400, per_protein_draws=100, seed=0)
    assert abs(res.pooled_shift) < 0.05
    assert res.mid_tertile == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# The two statistics answer different questions
# --------------------------------------------------------------------------- #
def test_both_ends_is_invisible_to_position_but_not_to_distance() -> None:
    """A protein with APRs at BOTH ends has a mean position near 0.5.

    This is exactly the case a decile histogram or a mean-position test reads as
    'unremarkable'. The terminal-distance statistic sees it.
    """
    regions = {f"p{i}": [(10, 17), (20, 27), (570, 577), (580, 587)]
               for i in range(20)}
    lengths = {f"p{i}": 600 for i in range(20)}
    res = terminal_bias(regions, lengths, draws=400, per_protein_draws=100, seed=0)
    assert abs(res.pooled_shift) < 0.05                    # position: nothing
    assert res.pooled_distance < res.pooled_null_distance  # distance: closer
    assert res.pooled_distance_p < 0.05


def test_one_sided_bias_moves_position_far_more_than_distance() -> None:
    """Conversely: APRs packed in the N-terminal third are not 'near an end'.

    They shift the mean position strongly while sitting about as far from the
    nearer terminus as relocation gives. This is the FXR1 pattern -- an
    N-terminal HALF, not a terminal affinity -- and on the real data the
    distance statistic returns p = 0.26 against p at the floor for position.

    The assertion compares effect sizes rather than thresholding a p-value: the
    distance shift here is small but not exactly zero, and which side of 0.05 it
    lands on depends on the seed, which would make a p-threshold test flaky
    without making it more informative.
    """
    regions = {f"p{i}": [(150, 157), (170, 177), (190, 197)] for i in range(20)}
    lengths = {f"p{i}": 600 for i in range(20)}
    res = terminal_bias(regions, lengths, draws=400, per_protein_draws=100, seed=0)
    assert res.pooled_shift < -0.1                       # position: strong
    distance_shift = abs(res.pooled_distance - res.pooled_null_distance)
    assert distance_shift < abs(res.pooled_shift) / 5    # distance: far weaker


# --------------------------------------------------------------------------- #
# Replication, which is the point of the module
# --------------------------------------------------------------------------- #
def test_replication_counts_individual_proteins() -> None:
    """Half N-shifted, half C-shifted, pooled mean near 0.5.

    The pooled statistic says 'no bias'; the replication count says 'every
    protein is biased, in one of two directions'. Reporting only the first would
    lose the result entirely.
    """
    regions, lengths = {}, {}
    for i in range(20):
        if i % 2:
            regions[f"p{i}"] = [(20, 27), (60, 67)]
        else:
            regions[f"p{i}"] = [(530, 537), (570, 577)]
        lengths[f"p{i}"] = 600
    res = terminal_bias(regions, lengths, draws=400, per_protein_draws=100, seed=0)
    assert abs(res.pooled_shift) < 0.05
    assert res.fraction_n_shifted == pytest.approx(0.5)
    assert all(abs(e.position_shift) > 0.1 for e in res.per_protein)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_empty_input_is_not_an_error() -> None:
    res = terminal_bias({}, {}, draws=10, per_protein_draws=10)
    assert res.n_proteins == 0
    assert "No regions" in format_terminal_report(res)


def test_min_length_filters_fragments() -> None:
    regions = {"full": [(20, 27)], "fragment": [(5, 12)]}
    lengths = {"full": 600, "fragment": 40}
    res = terminal_bias(regions, lengths, draws=100, per_protein_draws=50,
                        min_length=100)
    assert res.n_proteins == 1
    assert res.per_protein[0].protein == "full"


def test_p_floor_is_reported() -> None:
    regions, lengths = _uniformish(10, 600, [20, 40])
    res = terminal_bias(regions, lengths, draws=199, per_protein_draws=50, seed=0)
    assert res.p_floor == pytest.approx(1 / 200)
    assert "floor" in format_terminal_report(res)


def test_length_confound_is_reported() -> None:
    """Fixed absolute APRs + varying chain length => correlated normalised position.

    This is the FXR1 case: APRs sit at fixed coordinates in the structured half
    while the C-terminal tail varies, so a longer chain pushes the same APRs to a
    lower normalised position. Part of a normalised shift is then arithmetic, and
    the report has to say so.
    """
    regions = {f"p{i}": [(100, 107), (150, 157)] for i in range(30)}
    lengths = {f"p{i}": 400 + 20 * i for i in range(30)}
    res = terminal_bias(regions, lengths, draws=200, per_protein_draws=50, seed=0)
    assert res.length_correlation < -0.5
    assert "Non-trivial" in format_terminal_report(res)


def test_report_leads_with_replication() -> None:
    regions, lengths = _uniformish(10, 600, [20, 60])
    res = terminal_bias(regions, lengths, draws=200, per_protein_draws=50, seed=0)
    text = format_terminal_report(res)
    assert text.index("REPLICATION ACROSS PROTEINS") < text.index("POOLED")


# --------------------------------------------------------------------------- #
# Multiple-comparison correction, and the two numbers it separates
# --------------------------------------------------------------------------- #
def test_q_values_are_filled_and_at_least_the_raw_p() -> None:
    regions, lengths = _uniformish(30, 600, [20, 60, 100])
    res = terminal_bias(regions, lengths, draws=200, per_protein_draws=200, seed=0)
    assert all(np.isfinite(e.q_position) for e in res.per_protein)
    assert all(e.q_position >= e.p_position - 1e-12 for e in res.per_protein)


def test_correction_is_over_the_whole_per_protein_family() -> None:
    """More proteins, same per-protein evidence => weaker individual q-values."""
    few, lf = _uniformish(5, 600, [20, 60, 100])
    many, lm = _uniformish(80, 600, [20, 60, 100])
    q_few = min(e.q_position for e in
                terminal_bias(few, lf, draws=200, per_protein_draws=400,
                              seed=0).per_protein)
    q_many = min(e.q_position for e in
                 terminal_bias(many, lm, draws=200, per_protein_draws=400,
                               seed=0).per_protein)
    assert q_many >= q_few


def test_direction_and_individual_significance_are_separate_claims() -> None:
    """The FXR1 shape: unanimous direction, little individual significance.

    Three regions give a per-protein permutation test very little resolution,
    so almost nothing survives FDR while every protein still points the same
    way. Reporting only the discovery count would lose the effect; reporting
    only the count of shifted proteins would overstate each one.
    """
    regions, lengths = _uniformish(40, 600, [120, 160, 200])
    res = terminal_bias(regions, lengths, draws=200, per_protein_draws=200, seed=0)
    assert res.n_shifted == res.n_directional == 40
    assert res.sign_test_p < 1e-9
    assert res.n_significant <= res.n_shifted


def test_sign_test_sees_no_direction_when_there_is_none() -> None:
    regions, lengths = {}, {}
    for i in range(40):
        regions[f"p{i}"] = [(20, 27)] if i % 2 else [(570, 577)]
        lengths[f"p{i}"] = 600
    res = terminal_bias(regions, lengths, draws=200, per_protein_draws=100, seed=0)
    assert res.n_shifted == 20
    assert res.sign_test_p > 0.5


def test_ties_are_excluded_from_the_sign_test() -> None:
    regions, lengths = _uniformish(10, 600, [20])
    res = terminal_bias(regions, lengths, draws=100, per_protein_draws=100, seed=0)
    assert res.n_directional <= len(res.per_protein)
    assert res.n_shifted <= res.n_directional


def test_report_states_both_numbers_and_the_p_floor() -> None:
    regions, lengths = _uniformish(20, 600, [20, 60])
    res = terminal_bias(regions, lengths, draws=200, per_protein_draws=150, seed=0)
    text = format_terminal_report(res)
    assert "DIRECTION" in text and "INDIVIDUALLY" in text
    assert "Sign test" in text
    assert "Benjamini-Hochberg" in text
    assert "per-protein p floor" in text
    assert "phylogen" in text.lower()
