"""The normalised track from the execution layer, and why APR is mandatory.

Every other adapter parses one tool's native export. This one parses what
aggressor-wrappers produces after it has already applied each tool's own threshold
AND direction. The call is what crosses; re-deriving it here from Score would undo
that work and invert PASTA.
"""

from __future__ import annotations

import pandas as pd
import pytest

from amyloscope.io.adapters import AdapterError, get_adapter, parse_aggressor

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _track(path, *, apr=True, scores=None, calls=None, residues=True, n=None):
    n = len(ABETA42) if n is None else n
    frame = {
        "Number": range(1, n + 1),
        "Score": scores if scores is not None else [0.1] * n,
    }
    if residues:
        # Padded rather than sliced: the length-mismatch test needs n > len(ABETA42),
        # and a short Residue column would fail in the DataFrame constructor instead
        # of reaching the check under test.
        letters = (list(ABETA42) * (n // len(ABETA42) + 1))[:n]
        frame["Residue"] = letters
    if apr:
        frame["APR"] = (
            calls if calls is not None
            else [i in range(16, 22) for i in range(1, n + 1)]
        )
    pd.DataFrame(frame).to_csv(path, index=False)
    return path


def test_registered_under_its_own_key():
    assert get_adapter("aggressor") is parse_aggressor


def test_the_call_is_read_from_apr_not_derived_from_score(tmp_path):
    """All scores are 0.1 -- below any sane cut -- yet the APR column says 16-21.
    A generic threshold on Score would call nothing here."""
    out = parse_aggressor(_track(tmp_path / "t.csv"), ABETA42)
    called = out.loc[out["APR"] == 1, "Number"].tolist()
    assert called == list(range(16, 22))
    assert "".join(ABETA42[i - 1] for i in called) == "KLVFFA"


def test_an_inverted_scale_survives_because_the_call_crossed(tmp_path):
    """PASTA's energy is amyloidogenic when LOW. Negative scores with APR set is
    exactly the case a `Score >= cut` on this side would get backwards."""
    n = len(ABETA42)
    path = _track(
        tmp_path / "pasta.csv",
        scores=[-0.2] * 15 + [-4.0] * 6 + [-0.2] * (n - 21),
        calls=[i in range(16, 22) for i in range(1, n + 1)],
    )
    out = parse_aggressor(path, ABETA42)
    assert out.loc[out["APR"] == 1, "Number"].tolist() == list(range(16, 22))
    assert (out["Score"] < 0).all(), "the native scale is carried through unchanged"


def test_a_track_without_apr_is_refused_with_the_reason(tmp_path):
    """Defaulting to a threshold would make a normalisation that lost its
    conclusion look like a tool that called nothing."""
    with pytest.raises(AdapterError) as excinfo:
        parse_aggressor(_track(tmp_path / "t.csv", apr=False), ABETA42)
    message = str(excinfo.value)
    assert "no APR column" in message
    assert "PASTA" in message, "the error should say what would go wrong"
    assert "aggressor-export-amyloscope" in message, "and how to fix it"


@pytest.mark.parametrize(
    "written,expected",
    [
        ([True, False], [1, 0]),
        (["True", "False"], [1, 0]),
        (["true", "false"], [1, 0]),
        ([1, 0], [1, 0]),
        ([1.0, 0.0], [1, 0]),
        (["yes", "no"], [1, 0]),
    ],
)
def test_booleans_survive_a_round_trip_through_csv(tmp_path, written, expected):
    """pandas writes True/False, a spreadsheet may write TRUE/1, and a CSV has no
    booleans at all -- so every spelling has to mean the same thing."""
    pd.DataFrame(
        {"Number": [1, 2], "Residue": ["D", "A"], "Score": [0.1, 0.2], "APR": written}
    ).to_csv(tmp_path / "t.csv", index=False)
    out = parse_aggressor(tmp_path / "t.csv", "DA")
    assert out["APR"].tolist() == expected


def test_a_length_mismatch_names_both_numbers_and_the_likely_cause(tmp_path):
    """The failure that broke a 372-protein export: a track one residue longer than
    the configured sequence. Caught here with both counts, rather than as a
    KeyError three layers down."""
    with pytest.raises(AdapterError) as excinfo:
        parse_aggressor(_track(tmp_path / "t.csv", n=len(ABETA42) + 1), ABETA42)
    message = str(excinfo.value)
    assert f"{len(ABETA42) + 1} positions" in message
    assert f"{len(ABETA42)} residues" in message
    assert "duplicate header" in message


def test_provenance_columns_are_carried_when_present(tmp_path):
    n = len(ABETA42)
    pd.DataFrame(
        {
            "Number": range(1, n + 1),
            "Residue": list(ABETA42),
            "Score": [0.1] * n,
            "APR": [0] * n,
            "window_size": [6] * n,
            "aggregate": ["mean"] * n,
            "heads_used": [4] * n,
            "coverage_depth": [6] * n,
        }
    ).to_csv(tmp_path / "t.csv", index=False)
    out = parse_aggressor(tmp_path / "t.csv", ABETA42)
    assert out["window_size"].iloc[0] == 6
    assert out["aggregate"].iloc[0] == "mean"
    assert out["heads_used"].iloc[0] == 4, "a four-head run must stay identifiable"


def test_residues_come_from_the_sequence_when_the_track_omits_them(tmp_path):
    out = parse_aggressor(_track(tmp_path / "t.csv", residues=False), ABETA42)
    assert out["Residue"].tolist() == list(ABETA42)
