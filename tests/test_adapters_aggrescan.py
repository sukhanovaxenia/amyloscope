"""AGGRESCAN adapter: which column carries the Hot Spot call.

AGGRESCAN marks Hot Spots in a column, and the column is NHSA -- not HSA, and
not a threshold re-derived from a4v. The Abeta42 row set below is verbatim from
the service, and Y10 is the discriminating case: it carries area but is not a
Hot Spot, because AGGRESCAN assigns shared area before applying its run-length
requirement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amyloscope.io.adapters import parse_aggrescan

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

A4V = [
    -0.631, -0.631, -0.554, -0.393, -0.753, -0.530, -0.988, -0.508, -0.584, 0.102,
    -0.045, -0.145, -0.623, -0.527, -0.570, -0.044, 0.513, 1.110, 1.289, 0.731,
    0.045, 0.013, -0.445, -0.497, -0.475, -0.294, -0.719, -0.620, -0.196, 0.428,
    0.508, 0.891, 1.080, 1.034, 0.563, 0.563, 0.606, 0.742, 0.788, 0.888, 0.778, 0.778,
]
HSA = (
    [0.0] * 9 + [0.122] + [0.0] * 6 + [3.821] * 6 + [0.0] * 7 + [9.906] * 13
)
NHSA = [0.0] * 16 + [0.637] * 6 + [0.0] * 7 + [0.762] * 13
#: AGGRESCAN's own answer for this sequence: nHS = 2.
HOT_SPOTS = {*range(17, 23), *range(30, 43)}


def _write(path: Path, *, flag: bool) -> Path:
    frame = pd.DataFrame(
        {
            "Number": range(1, 43),
            "AA": list(ABETA42),
            "a4v": A4V,
            "HSA": HSA,
            "NHSA": NHSA,
            "a4vAHS": [round(v, 3) for v in A4V],
        }
    )
    if flag:
        frame["Prediction"] = [
            1 if n in HOT_SPOTS else "" for n in frame["Number"]
        ]
    frame.to_csv(path, index=False)
    return path


def test_marker_column_is_honoured_when_present(tmp_path):
    out = parse_aggrescan(_write(tmp_path / "flagged.csv", flag=True), ABETA42)
    called = set(out.loc[out["is_hotspot"], "Number"])
    assert called == HOT_SPOTS


def test_nhsa_recovers_the_call_for_an_export_with_no_marker(tmp_path):
    """Hand-scraped AGGRESCAN tables carry the six data columns and no flag.

    Without this fallback they load with no Hot Spots at all, which reads
    downstream as AGGRESCAN abstaining rather than as a missing column.
    """
    out = parse_aggrescan(_write(tmp_path / "bare.csv", flag=False), ABETA42)
    called = set(out.loc[out["is_hotspot"], "Number"])
    assert called == HOT_SPOTS


def test_hsa_would_have_over_called_y10(tmp_path):
    """The reason the fallback reads NHSA and not HSA."""
    out = parse_aggrescan(_write(tmp_path / "bare.csv", flag=False), ABETA42)
    y10 = out.loc[out["Number"] == 10].iloc[0]
    assert y10["Residue"] == "Y"
    assert y10["HSA"] > 0
    assert y10["NHSA"] == 0
    assert not bool(y10["is_hotspot"])


def test_the_two_routes_agree(tmp_path):
    flagged = parse_aggrescan(_write(tmp_path / "a.csv", flag=True), ABETA42)
    bare = parse_aggrescan(_write(tmp_path / "b.csv", flag=False), ABETA42)
    pd.testing.assert_series_equal(
        flagged["is_hotspot"].reset_index(drop=True),
        bare["is_hotspot"].reset_index(drop=True),
        check_names=False,
    )


def test_score_stays_a4v(tmp_path):
    out = parse_aggrescan(_write(tmp_path / "a.csv", flag=True), ABETA42)
    assert out["Score"].tolist() == pytest.approx(A4V)
