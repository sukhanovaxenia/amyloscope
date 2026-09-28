"""AmyloDeep adapter: window scores must be projected before they reach a tier.

AmyloDeep's CLI emits one row per rolling WINDOW with a 0-based window start.
Reading that as a per-residue track would shift every score one position toward
the N-terminus and mislocate every region by about half a window, so the adapter
refuses a window table outright rather than silently accepting it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from amyloscope.io.adapters import AdapterError, get_adapter, parse_amylodeep

ABETA42 = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"


def _projected(path, scores, *, base=1, window=10):
    """A per-residue table as aggressor-wrappers' amylodeep parser writes it."""
    pd.DataFrame(
        {
            "Number": range(base, base + len(scores)),
            "Residue": list(ABETA42),
            "Score": scores,
            "window_size": [window] * len(scores),
        }
    ).to_csv(path, index=False)
    return path


def _raw_windows(path, *, window=10):
    """A raw AmyloDeep CLI csv: one row per window, 0-based window start."""
    n_windows = len(ABETA42) - window + 1
    pd.DataFrame(
        {
            "sequence_id": ["input_sequence"] * n_windows,
            "position": range(n_windows),
            "probability": [0.1] * n_windows,
            "sequence_length": [len(ABETA42)] * n_windows,
            "avg_probability": [0.1] * n_windows,
            "max_probability": [0.1] * n_windows,
        }
    ).to_csv(path, index=False)
    return path


def test_registered_under_its_own_key():
    assert get_adapter("amylodeep") is parse_amylodeep


def test_projected_table_loads(tmp_path):
    scores = [0.05] * 16 + [0.9] * 6 + [0.05] * 20
    out = parse_amylodeep(_projected(tmp_path / "a.csv", scores), ABETA42)
    assert len(out) == len(ABETA42)
    assert out["Number"].tolist() == list(range(1, len(ABETA42) + 1))
    assert set(out.loc[out["APR"] == 1, "Number"]) == set(range(17, 23))
    assert out["window_size"].iloc[0] == 10


def test_zero_based_input_is_shifted_to_one_based(tmp_path):
    """The silent corruption this guards: a one-residue N-terminal shift."""
    scores = [0.0] * len(ABETA42)
    scores[16] = 0.9  # 0-based index 16 == residue 17
    out = parse_amylodeep(_projected(tmp_path / "a.csv", scores, base=0), ABETA42)
    assert out["Number"].tolist() == list(range(1, len(ABETA42) + 1))
    assert out.loc[out["Score"] > 0.5, "Number"].tolist() == [17]


def test_raw_window_table_is_refused_with_the_reason(tmp_path):
    """33 windows for 42 residues is not a per-residue track and must not pass."""
    with pytest.raises(AdapterError, match="WINDOW table"):
        parse_amylodeep(_raw_windows(tmp_path / "raw.csv"), ABETA42)


def test_the_refusal_names_the_consequence(tmp_path):
    with pytest.raises(AdapterError, match="half a window"):
        parse_amylodeep(_raw_windows(tmp_path / "raw.csv"), ABETA42)


def test_threshold_is_configurable(tmp_path):
    scores = [0.6] * len(ABETA42)
    lenient = parse_amylodeep(_projected(tmp_path / "a.csv", scores), ABETA42, threshold=0.5)
    strict = parse_amylodeep(_projected(tmp_path / "b.csv", scores), ABETA42, threshold=0.7)
    assert lenient["APR"].sum() == len(ABETA42)
    assert strict["APR"].sum() == 0


# --- AmyloDeep 0.4 writes its two grains as two labelled tables -------------
#
# 0.3 emitted one row per window and nothing else, so the adapter had only the
# row count to go on. 0.4's residue table names its index residue_number and its
# window table names window_index/window_start_0based, which makes the grain
# readable off the header -- and carries the window spread behind each residue,
# which the aggregated Score cannot convey on its own.


def _amylodeep_04_residue(path, scores, *, window=6, aggregate="mean", heads=5):
    """A residue table as amylodeep 0.4 `--resolution residue` writes it."""
    n = len(scores)
    pd.DataFrame(
        {
            "sequence_id": ["input_sequence"] * n,
            "sequence_length": [len(ABETA42)] * n,
            "window_size": [window] * n,
            "aggregate": [aggregate] * n,
            "heads_used": [heads] * n,
            "avg_probability": [sum(scores) / n] * n,
            "max_probability": [max(scores)] * n,
            "residue_number": range(1, n + 1),
            "residue": list(ABETA42)[:n],
            "probability": scores,
            "coverage_depth": [min(window, n)] * n,
            "window_min": [min(scores)] * n,
            "window_mean": scores,
            "window_max": [max(scores)] * n,
            "window_support": [0.5] * n,
            "window_std": [0.1] * n,
            "window_at_start": scores,
        }
    ).to_csv(path, index=False)
    return path


def _amylodeep_04_windows(path, *, window=6):
    """A window table as amylodeep 0.4 `--resolution window` writes it."""
    n = len(ABETA42) - window + 1
    pd.DataFrame(
        {
            "sequence_id": ["input_sequence"] * n,
            "sequence_length": [len(ABETA42)] * n,
            "window_size": [window] * n,
            "aggregate": ["mean"] * n,
            "heads_used": [5] * n,
            "avg_probability": [0.1] * n,
            "max_probability": [0.1] * n,
            "window_index": range(1, n + 1),
            "window_start_0based": range(n),
            "first_residue": range(1, n + 1),
            "last_residue": range(window, window + n),
            "window_sequence": [ABETA42[i : i + window] for i in range(n)],
            "probability": [0.1] * n,
        }
    ).to_csv(path, index=False)
    return path


def test_amylodeep_04_residue_table_loads(tmp_path):
    """The tool's own residue CSV, not only aggressor-wrappers' rewrite of it."""
    scores = [0.05] * 16 + [0.9] * 6 + [0.05] * 20
    out = parse_amylodeep(_amylodeep_04_residue(tmp_path / "a.csv", scores), ABETA42)
    assert len(out) == len(ABETA42)
    assert out["Number"].tolist() == list(range(1, len(ABETA42) + 1))
    assert set(out.loc[out["APR"] == 1, "Number"]) == set(range(17, 23))
    assert out["Residue"].tolist() == list(ABETA42)


def test_window_provenance_columns_survive(tmp_path):
    """window_size and aggregate decide how far a boundary can be trusted."""
    scores = [0.5] * len(ABETA42)
    out = parse_amylodeep(
        _amylodeep_04_residue(tmp_path / "a.csv", scores, window=6, aggregate="max"),
        ABETA42,
    )
    assert out["window_size"].iloc[0] == 6
    assert out["aggregate"].iloc[0] == "max"
    assert out["coverage_depth"].iloc[0] == 6
    for col in ("window_min", "window_mean", "window_max", "window_support", "window_std"):
        assert col in out.columns, col


def test_a_four_head_run_stays_identifiable(tmp_path):
    """A run that dropped the XGBoost head must not read as a full ensemble."""
    scores = [0.5] * len(ABETA42)
    out = parse_amylodeep(
        _amylodeep_04_residue(tmp_path / "a.csv", scores, heads=4), ABETA42
    )
    assert out["heads_used"].iloc[0] == 4


def test_04_window_table_is_refused_by_its_marker_column(tmp_path):
    """37 windows for 42 residues would also fail the row count; the column is
    the clearer signal, and it is what the error should name."""
    with pytest.raises(AdapterError, match="window_start_0based"):
        parse_amylodeep(_amylodeep_04_windows(tmp_path / "w.csv"), ABETA42)


def test_a_window_table_whose_row_count_matches_is_still_refused(tmp_path):
    """The row-count heuristic alone cannot catch window_size == 1, where there is
    one window per residue. The marker column can, which is why it is checked
    first rather than as a fallback."""
    n = len(ABETA42)
    pd.DataFrame(
        {
            "window_index": range(1, n + 1),
            "window_start_0based": range(n),
            "first_residue": range(1, n + 1),
            "last_residue": range(1, n + 1),
            "window_sequence": list(ABETA42),
            "probability": [0.9] * n,
        }
    ).to_csv(tmp_path / "w1.csv", index=False)
    with pytest.raises(AdapterError, match="WINDOW table"):
        parse_amylodeep(tmp_path / "w1.csv", ABETA42)
