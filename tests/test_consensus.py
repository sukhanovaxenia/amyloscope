"""Smoke and unit tests for amyloscope.

These are self-contained: synthetic tracks and configs are built in-process, so
the suite does not depend on the shipped example fixtures. The emphasis is on
the invariants that the refactor was meant to guarantee -- declarative
detection, fractional (predictor-count-invariant) consensus tiers, and a
working end-to-end run.
"""

from __future__ import annotations

import pandas as pd
import pytest

from amyloscope.config import (
    ConsensusConfig,
    DetectionStrategy,
    Domain,
    PipelineConfig,
    ProteinSpec,
    ToolSpec,
    load_config,
)
from amyloscope.core.consensus import compute_consensus, resolve_count
from amyloscope.core.regions import call_regions
from amyloscope.io.adapters import available_adapters, get_adapter
from amyloscope.io.loader import load_dataset
from amyloscope.pipeline import run

# --------------------------------------------------------------------------- #
# Detection primitives
# --------------------------------------------------------------------------- #


def _track(scores, residues=None, **extra):
    n = len(scores)
    data = {
        "Number": list(range(1, n + 1)),
        "Residue": list(residues) if residues else [""] * n,
        "Score": scores,
    }
    data.update(extra)
    return pd.DataFrame(data)


def test_above_calls_contiguous_region():
    df = _track([0, 0, 1, 1, 1, 1, 1, 0, 0])
    strat = DetectionStrategy(method="above", column="Score", threshold=0.5)
    assert call_regions(df, strat, min_length=5) == [(3, 7)]


def test_below_for_energy_scores():
    # PASTA-like: a run of low (stable) energies is the APR.
    df = _track([0.0, 0.0, -3.0, -3.0, -3.0, -3.0, -3.0, 0.0])
    strat = DetectionStrategy(method="below", column="Score", threshold=-2.0)
    assert call_regions(df, strat, min_length=5) == [(3, 7)]


def test_flag_accepts_bool_and_string():
    df = _track([0] * 6, Fold=["-", "-", "f", "f", "f", "f"])
    strat = DetectionStrategy(
        method="flag", column="Fold", flag_true_values=["f"]
    )
    # 4-residue run is below the default floor...
    assert call_regions(df, strat, min_length=5) == []
    # ...but is returned when the floor is lowered.
    assert call_regions(df, strat, min_length=4) == [(3, 6)]


def test_notnull_detects_label_only_hotspots():
    df = _track([0.0] * 6, Prediction=["", "", "HS", "HS", "HS", "HS"])
    strat = DetectionStrategy(method="notnull", column="Prediction")
    assert call_regions(df, strat, min_length=4) == [(3, 6)]


def test_present_treats_every_row_as_apr():
    # Hit-only format: sparse residue indices with a gap.
    df = pd.DataFrame(
        {"Number": [10, 11, 12, 13, 14, 40, 41], "Residue": [""] * 7,
         "Score": [1.0] * 7}
    )
    strat = DetectionStrategy(method="present")
    # The gap closes the first interval; the 2-residue tail is below the floor.
    assert call_regions(df, strat, min_length=5) == [(10, 14)]


def test_min_length_floor_is_enforced():
    df = _track([0, 1, 1, 1, 0])  # 3-residue hit
    strat = DetectionStrategy(method="above", column="Score", threshold=0.5)
    assert call_regions(df, strat, min_length=5) == []


# --------------------------------------------------------------------------- #
# Fractional consensus
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fraction,n,expected",
    [
        (1.0, 8, 8), (0.75, 8, 6), (0.5, 8, 4),
        (1.0, 6, 6), (0.75, 6, 5), (0.5, 6, 3),
        (0.5, 3, 2), (0.1, 1, 1),  # floor of 1
    ],
)
def test_resolve_count_rescales_with_panel(fraction, n, expected):
    assert resolve_count(fraction, n) == expected


def _toy_config(tmp_path, n_tools=4, agreeing=3):
    """A tiny panel where `agreeing` of `n_tools` call one shared window."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    length = 30
    window = range(10, 21)  # 11 residues
    tools = []
    for k in range(n_tools):
        path = tmp_path / f"tool{k}.tsv"
        rows = []
        for i in range(1, length + 1):
            hit = (i in window) and (k < agreeing)
            rows.append(f"{i}\t{1.0 if hit else 0.0}")
        path.write_text("\n".join(rows) + "\n")
        tools.append(
            ToolSpec(
                name=f"tool{k}",
                adapter="waltz",
                path_template=str(path),
                detection=DetectionStrategy(method="nonzero", column="Score"),
            )
        )
    cfg = PipelineConfig(
        name="toy",
        proteins=[ProteinSpec(id="P", length=length,
                              domains=[Domain("dom", 8, 22, "structured_core")])],
        tools=tools,
        consensus=ConsensusConfig(min_region_length=5),
        output_dir=str(tmp_path / "out"),
    )
    cfg.validate()
    return cfg


def test_consensus_classifies_by_fraction(tmp_path):
    # 3 of 4 tools agree => 0.75 => 'strong' (>=0.75, <1.0).
    cfg = _toy_config(tmp_path, n_tools=4, agreeing=3)
    res = compute_consensus(load_dataset(cfg))
    regions = res.regions["P"]
    assert len(regions) == 1
    assert regions[0].tier == "strong"
    assert regions[0].n_tools == 3


def test_consensus_unanimous_when_all_agree(tmp_path):
    cfg = _toy_config(tmp_path, n_tools=4, agreeing=4)
    res = compute_consensus(load_dataset(cfg))
    assert res.regions["P"][0].tier == "unanimous"


def test_consensus_invariant_to_panel_size(tmp_path):
    # Same 75% agreement at two panel sizes must yield the same tier.
    c1 = _toy_config(tmp_path / "a", n_tools=4, agreeing=3)
    c2 = _toy_config(tmp_path / "b", n_tools=8, agreeing=6)
    t1 = compute_consensus(load_dataset(c1)).regions["P"][0].tier
    t2 = compute_consensus(load_dataset(c2)).regions["P"][0].tier
    assert t1 == t2 == "strong"


# --------------------------------------------------------------------------- #
# Registry and end-to-end
# --------------------------------------------------------------------------- #


def test_all_expected_adapters_registered():
    expected = {
        "aggrescan", "appnn", "foldamyloid", "pasta2",
        "waltz", "aggreprot", "crossbeta", "archcandy",
    }
    assert expected.issubset(set(available_adapters()))
    for key in expected:
        assert callable(get_adapter(key))


def test_end_to_end_writes_artifacts(tmp_path):
    cfg = _toy_config(tmp_path, n_tools=4, agreeing=4)
    art = run(cfg, make_figures=False)
    names = {p.split("/")[-1] for p in art.written_files}
    assert "consensus_regions.tsv" in names
    assert "consensus_statistics.txt" in names
    assert "domain_overlap.txt" in names  # protein declares a domain
    for path in art.written_files:
        with open(path) as handle:
            assert handle.read().strip()


def test_config_roundtrips_from_yaml(tmp_path):
    yaml_text = """
name: rt
proteins:
  - id: P
    length: 12
    domains:
      - {name: d, start: 1, stop: 12, category: disordered}
tools:
  - name: T
    adapter: waltz
    path: nonexistent_{protein}.tsv
    detection: {method: nonzero, column: Score}
consensus:
  tiers:
    - {name: all, min_fraction: 1.0, color: "#000000"}
"""
    path = tmp_path / "c.yaml"
    path.write_text(yaml_text)
    cfg = load_config(path)
    assert cfg.name == "rt"
    assert cfg.protein_ids == ["P"]
    assert cfg.tool_names == ["T"]
    assert cfg.consensus.tiers[0].min_fraction == 1.0
