"""Tests for the in-silico mutagenesis module.

Self-contained, with one realistic sequence (human GAPDH, reconstructed from the
TANGO fixture) to exercise the full generate/write path. The emphasis is on the
directional contract — activating raises the chosen grammar's propensity,
inhibiting lowers it via beta-breakers/charge, neutral stays near zero — and on
the target-resolution and combinatorial bounds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from amyloscope.mutate import (
    EffectWeights,
    MutagenesisConfig,
    MutateConfigError,
    MutationSpec,
    ProteinJob,
    SequenceSpec,
    TargetSpec,
    aggregation_load,
    charge_patterning,
    compare_consensus,
    generate,
    load_mutate_config,
    propose,
    resolve_targets,
    run,
)

_DATA = Path(__file__).parent / "data"
_W = EffectWeights()


def _gapdh() -> str:
    df = pd.read_csv(_DATA / "tango_GAPDH.txt", sep="\t", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return "".join(str(a).strip() for a in df["aa"])


# --------------------------------------------------------------------------- #
# Scales / effect
# --------------------------------------------------------------------------- #


def test_charge_patterning_scd_sign():
    # Sawle-Ghosh SCD grows more negative as like-charges segregate; an
    # alternating mix sits near zero.
    seg = charge_patterning("KKKKKEEEEE")
    mix = charge_patterning("KEKEKEKEKE")
    assert seg.scd < mix.scd
    assert seg.fcr == pytest.approx(1.0)
    assert seg.ncpr == pytest.approx(0.0)


def test_aggregation_load_responds_to_hydrophobic_core():
    # Inserting a hydrophobic/high-beta stretch raises amyloid load.
    low = aggregation_load("GSGSGSGSGSGS", "amyloid", _W)
    high = aggregation_load("GSGVIVIVIVGS", "amyloid", _W)
    assert high > low


def test_condensate_load_tracks_aromatics():
    plain = aggregation_load("GSGSGSGSGS", "condensate", _W)
    sticky = aggregation_load("GSYGSYGSYG", "condensate", _W)
    assert sticky > plain


# --------------------------------------------------------------------------- #
# Rules / directionality
# --------------------------------------------------------------------------- #


def test_activating_raises_and_inhibiting_lowers():
    # Target an Ala core position, which can move up (->V/I) or down (->P).
    # Targeting an already-maximal Val would correctly yield no activating move.
    seq = "VVVVVAAAAAVVVVV"  # Ala core at 6-10, target the middle
    act = propose(seq, 8, 6, 10, "amyloid", "activating", _W, top_k=1)
    inh = propose(seq, 8, 6, 10, "amyloid", "inhibiting", _W, top_k=1)
    assert act and act[0].effect > 0
    assert inh and inh[0].effect < 0
    # The strongest inhibitor at a core position should be a beta-breaker.
    assert inh[0].mut in "PG"


def test_neutral_is_conservative_and_near_zero():
    seq = "AAAAAVVVVVAAAAA"
    neu = propose(seq, 8, 6, 10, "amyloid", "neutral", _W, top_k=3)
    assert neu
    for s in neu:
        assert s.mut in "ILM"  # conservative partners of V (amyloid)
        assert abs(s.effect) < 0.5


def test_condensate_neutral_excludes_aromatic_swaps():
    # Phe<->Tyr is near-neutral for amyloid but not for the condensate grammar,
    # so a Phe has no condensate-conservative partner.
    seq = "GGGGFGGGG"
    amyloid_partners = propose(seq, 5, 5, 5, "amyloid", "neutral", _W, top_k=3)
    condensate_partners = propose(seq, 5, 5, 5, "condensate", "neutral", _W, top_k=3)
    assert any(s.mut == "Y" for s in amyloid_partners)
    assert condensate_partners == []


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #


def _cfg(**kw) -> MutagenesisConfig:
    base = dict(
        sequence=SequenceSpec(id="GAPDH", sequence=_gapdh()),
        targets=TargetSpec(regions=[(42, 47)]),
        modes=["amyloid"],
        vectors=["activating", "inhibiting", "neutral"],
        mutation=MutationSpec(orders=[1], max_per_order=10),
    )
    base.update(kw)
    return MutagenesisConfig(**base)


def test_resolve_targets_unions_sources_and_honours_protected():
    cfg = _cfg(
        targets=TargetSpec(regions=[(42, 47)], positions=[130]),
        mutation=MutationSpec(orders=[1], protected_positions=[44]),
    )
    positions = {tp.position for tp in resolve_targets(cfg)}
    assert positions == {42, 43, 45, 46, 47, 130}  # 44 protected out


def test_resolve_targets_from_consensus(tmp_path):
    tsv = tmp_path / "consensus.tsv"
    tsv.write_text(
        "Protein\tClass\tRegion\tTools\tLength\tn_tools\n"
        "GAPDH:42-44\tunanimous\tYMV\tA,B\t3\t8\n"
        "OTHER:10-12\tstrong\tAAA\tA\t3\t6\n"
    )
    cfg = _cfg(
        targets=TargetSpec(consensus_tsv=str(tsv), consensus_protein="GAPDH"),
    )
    positions = {tp.position for tp in resolve_targets(cfg)}
    assert positions == {42, 43, 44}  # OTHER filtered out


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def test_generate_directionality_on_load():
    cfg = _cfg(
        modes=["amyloid"], vectors=["activating", "inhibiting", "neutral"],
        mutation=MutationSpec(orders=[1], max_per_order=5),
    )
    recs = generate(cfg)
    by_vec = {}
    for r in recs:
        by_vec.setdefault(r.vector, []).append(r.delta_load)
    assert max(by_vec["activating"]) > 0
    assert min(by_vec["inhibiting"]) < 0
    assert max(abs(x) for x in by_vec["neutral"]) < max(by_vec["activating"])


def test_double_mutants_stack_effects():
    cfg = _cfg(
        targets=TargetSpec(regions=[(40, 49)], positions=[130]),
        mutation=MutationSpec(orders=[1, 2], max_per_order=10, top_k_positions=8),
    )
    recs = generate(cfg)
    best_single = max(
        r.delta_load for r in recs if r.vector == "activating" and r.order == 1
    )
    best_double = max(
        r.delta_load for r in recs if r.vector == "activating" and r.order == 2
    )
    assert best_double > best_single  # two activating edits beat one


def test_nonsense_truncates_at_region_start():
    cfg = _cfg(
        targets=TargetSpec(regions=[(42, 47), (130, 135)]),
        vectors=["nonsense"],
        mutation=MutationSpec(orders=[1], truncate_at="region_start"),
    )
    recs = [r for r in generate(cfg) if r.vector == "nonsense"]
    lengths = sorted(len(r.sequence) for r in recs)
    assert lengths == [41, 129]  # truncated just before residue 42 and 130
    assert all(r.delta_load < 0 for r in recs)  # ablation lowers load


def test_max_total_caps_output():
    cfg = _cfg(
        targets=TargetSpec(scan_whole_sequence=True),
        modes=["amyloid", "condensate"],
        vectors=["activating", "inhibiting", "neutral"],
        mutation=MutationSpec(orders=[1], max_per_order=100, max_total=50),
    )
    assert len(generate(cfg)) <= 50


# --------------------------------------------------------------------------- #
# I/O and compare
# --------------------------------------------------------------------------- #


def test_run_writes_fasta_and_manifest(tmp_path):
    cfg = _cfg(output_dir=str(tmp_path), name="t")
    art = run(cfg)
    names = {Path(p).name for p in art.written_files}
    # Single protein -> per-protein files named by id, no combined manifest.
    assert names == {"GAPDH_mutants.fasta", "GAPDH_manifest.tsv"}
    fasta = (tmp_path / "GAPDH_mutants.fasta").read_text()
    assert fasta.startswith(">GAPDH|wildtype")
    manifest = (tmp_path / "GAPDH_manifest.tsv").read_text().strip().splitlines()
    assert manifest[0].split("\t")[:2] == ["protein", "mutant_id"]
    assert len(manifest) == len(art.records) + 1
    assert all(r.protein == "GAPDH" for r in art.records)


def test_compare_consensus_gain_loss(tmp_path):
    wt = tmp_path / "wt.tsv"
    mut = tmp_path / "mut.tsv"
    header = "Protein\tClass\tRegion\tTools\tLength\tn_tools\n"
    wt.write_text(header + "P:42-47\tunanimous\tYMVYMF\tA\t6\t8\n")
    mut.write_text(
        header
        + "P:42-47\tstrong\tYMVYMF\tA\t6\t6\n"
        + "P:100-110\tstrong\tXXXXXXXXXXX\tA\t11\t6\n"
    )
    diff = compare_consensus(wt, mut, protein="P")
    assert [f"{a}-{b}" for a, b in diff.gained] == ["100-110"]
    assert diff.lost == []
    assert diff.residue_delta == 11


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


def test_config_rejects_unknown_mode_and_empty_targets():
    with pytest.raises(MutateConfigError):
        MutagenesisConfig(
            sequence=SequenceSpec(id="x", sequence="ACDEF"),
            targets=TargetSpec(regions=[(1, 3)]),
            modes=["bogus"],
        ).validate()
    with pytest.raises(MutateConfigError):
        MutagenesisConfig(
            sequence=SequenceSpec(id="x", sequence="ACDEF"),
            targets=TargetSpec(),  # nothing set
        ).validate()


def test_mutate_config_roundtrips_from_yaml(tmp_path):
    (tmp_path / "wt.fasta").write_text(">x\nMGKVKVGVNGFGRIGRL\n")
    yaml_text = """
name: rt
sequence: {id: x, fasta: wt.fasta}
targets: {regions: [[3, 8]], positions: [12]}
modes: [amyloid, condensate]
vectors: [activating, inhibiting, neutral, nonsense]
mutation: {orders: [1, 2], max_total: 99, protected_positions: [5]}
"""
    (tmp_path / "c.yaml").write_text(yaml_text)
    cfg = load_mutate_config(tmp_path / "c.yaml")
    assert len(cfg.proteins) == 1
    job = cfg.proteins[0]
    assert job.sequence.sequence.startswith("MGKVK")
    assert cfg.modes == ["amyloid", "condensate"]
    assert cfg.mutation.max_total == 99
    assert job.targets.regions == [(3, 8)]
    assert job.protected_positions == [5]  # legacy mutation.protected_positions folded in


# --------------------------------------------------------------------------- #
# Multiple proteins (panel)
# --------------------------------------------------------------------------- #


def _panel_cfg(**kw) -> MutagenesisConfig:
    seq = _gapdh()
    base = dict(
        proteins=[
            ProteinJob(
                sequence=SequenceSpec(id="GAPDH", sequence=seq),
                targets=TargetSpec(regions=[(42, 47)]),
            ),
            ProteinJob(
                sequence=SequenceSpec(id="MINI", sequence="MGKVVVGGAAALLLFFFKKK"),
                targets=TargetSpec(regions=[(6, 12)]),
            ),
        ],
        modes=["amyloid"],
        vectors=["activating", "inhibiting", "neutral"],
        mutation=MutationSpec(orders=[1], max_per_order=5),
    )
    base.update(kw)
    return MutagenesisConfig(**base)


def test_panel_generates_records_for_every_protein():
    recs = generate(_panel_cfg())
    proteins = {r.protein for r in recs}
    assert proteins == {"GAPDH", "MINI"}
    # Records are labelled and id-prefixed per protein.
    assert all(r.id.startswith(r.protein + "_") for r in recs)


def test_panel_writes_per_protein_and_combined_manifest(tmp_path):
    art = run(_panel_cfg(output_dir=str(tmp_path), name="panel"))
    names = {Path(p).name for p in art.written_files}
    assert "GAPDH_mutants.fasta" in names
    assert "MINI_mutants.fasta" in names
    assert "GAPDH_manifest.tsv" in names
    assert "MINI_manifest.tsv" in names
    assert "panel_manifest.tsv" in names  # combined only when >1 protein
    combined = (tmp_path / "panel_manifest.tsv").read_text().strip().splitlines()
    bodies = [ln.split("\t")[0] for ln in combined[1:]]
    assert set(bodies) == {"GAPDH", "MINI"}
    # Each per-protein FASTA carries its own wild type first.
    assert (tmp_path / "MINI_mutants.fasta").read_text().startswith(">MINI|wildtype")


def test_panel_protected_positions_are_per_protein():
    cfg = _panel_cfg()
    cfg.proteins[1].protected_positions = [8]  # protect a MINI core residue
    recs = generate(cfg)
    mini_positions = {
        s.position
        for r in recs if r.protein == "MINI"
        for s in r.substitutions
    }
    assert 8 not in mini_positions  # protected on MINI
    # GAPDH is unaffected by MINI's protection.
    assert any(r.protein == "GAPDH" for r in recs)


def test_panel_max_total_is_per_protein():
    cfg = _panel_cfg(
        mutation=MutationSpec(orders=[1], max_per_order=100, max_total=4),
    )
    recs = generate(cfg)
    from collections import Counter

    counts = Counter(r.protein for r in recs)
    assert counts["GAPDH"] <= 4
    assert counts["MINI"] <= 4


def test_consensus_protein_auto_defaults_to_sequence_id(tmp_path):
    tsv = tmp_path / "panel_consensus.tsv"
    tsv.write_text(
        "Protein\tClass\tRegion\tTools\tLength\tn_tools\n"
        "GAPDH:42-44\tunanimous\tYMV\tA\t3\t8\n"
        "MINI:6-8\tstrong\tVGG\tA\t3\t6\n"
    )
    # No consensus_protein given -> each job filters to its own id.
    job_g = ProteinJob(
        sequence=SequenceSpec(id="GAPDH", sequence=_gapdh()),
        targets=TargetSpec(consensus_tsv=str(tsv)),
    )
    job_m = ProteinJob(
        sequence=SequenceSpec(id="MINI", sequence="MGKVVVGGAAALLLFFFKKK"),
        targets=TargetSpec(consensus_tsv=str(tsv)),
    )
    assert job_g.targets.consensus_protein == "GAPDH"
    cfg = MutagenesisConfig(proteins=[job_g, job_m], modes=["amyloid"])
    g_pos = {tp.position for tp in resolve_targets(cfg, job_g)}
    m_pos = {tp.position for tp in resolve_targets(cfg, job_m)}
    assert g_pos == {42, 43, 44}
    assert m_pos == {6, 7, 8}


def test_panel_yaml_roundtrip(tmp_path):
    (tmp_path / "a.fasta").write_text(">GAPDH\n" + _gapdh() + "\n")
    yaml_text = """
name: panel
modes: [amyloid, condensate]
vectors: [activating, inhibiting, neutral, nonsense]
mutation: {orders: [1, 2], max_total: 50}
proteins:
  - sequence: {id: GAPDH, fasta: a.fasta}
    targets: {regions: [[42, 47]]}
    protected_positions: [152]
  - sequence: {id: MINI, sequence: MGKVVVGGAAALLLFFFKKK}
    targets: {regions: [[6, 12]], positions: [3]}
"""
    (tmp_path / "panel.yaml").write_text(yaml_text)
    cfg = load_mutate_config(tmp_path / "panel.yaml")
    assert [j.sequence.id for j in cfg.proteins] == ["GAPDH", "MINI"]
    assert cfg.proteins[0].protected_positions == [152]
    assert cfg.proteins[1].targets.regions == [(6, 12)]
    assert cfg.proteins[1].targets.positions == [3]


def test_duplicate_protein_ids_rejected():
    with pytest.raises(MutateConfigError):
        MutagenesisConfig(
            proteins=[
                ProteinJob(SequenceSpec("X", "ACDEFGHIK"), TargetSpec(regions=[(1, 3)])),
                ProteinJob(SequenceSpec("X", "ACDEFGHIK"), TargetSpec(regions=[(1, 3)])),
            ]
        ).validate()
