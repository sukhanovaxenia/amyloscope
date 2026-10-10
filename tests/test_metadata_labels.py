"""Display labels from a metadata table, and the policy behind them."""

from __future__ import annotations

import pytest

from amyloscope.config import ConfigError, ProteinSpec, load_config
from amyloscope.io.metadata import (
    MetadataError,
    apply_labels,
    build_labels,
    load_labels,
    read_table,
    render,
)

ROWS = [
    {"id": "9606.ENSP1", "tax_name": "Homo sapiens", "group": "FXR1",
     "protein_name": "FXR1"},
    {"id": "10090.ENSP2", "tax_name": "Mus musculus", "group": "FXR1",
     "protein_name": "Fxr1"},
    {"id": "7227.FBpp3", "tax_name": "Drosophila melanogaster", "group": "",
     "protein_name": "Fmr1"},
]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_a_missing_field_leaves_no_gap_and_no_braces() -> None:
    assert render("{tax_name} {group}", {"tax_name": "Homo sapiens"}) == "Homo sapiens"
    assert render("{a} {b}", {}) == ""


def test_whitespace_is_collapsed_not_merely_stripped() -> None:
    row = {"tax_name": "Homo sapiens", "group": "", "x": "FXR1"}
    assert render("{tax_name} {group} {x}", row) == "Homo sapiens FXR1"


def test_a_none_value_is_treated_as_absent() -> None:
    assert render("{tax_name} {group}", {"tax_name": None, "group": "FXR1"}) == "FXR1"


# --------------------------------------------------------------------------- #
# Building the map
# --------------------------------------------------------------------------- #
def test_the_default_label_is_species_plus_group() -> None:
    labels, _ = build_labels(ROWS)
    assert labels["9606.ENSP1"] == "Homo sapiens FXR1"


def test_the_fallback_chain_still_labels_a_partial_row() -> None:
    """A row with no group gets the species, not a blank or a guess."""
    labels, _ = build_labels(ROWS)
    assert labels["7227.FBpp3"] == "Drosophila melanogaster"


def test_a_row_with_nothing_usable_gets_its_id() -> None:
    labels, _ = build_labels([{"id": "x", "tax_name": "", "group": ""}])
    assert labels["x"] == "x"


def test_collisions_are_marked_on_every_member_of_the_set() -> None:
    """A suffix on one row reads as a correction to that row.

    On both it reads as a property of the pair, which is what it is.
    """
    rows = [
        {"id": "9606.A", "tax_name": "Homo sapiens", "group": "FXR1"},
        {"id": "9606.B", "tax_name": "Homo sapiens", "group": "FXR1"},
        {"id": "10090.C", "tax_name": "Mus musculus", "group": "FXR1"},
    ]
    labels, n = build_labels(rows)
    assert labels["9606.A"] == "Homo sapiens FXR1 [9606.A]"
    assert labels["9606.B"] == "Homo sapiens FXR1 [9606.B]"
    assert labels["10090.C"] == "Mus musculus FXR1"
    assert n == 2


def test_the_suffix_uses_the_accession_when_the_table_carries_one() -> None:
    """Inside a colliding set the species is identical by construction.

    So a full `9606.ENSP00000350170` spends axis width on a taxid prefix that is
    the same on every member of the set. amyloevo.io.annotate makes the same
    choice; the next test pins the two together.
    """
    rows = [
        {"id": "9606.ENSP1", "protein": "ENSP1",
         "tax_name": "Homo sapiens", "group": "FXR1"},
        {"id": "9606.ENSP2", "protein": "ENSP2",
         "tax_name": "Homo sapiens", "group": "FXR1"},
    ]
    labels, _ = build_labels(rows)
    assert labels["9606.ENSP1"] == "Homo sapiens FXR1 [ENSP1]"
    assert labels["9606.ENSP2"] == "Homo sapiens FXR1 [ENSP2]"


def test_this_matches_amyloevos_implementation_of_the_same_policy() -> None:
    """Two implementations of one policy that disagree are worse than one.

    amyloevo builds these tables and amyloscope consumes them, so a figure
    labelled by either must read the same. This caught a real divergence: the
    suffix was the full id here and the accession there, differing on 52 of the
    372 FXR1 rows.
    """
    amyloevo = pytest.importorskip("amyloevo.io.annotate")
    rows = [
        {"id": "9606.ENSP1", "protein": "ENSP1", "taxid": 9606,
         "tax_name": "Homo sapiens", "group": "FXR1", "protein_name": "FXR1"},
        {"id": "9606.ENSP2", "protein": "ENSP2", "taxid": 9606,
         "tax_name": "Homo sapiens", "group": "FXR1", "protein_name": "FXR1"},
        {"id": "10090.ENSP3", "protein": "ENSP3", "taxid": 10090,
         "tax_name": "Mus musculus", "group": "FXR1", "protein_name": "Fxr1"},
        {"id": "7227.FBpp4", "protein": "FBpp4", "taxid": 7227,
         "tax_name": "Drosophila melanogaster", "group": "", "protein_name": "Fmr1"},
    ]
    theirs, their_n = amyloevo.make_labels(
        rows, template=amyloevo.DEFAULT_LABEL_TEMPLATE
    )
    ours, our_n = build_labels(rows)
    assert [ours[r["id"]] for r in rows] == theirs
    assert our_n == their_n


def test_a_row_labelled_by_its_own_id_is_not_then_suffixed_with_it() -> None:
    rows = [{"id": "x", "tax_name": "", "group": ""},
            {"id": "y", "tax_name": "", "group": ""}]
    labels, n = build_labels(rows)
    assert labels == {"x": "x", "y": "y"}
    assert n == 0


def test_the_template_is_configurable() -> None:
    """One species' paralogues want the group; one group's orthologues the species."""
    labels, _ = build_labels(ROWS, template="{tax_name}")
    assert labels["10090.ENSP2"] == "Mus musculus"
    assert labels["9606.ENSP1"] == "Homo sapiens"

    one_species = [
        {"id": "9606.A", "tax_name": "Homo sapiens", "group": "FXR1"},
        {"id": "9606.B", "tax_name": "Homo sapiens", "group": "FMR1"},
    ]
    labels, n = build_labels(one_species, template="{group}")
    assert labels == {"9606.A": "FXR1", "9606.B": "FMR1"}
    assert n == 0


def test_a_template_too_coarse_for_the_set_collides_rather_than_misleads() -> None:
    """Two FXR1 orthologues under "{group}" are both "FXR1".

    The policy does not silently let that stand: the rows are disambiguated, so
    the axis is ugly in a way that tells the reader the template is wrong for
    this figure, rather than readable and wrong.
    """
    labels, n = build_labels(ROWS, template="{group}")
    assert labels["9606.ENSP1"] == "FXR1 [9606.ENSP1]"
    assert labels["10090.ENSP2"] == "FXR1 [10090.ENSP2]"
    assert n == 2


def test_a_table_without_the_id_column_is_refused_by_name() -> None:
    with pytest.raises(MetadataError, match="no 'id' column"):
        build_labels([{"accession": "x", "tax_name": "Homo sapiens"}])


def test_rows_with_a_blank_id_are_skipped_not_collapsed_into_one() -> None:
    labels, _ = build_labels([{"id": "", "tax_name": "A"}, {"id": "k", "tax_name": "B"}])
    assert labels == {"k": "B"}


# --------------------------------------------------------------------------- #
# Reading a file
# --------------------------------------------------------------------------- #
def test_a_tsv_is_read_as_tab_separated(tmp_path) -> None:
    """A .tsv parsed as CSV gives one column named after the whole header.

    Every label then falls back to the id, which looks like missing metadata
    rather than a parsing mistake -- so the delimiter is taken from the suffix.
    """
    path = tmp_path / "meta.tsv"
    path.write_text("id\ttax_name\tgroup\n9606.A\tHomo sapiens\tFXR1\n")
    labels, _ = load_labels(path)
    assert labels["9606.A"] == "Homo sapiens FXR1"


def test_a_csv_is_read_as_comma_separated(tmp_path) -> None:
    path = tmp_path / "meta.csv"
    path.write_text("id,tax_name,group\n9606.A,Homo sapiens,FXR1\n")
    labels, _ = load_labels(path)
    assert labels["9606.A"] == "Homo sapiens FXR1"


def test_an_unsuffixed_file_is_sniffed_from_its_header(tmp_path) -> None:
    path = tmp_path / "meta"
    path.write_text("id\ttax_name\n9606.A\tHomo sapiens\n")
    assert read_table(path)[0]["tax_name"] == "Homo sapiens"


def test_a_missing_table_names_what_it_should_contain(tmp_path) -> None:
    with pytest.raises(MetadataError, match="one row per protein"):
        read_table(tmp_path / "absent.tsv")


# --------------------------------------------------------------------------- #
# Applying to proteins
# --------------------------------------------------------------------------- #
def test_labels_land_on_display_name() -> None:
    proteins = [ProteinSpec(id="9606.ENSP1"), ProteinSpec(id="10090.ENSP2")]
    labels, _ = build_labels(ROWS)
    n, missing = apply_labels(proteins, labels)
    assert n == 2 and missing == []
    assert proteins[0].label == "Homo sapiens FXR1"


def test_an_explicit_display_name_is_not_overruled() -> None:
    """A name written in the config is a statement, not a placeholder."""
    proteins = [ProteinSpec(id="9606.ENSP1", display_name="human FXR1")]
    labels, _ = build_labels(ROWS)
    apply_labels(proteins, labels)
    assert proteins[0].label == "human FXR1"


def test_a_protein_absent_from_the_table_is_reported_not_guessed() -> None:
    proteins = [ProteinSpec(id="ghost")]
    n, missing = apply_labels(proteins, build_labels(ROWS)[0])
    assert n == 0 and missing == ["ghost"]
    assert proteins[0].label == "ghost"


# --------------------------------------------------------------------------- #
# Through the config
# --------------------------------------------------------------------------- #
def _write_config(tmp_path, labels_block: str) -> "object":
    (tmp_path / "meta.tsv").write_text(
        "id\ttax_name\tgroup\n"
        "P1\tHomo sapiens\tFXR1\n"
        "P2\tMus musculus\tFXR1\n"
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: test\n"
        "proteins:\n"
        "  - id: P1\n"
        "  - id: P2\n"
        "tools:\n"
        "  - name: appnn\n"
        "    adapter: appnn\n"
        "    path: '{protein}_appnn.csv'\n"
        "    enabled: true\n"
        "    detection:\n"
        "      kind: threshold\n"
        "      column: score\n"
        "      threshold: 0.5\n"
        f"{labels_block}"
    )
    return path


def test_the_config_picks_labels_up_from_a_relative_path(tmp_path) -> None:
    path = _write_config(tmp_path, "labels:\n  table: meta.tsv\n")
    cfg = load_config(path)
    assert cfg.label_for("P1") == "Homo sapiens FXR1"
    assert cfg.label_for("P2") == "Mus musculus FXR1"


def test_no_labels_section_leaves_the_ids_alone(tmp_path) -> None:
    cfg = load_config(_write_config(tmp_path, ""))
    assert cfg.label_for("P1") == "P1"


def test_a_labels_section_without_a_table_says_how_to_opt_out(tmp_path) -> None:
    path = _write_config(tmp_path, "labels:\n  template: '{group}'\n")
    with pytest.raises(ConfigError, match="Drop the whole 'labels' section"):
        load_config(path)


def test_require_all_refuses_a_partially_covered_set(tmp_path) -> None:
    """Half the axis in species names and half in accessions is the worst case."""
    (tmp_path / "meta.tsv").write_text("id\ttax_name\tgroup\nP1\tHomo sapiens\tFXR1\n")
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: test\n"
        "proteins:\n  - id: P1\n  - id: P2\n"
        "tools:\n  - name: appnn\n    adapter: appnn\n"
        "    path: '{protein}_appnn.csv'\n    enabled: true\n    detection:\n"
        "      kind: threshold\n      column: score\n      threshold: 0.5\n"
        "labels:\n  table: meta.tsv\n  require_all: true\n"
    )
    with pytest.raises(ConfigError, match="have no row in"):
        load_config(path)
