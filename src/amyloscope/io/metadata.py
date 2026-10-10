"""Display labels for proteins, read from a metadata table.

``ProteinSpec.display_name`` is the single place a figure's axis label comes
from, and for a handful of proteins writing it into the config by hand is right.
For a sweep it is not: a 372-sequence orthologue set cannot carry 372 hand-typed
names, and an axis of accessions makes the reader do a lookup for every row.

So a table supplies them. One row per protein, an ``id`` column matching the
protein ids, and whatever else the table has -- species name, gene name, group,
lineage. The label is a template over those columns, because what belongs on an
axis depends on what the figure is comparing: a sweep across species wants the
species, a comparison of one species' paralogues wants the group, and an axis
that repeats the same word on every row is worse than no label at all.

On the choice itself, since it is a choice and not a formatting detail
-----------------------------------------------------------------------

``9606.ENSP00000350170`` is unique, stable and unreadable. ``Homo sapiens FXR1``
is readable and not guaranteed unique -- a species with two isoforms in the same
group gives two identical labels and the reader cannot tell the rows apart.
Neither works alone, so:

1. **Label by meaning.** The default template is ``{tax_name} {group}``: what a
   reader of a comparative figure is actually comparing.
2. **Fall back down a chain**, so a partly resolved table still labels every
   row: the full template, then the species, then the group, then the bare id.
   A row with no metadata gets its id, never a blank or a guess.
3. **Disambiguate collisions with the id, on every member of the colliding
   set.** A suffix on one row of a pair reads as a correction to that row; on
   both it reads as a property of the pair, which is what it is.
4. **The id stays authoritative.** This module only ever sets
   ``display_name``. Every path lookup, join and provenance record still goes
   through ``ProteinSpec.id``, so a figure becoming readable never costs the
   ability to say which sequence exactly a row was.

The same policy is implemented in ``amyloevo.io.annotate``, which builds these
tables. This module is the consumer: it does not resolve taxonomy, and will not
invent a species name that the table does not carry.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

#: Default template. Fields are the table's own column names.
DEFAULT_TEMPLATE = "{tax_name} {group}"

#: Tried in order when the template renders empty for a row.
FALLBACKS = ("{tax_name} {group}", "{tax_name}", "{group}", "{protein_name}")

#: Column holding the protein id. Must match ``ProteinSpec.id``.
DEFAULT_ID_COLUMN = "id"

#: Column preferred for the disambiguating suffix, when the table has it.
#: Inside a colliding set the species is identical by construction, so a full
#: ``9606.ENSP00000350170`` spends axis width on a taxid prefix that is the same
#: on every member; the accession alone separates them. Falls back to the id.
#: ``amyloevo.io.annotate`` makes the same choice, and a test in each package
#: pins the two together -- two implementations of one policy that disagree on
#: the output are worse than one implementation that is merely imperfect.
SUFFIX_COLUMN = "protein"

_FIELD = re.compile(r"\{(\w+)\}")


class MetadataError(ValueError):
    """The table cannot be used to label proteins."""


def _sniff(path: Path) -> str:
    """Delimiter from the extension, falling back to a sniff of the header.

    A .tsv read as comma-separated yields one column whose name is the whole
    header line, and then every label silently falls back to the id -- a
    failure that looks like missing metadata rather than a parsing mistake.
    """
    suffix = path.suffix.lower()
    if suffix in (".tsv", ".tab", ".txt"):
        return "\t"
    if suffix == ".csv":
        return ","
    header = path.read_text(encoding="utf-8").split("\n", 1)[0]
    return "\t" if header.count("\t") >= header.count(",") else ","


def read_table(path: str | Path, *, delimiter: str | None = None) -> list[dict]:
    """Read a metadata table as a list of row dicts."""
    path = Path(path)
    if not path.is_file():
        raise MetadataError(
            f"metadata table not found: {path}. It is one row per protein with "
            f"an '{DEFAULT_ID_COLUMN}' column matching the protein ids; "
            f"amyloevo's `rule taxonomy_table` writes one."
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(
            handle, delimiter=delimiter or _sniff(path)
        )
        if not reader.fieldnames:
            raise MetadataError(f"{path}: no header row")
        return [dict(row) for row in reader]


def render(template: str, row: dict) -> str:
    """Render one template against a row, dropping fields the row lacks.

    A missing or empty field becomes nothing rather than raising or leaving
    ``{tax_name}`` visible on an axis, and the surrounding whitespace is
    collapsed so a dropped field does not leave a gap.
    """
    def substitute(match: re.Match) -> str:
        value = row.get(match.group(1), "")
        return "" if value is None else str(value).strip()

    return " ".join(_FIELD.sub(substitute, template).split())


def build_labels(
    rows,
    *,
    template: str = DEFAULT_TEMPLATE,
    id_column: str = DEFAULT_ID_COLUMN,
    fallbacks=FALLBACKS,
) -> tuple[dict[str, str], int]:
    """Map protein id -> display label. Returns ``(labels, n_disambiguated)``.

    Collisions are resolved by appending the id to EVERY member of the
    colliding set.
    """
    rows = list(rows)
    if rows and id_column not in rows[0]:
        raise MetadataError(
            f"metadata table has no '{id_column}' column; columns are "
            f"{sorted(rows[0])}. Without it there is nothing to join labels on."
        )

    drafted: dict[str, str] = {}
    suffixes: dict[str, str] = {}
    for row in rows:
        pid = str(row.get(id_column, "")).strip()
        if not pid:
            continue
        label = render(template, row)
        if not label:
            for candidate in fallbacks:
                label = render(candidate, row)
                if label:
                    break
        drafted[pid] = label or pid
        suffixes[pid] = str(row.get(SUFFIX_COLUMN, "") or "").strip() or pid

    counts: dict[str, int] = {}
    for label in drafted.values():
        counts[label] = counts.get(label, 0) + 1

    labels: dict[str, str] = {}
    disambiguated = 0
    for pid, label in drafted.items():
        if counts[label] > 1 and label != pid:
            labels[pid] = f"{label} [{suffixes[pid]}]"
            disambiguated += 1
        else:
            labels[pid] = label
    return labels, disambiguated


def apply_labels(proteins, labels: dict[str, str]):
    """Set ``display_name`` on every protein the table covers.

    A protein already carrying an explicit ``display_name`` keeps it: a name
    written in the config is a deliberate statement and a table should not
    silently overrule it. Returns ``(n_labelled, unlabelled_ids)``.
    """
    labelled = 0
    missing: list[str] = []
    for protein in proteins:
        if protein.display_name:
            continue
        label = labels.get(protein.id)
        if label and label != protein.id:
            protein.display_name = label
            labelled += 1
        else:
            missing.append(protein.id)
    return labelled, missing


def load_labels(
    path: str | Path,
    *,
    template: str = DEFAULT_TEMPLATE,
    id_column: str = DEFAULT_ID_COLUMN,
    delimiter: str | None = None,
) -> tuple[dict[str, str], int]:
    """Read a table and build the label map from it."""
    return build_labels(
        read_table(path, delimiter=delimiter),
        template=template,
        id_column=id_column,
    )
