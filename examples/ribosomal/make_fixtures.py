#!/usr/bin/env python3
"""Generate small, format-correct synthetic predictor outputs for the example.

These are NOT real predictions. They exist so that ``amyloscope run`` on the
shipped example config produces output out-of-the-box, and so the package has a
deterministic end-to-end smoke target. Each writer emits exactly the native
layout the corresponding adapter parses (see ``amyloscope.io.adapters``).

For every protein we plant the same handful of aggregation-prone windows into
most predictors so that cross-tool consensus has something to converge on, with
a couple of tool-specific idiosyncrasies (a private hit, a dropped window) to
keep the consensus tiers non-degenerate.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

AA = "ACDEFGHIKLMNPQRSTVWY"

# protein id -> (length, list of "true" APR windows planted across tools)
PROTEINS: dict[str, tuple[int, list[tuple[int, int]]]] = {
    "RPS2": (120, [(18, 30), (61, 72), (95, 104)]),
    "RPL36": (105, [(12, 22), (55, 66), (88, 97)]),
    "RPS28": (69, [(9, 19), (40, 50)]),
    "RPL27": (110, [(15, 26), (70, 81)]),
}

# A few tool-specific perturbations so tiers aren't all unanimous:
#   - "private" extra windows that only some tools see
#   - "missing" windows a given tool fails to call
PRIVATE = {"waltz": (3, 8)}          # one tool sees an extra short hit
DROP = {"archcandy": -1}              # archcandy drops each protein's last window


def seq_for(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(AA) for _ in range(length))


def in_any(pos: int, windows: list[tuple[int, int]]) -> bool:
    return any(a <= pos <= b for a, b in windows)


def windows_for(tool: str, base: list[tuple[int, int]]) -> list[tuple[int, int]]:
    w = list(base)
    if tool in DROP:
        idx = DROP[tool]
        w = w[:idx] if idx < 0 else [x for i, x in enumerate(w) if i != idx]
    if tool in PRIVATE:
        w = [PRIVATE[tool], *w]
    return w


def write_aggrescan(p: str, length: int, seq: str, win, path: Path) -> None:
    # header=1: line 0 skipped, line 1 consumed as header, then 7-col rows.
    lines = ["# AGGRESCAN3D per-residue output (synthetic)",
             "Position,Residue,a3v,HSA,NHSA,a4vAHS,Prediction"]
    for i, aa in enumerate(seq, start=1):
        apr = in_any(i, win)
        a3v = round(0.8 if apr else -0.6, 3)
        pred = "HotSpot" if apr else ""
        lines.append(f"{i},{aa},{a3v},{a3v:.3f},{-a3v:.3f},{a3v:.3f},{pred}")
    path.write_text("\n".join(lines) + "\n")


def write_appnn(p: str, length: int, seq: str, win, path: Path) -> None:
    # TSV, header row; columns id/position/residue/score/is_hotspot/overall.
    rows = ["id\tposition\tresidue\tscore\tis_hotspot\toverall"]
    for i, aa in enumerate(seq, start=1):
        apr = in_any(i, win)
        score = round(0.85 if apr else 0.15, 3)
        rows.append(f"{p}\t{i}\t{aa}\t{score}\t{apr}\t0.5")
    path.write_text("\n".join(rows) + "\n")


def write_foldamyloid(p: str, length: int, seq: str, win, path: Path) -> None:
    # CSV, header row; columns number/residue/fold/score; APR => fold == 'f'.
    rows = ["number,residue,fold,score"]
    for i, aa in enumerate(seq, start=1):
        apr = in_any(i, win)
        rows.append(f"{i},{aa},{'f' if apr else '-'},{21.4 if apr else 12.0}")
    path.write_text("\n".join(rows) + "\n")


def write_pasta2(p: str, length: int, seq: str, win, path: Path) -> None:
    # one energy per line; lower (more negative) = more aggregation-prone.
    vals = []
    for i in range(1, length + 1):
        vals.append(f"{-4.2 if in_any(i, win) else -0.3:.3f}")
    path.write_text("\n".join(vals) + "\n")


def write_waltz(p: str, length: int, seq: str, win, path: Path) -> None:
    # TSV number<TAB>score; 0 outside matrix hits, nonzero inside.
    rows = []
    for i in range(1, length + 1):
        rows.append(f"{i}\t{96.0 if in_any(i, win) else 0.0:.1f}")
    path.write_text("\n".join(rows) + "\n")


def write_aggreprot(p: str, length: int, seq: str, win, path: Path) -> None:
    # skiprows=1 then header row; drop struct_position/sasa/transmembrane.
    rows = ["# AggreProt synthetic output",
            "position,residue,score,struct_position,sasa,transmembrane"]
    for i, aa in enumerate(seq, start=1):
        apr = in_any(i, win)
        rows.append(f"{i},{aa},{0.82 if apr else 0.10:.3f},C,0.3,0")
    path.write_text("\n".join(rows) + "\n")


def write_crossbeta(p: str, length: int, seq: str, win, path: Path) -> None:
    # JSON: {id: [ {AA_list:[{Number,Residue,Score,score_list}, ...]} ]}
    aa_list = []
    for i, aa in enumerate(seq, start=1):
        apr = in_any(i, win)
        aa_list.append(
            {"Number": i, "Residue": aa, "Score": 0.9 if apr else 0.2,
             "score_list": []}
        )
    json.dump({p: [{"AA_list": aa_list}]}, path.open("w"))


def write_archcandy(p: str, length: int, seq: str, win, path: Path) -> None:
    # region list: cols id,sequence,rank,start,stop,score (present => APR).
    rows = ["id,sequence,rank,start,stop,score"]
    for k, (a, b) in enumerate(win, start=1):
        frag = seq[a - 1:b]
        rows.append(f"{p}_{k},{frag},{k},{a},{b},{0.6 + 0.05 * k:.2f}")
    path.write_text("\n".join(rows) + "\n")


WRITERS = {
    "aggrescan": ("csv", write_aggrescan),
    "appnn": ("tsv", write_appnn),
    "foldamyloid": ("csv", write_foldamyloid),
    "pasta2": ("txt", write_pasta2),
    "waltz": ("tsv", write_waltz),
    "aggreprot": ("csv", write_aggreprot),
    "crossbeta": ("json", write_crossbeta),
    "archcandy": ("csv", write_archcandy),
}


def main() -> None:
    rng = random.Random(1234)
    for tool, (ext, writer) in WRITERS.items():
        (DATA / tool).mkdir(parents=True, exist_ok=True)
        for p, (length, base) in PROTEINS.items():
            seq = seq_for(length, rng)
            win = windows_for(tool, base)
            writer(p, length, seq, win, DATA / tool / f"{p}.{ext}")
    print(f"wrote fixtures for {len(PROTEINS)} proteins x {len(WRITERS)} tools "
          f"under {DATA}")


if __name__ == "__main__":
    main()
