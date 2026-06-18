# Adding a predictor

Supporting a new aggregation predictor is a two-step change — register a parser,
then reference it in the config. No core module is edited. This adapter registry
is the seam that replaces the original `if tool == "..."` parsing ladder, where
adding a tool meant editing the loader's control flow.

## The normalised-track contract

Every adapter parses one predictor's native output into a *normalised track*: a
`pandas.DataFrame` carrying at minimum three columns, plus any auxiliary columns
the detection strategy reads.

| Column    | Type  | Meaning                                              |
| --------- | ----- | ---------------------------------------------------- |
| `Number`  | int   | 1-based residue index.                               |
| `Residue` | str   | One-letter code; may be `""` when the format omits it. |
| `Score`   | float | Per-residue score (may be `NaN` where undefined).    |

Auxiliary columns are whatever your detection strategy names — for example a
boolean `is_hotspot`, a string `Fold` flag, or a `Prediction` label. The
detection strategy declared in the config decides which column is read and how:
`above`/`below` read `Score` (or a named numeric column), `flag` reads a named
flag column, `notnull`/`nonzero` test a named column, and `present` ignores
columns entirely (every emitted row is an APR).

The helper `amyloscope.io.adapters._finalise` enforces the contract — it checks
the required columns are present, coerces `Number` to integer and `Score` to
numeric, strips `Residue`, drops rows with an unparseable index, and sorts by
residue. Run your parsed frame through it and downstream layers will accept the
track.

## Step 1 — register a parser

```python
from pathlib import Path
import pandas as pd

from amyloscope import register_adapter
from amyloscope.io.adapters import _finalise


@register_adapter("mytool")
def parse_mytool(path: Path) -> pd.DataFrame:
    """Parse MyTool's per-residue CSV into a normalised track."""
    df = pd.read_csv(path)                     # whatever the native format is
    df = df.rename(columns={"pos": "Number", "aa": "Residue", "prop": "Score"})
    return _finalise(df, path)
```

The decorator registers the parser under a case-insensitive key. Importing the
module that defines it is enough to register it; if you keep your adapters in a
separate module, import that module before running the pipeline (the built-in
adapters are registered the same way, by importing `amyloscope.io.adapters`).

For an interval/region-list format rather than a per-residue one, expand each
interval to per-residue rows and pair it with `detection: {method: present}` —
see the bundled `archcandy` adapter for a worked example, including defensive
handling when the residue string and coordinate span disagree in length.

## Step 2 — reference it in the config

```yaml
tools:
  - name: MyTool
    adapter: mytool                       # the key you registered
    path: data/mytool/{protein}.csv
    color: "#444444"
    detection: {method: above, column: Score, threshold: 0.5}
```

That is the whole change. The tool now participates in loading, APR calling,
consensus, statistics, domain overlap, and the per-protein track figure, with
no edits to any core file.

## Choosing a detection strategy

Match the primitive to how the tool encodes a hit:

- The tool outputs a **continuous propensity** and you have a cutoff → `above`
  (or `below` if the scale is inverted, as with a free energy).
- The tool outputs an explicit **per-residue flag** → `flag` with the flag's
  true value(s) in `flag_true_values`.
- The tool emits a **label only inside hotspots** and leaves it blank elsewhere
  → `notnull` on that column.
- The tool **zeroes** non-hit positions → `nonzero`.
- The tool reports **hit intervals only** → expand to rows and use `present`.

If a strategy needs a numeric `threshold` (`above`/`below`) or
`flag_true_values` (`flag`), the config validator enforces it, so a
mis-specified tool fails fast at load time rather than producing a silently
empty track.

## Verifying a new adapter

List the registry to confirm registration:

```bash
amyloscope adapters
```

Then validate and dry-run a config that references the new tool:

```bash
amyloscope validate config.yaml
amyloscope run config.yaml
```

A parser that raises, or whose output is missing a required column, is reported
as a per-tool load failure (a warning) rather than aborting the run — the
remaining predictors still produce a consensus, so a single broken adapter
degrades gracefully.
