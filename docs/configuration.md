# Configuration

An `amyloscope` analysis is fully specified by a YAML file with four blocks:
`proteins`, `tools`, `consensus`, and `viz`. Only `proteins` and `tools` are
required; the rest carry defaults. The file is parsed, every field is validated,
and a malformed config raises `ConfigError` with a located message before any
work begins.

```yaml
name: my_analysis          # optional; defaults to the file stem
output_dir: results        # where artifacts are written
hypothesis: >              # optional free text, surfaced in the statistics report
  Aggregation-prone regions are expected to coincide with ...

proteins: [...]
tools: [...]
consensus: {...}
viz: {...}
```

## `proteins`

The protein panel. Each entry:

| Key            | Required | Meaning                                                                 |
| -------------- | -------- | ----------------------------------------------------------------------- |
| `id`           | yes      | Stable identifier; resolves `{protein}` in tool paths (UniProt accession or gene symbol). |
| `display_name` | no       | Figure label. Falls back to `id`. This is the single configured replacement for the old triplicated `RPS2 -> eS2` maps. |
| `length`       | no       | Residue count. If omitted, inferred as the maximum residue index across loaded tracks. |
| `sequence`     | no       | One-letter sequence; when present it is the authoritative source for region-sequence extraction. |
| `domains`      | no       | List of annotated modules for overlap analysis.                         |

Each domain:

```yaml
domains:
  - {name: NAC, start: 61, stop: 95, category: low_complexity}
```

`start`/`stop` are 1-based inclusive. The `category` is an **abstract**
functional label, not a fold name — this is what decouples rendering from any
one protein family. The palette (below) maps categories to colours; an
unrecognised category falls back to a neutral grey. Recognised category
substrings include `rna_binding`, `dna_binding`, `structured_core`,
`catalytic`, `disordered`, `low_complexity`, `linker`, `localization`,
`regulatory`, `terminal`, and `conserved`.

!!! tip
    If a curated `sequence` is given, its length must equal a declared `length`;
    the validator rejects the inconsistency rather than guessing.

## `tools`

The predictor panel. Each entry names the **adapter** that parses the tool's
native output and the **detection strategy** that converts the parsed track
into a per-residue APR mask.

```yaml
tools:
  - name: Aggrescan
    adapter: aggrescan                       # registry key (see `amyloscope adapters`)
    path: data/aggrescan/{protein}.csv       # {protein} is replaced by each protein id
    color: "#1B9E77"                         # track/legend colour
    enabled: true                            # default true; set false to drop without deleting
    detection: {method: notnull, column: Prediction}
```

### Detection strategies

| `method`  | APR condition                          | Reads               | Notes                                        |
| --------- | -------------------------------------- | ------------------- | -------------------------------------------- |
| `above`   | `column >= threshold`                  | `column`, `threshold` | Continuous propensities / hydrophobicity.  |
| `below`   | `column <= threshold`                  | `column`, `threshold` | **Free-energy scores** (lower = more stable cross-beta). |
| `flag`    | value in `flag_true_values`            | `column`, `flag_true_values` | Accepts booleans and string flags alike. |
| `notnull` | value is populated (non-empty)         | `column`            | For tools that emit a label only inside hotspots. |
| `nonzero` | `value != 0`                           | `column`            | For matrices that zero non-hit positions.    |
| `present` | every parsed row is an APR             | —                   | For interval/region-list outputs.            |

`above` and `below` require a numeric `threshold`; `flag` requires a non-empty
`flag_true_values`. These constraints are validated.

The distinction between `above` and `below` is mechanistically load-bearing.
Energy-based predictors such as PASTA 2.0 report a pairing free energy in which
*more negative* values indicate *more stable* cross-beta structure. Calling
those with `above` would invert the biology and flag the least aggregation-prone
residues. The renderer is aware of this too: tracks scored with `below` are
drawn with an inverted y-axis so that "up" always means "more aggregation-prone".

## `consensus`

Governs how per-tool APRs are reconciled across the panel.

```yaml
consensus:
  tiers:
    - {name: unanimous, min_fraction: 1.0,  color: "#B2182B"}
    - {name: strong,    min_fraction: 0.75, color: "#2166AC"}
    - {name: moderate,  min_fraction: 0.5,  color: "#FDAE61"}
  min_region_length: 5
  coverage_fraction: 0.8
  overlap_resolution_max: 0.3
  denominator: configured
```

| Key                      | Default | Meaning                                                                 |
| ------------------------ | ------- | ----------------------------------------------------------------------- |
| `tiers`                  | 1.0 / 0.75 / 0.5 | Evidence tiers as **fractions** of the active predictor set. A region is found at the weakest tier and classified into the strongest tier its support satisfies. |
| `min_region_length`      | 5       | Minimum APR length in residues — the steric-zipper spine floor.         |
| `coverage_fraction`      | 0.8     | A predictor counts toward a consensus window only if its own call spans at least this fraction of the window, preventing single-residue clipping from inflating apparent agreement. |
| `overlap_resolution_max` | 0.3     | Maximum positional overlap permitted between two retained consensus regions during de-duplication. |
| `denominator`            | `configured` | `configured` resolves tier counts against the full enabled tool set (stable thresholds across proteins); `available` resolves against the tools that returned data for each protein (tolerant of missing inputs). |

**Fractional tiers are the central generalisation.** The original pipeline
hardcoded `unanimous == 8`, `strong >= 6`, `moderate >= 4`, which assumes
exactly eight predictors. Expressing these as 1.0 / 0.75 / 0.5 reproduces those
exact counts at a panel of eight, but the scheme self-adjusts: at six active
predictors the floors become 6/6, 5/6, 3/6 (`ceil(fraction × n)`, floored at 1).
The meaningful quantity is the *proportion* of orthogonal models that converge,
which is only interpretable relative to how many were run.

The overlap-resolution step retains the strongest of two competing regions,
preferring more supporting tools, then greater length, then earlier start.

## `viz`

Rendering options.

```yaml
viz:
  dpi: 300
  save_svg: true
  font_family: Arial
  x_padding: 15
  domain_palette:            # extend or override the default category -> colour map
    my_category: "#123456"
```

| Key              | Default | Meaning                                              |
| ---------------- | ------- | ---------------------------------------------------- |
| `dpi`            | 300     | Raster resolution for PNG output.                    |
| `save_svg`       | true    | Also emit a vector SVG beside each PNG.               |
| `font_family`    | Arial   | Figure font.                                         |
| `x_padding`      | 15      | Residues of padding past protein length on the x-axis. |
| `domain_palette` | built-in | Category → colour. Your keys are merged over the defaults; matching is case-insensitive substring, first match wins. |

## Validation

`load_config` validates structure and cross-field consistency: non-empty
protein ids, at least one enabled tool, unique tool names and protein ids,
valid domain intervals, valid detection methods with their required parameters,
tier fractions in `(0, 1]`, and a `coverage_fraction` in `(0, 1]`. Use the CLI
to check a config without running it:

```bash
amyloscope validate config.yaml
```
