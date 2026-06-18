# amyloscope

Consensus meta-scoring, domain mapping and visualisation of protein
amyloidogenicity predictions.

`amyloscope` is the post-processing layer that sits *downstream* of the
sequence-based aggregation predictors. It does not predict aggregation itself;
it ingests the per-residue tracks those tools emit, calls per-predictor
aggregation-prone regions (APRs), builds a fractional cross-predictor consensus,
maps the consensus onto a user-defined domain architecture, and renders
publication figures and statistical reports.

The package is **configuration-driven and protein-agnostic**. The analysis is
specified entirely in a YAML file — the predictor panel, their output paths and
APR-calling rules, the protein panel with its domains, and the consensus model.
Pointed at a different config, the same code analyses any protein, not only the
ribosomal panel it was originally written for.

## Why a consensus layer

No single aggregation predictor is authoritative. The published tools encode
mechanistically distinct hypotheses about what drives cross-beta assembly:
hydrophobic-cluster burial (Aggrescan), beta-pairing free energy (PASTA 2.0),
position-specific amyloid matrices (Waltz), packing of beta-arch motifs
(ArchCandy), and learned per-residue propensity (APPNN, AggreProt, CrossBeta).
Each has characteristic false positives. A residue window that is independently
flagged by several of these *orthogonal* models is therefore far more credibly
an intrinsic aggregation determinant than one called by any single tool — this
is the standard rationale for consensus amyloid prediction
([Prabakaran et al., 2021, *Brief. Bioinform.*](https://doi.org/10.1093/bib/bbab067)).

`amyloscope` makes that consensus the central object, and — importantly —
expresses agreement as a **fraction** of the predictors actually run rather than
an absolute count, so the evidence tiers are invariant to how many tools are in
the panel.

## Installation

```bash
git clone https://github.com/xenia/amyloscope
cd amyloscope
pip install -e ".[viz]"      # core + matplotlib/seaborn for figures
```

Optional extras: `.[docs]` for the documentation site, `.[dev]` for the test
and lint toolchain.

## Quick start

The shipped example reproduces a four-protein ribosomal analysis against
synthetic inputs, so it runs out of the box:

```bash
python examples/ribosomal/make_fixtures.py     # write synthetic predictor outputs
amyloscope run examples/ribosomal/config.yaml  # full pipeline -> examples/ribosomal/output/
```

Or from Python:

```python
from amyloscope import run_from_file

artifacts = run_from_file("examples/ribosomal/config.yaml")
for region in artifacts.consensus.all_regions():
    print(region.as_dict())
```

The composable stages are also public, if you want the intermediate objects
rather than the on-disk artifacts:

```python
from amyloscope import load_config, load_dataset, compute_consensus

cfg = load_config("config.yaml")
dataset = load_dataset(cfg)          # parse every predictor track for every protein
result = compute_consensus(dataset)  # fractional consensus regions per protein
```

## Command line

```
amyloscope run       CONFIG.yaml      # load, run, write all artifacts
amyloscope validate  CONFIG.yaml      # parse and validate the config, report panel size
amyloscope adapters                   # list registered predictor adapters
```

## What it produces

Running a config writes, into `output_dir`:

| Artifact                      | Contents                                                        |
| ----------------------------- | --------------------------------------------------------------- |
| `consensus_regions.tsv`       | Every consensus APR: protein, span, sequence, supporting tools, evidence tier |
| `consensus_statistics.txt`    | Tier distribution, length analysis (steric-zipper scale), positional enrichment (chi-square vs. uniform), and the configured hypothesis |
| `domain_overlap.txt`          | APR residues inside vs. outside each annotated domain, per protein and globally |
| `consensus_distribution.png/.svg` | Browser-style per-protein consensus map, stratified by tier  |
| `positional_enrichment.png/.svg`  | Decile histogram of consensus positions with chi-square test |
| `domain_architecture.png/.svg`    | Domain layout with consensus APRs overlaid                   |
| `<protein>_tracks.png/.svg`       | Per-predictor raw profiles for each protein                  |

## The configuration model

A config has four blocks: `proteins`, `tools`, `consensus`, and `viz`. The
minimum is a protein panel and a predictor panel; the rest carry sensible
defaults. See [`docs/configuration.md`](docs/configuration.md) for the full
schema. A minimal example:

```yaml
name: my_analysis
output_dir: results

proteins:
  - id: P37840                 # resolves file paths
    display_name: alpha-synuclein
    length: 140
    domains:
      - {name: NAC, start: 61, stop: 95, category: low_complexity}

tools:
  - name: Aggrescan
    adapter: aggrescan
    path: data/aggrescan/{protein}.csv
    detection: {method: notnull, column: Prediction}
  - name: PASTA2
    adapter: pasta2
    path: data/pasta2/{protein}.txt
    detection: {method: below, column: Score, threshold: -2.0}
```

### Detection strategies

How a parsed track becomes a per-residue APR mask is declared per tool, not
hardcoded. The available primitives:

| `method`  | APR when…                              | Typical tool                     |
| --------- | -------------------------------------- | -------------------------------- |
| `above`   | `column >= threshold`                  | Aggrescan a3v, AggreProt, CrossBeta |
| `below`   | `column <= threshold`                  | PASTA 2.0 (free energy: lower = more stable) |
| `flag`    | `column` in `flag_true_values`         | FoldAmyloid (`Fold == 'f'`), APPNN (`is_hotspot`) |
| `notnull` | `column` is populated                  | Aggrescan (`Prediction` label only in hotspots) |
| `nonzero` | `column != 0`                          | Waltz (0 outside matrix hits)    |
| `present` | any parsed row is an APR               | ArchCandy (reports hit intervals only) |

The `below` primitive is not cosmetic: free-energy scores invert the usual
sense, where more-negative values mark more-stable cross-beta pairing, so an
`above`-only design would silently miscall every energy-based predictor.

### The consensus model

Evidence tiers are fractions of the active predictor set:

```yaml
consensus:
  tiers:
    - {name: unanimous, min_fraction: 1.0}
    - {name: strong,    min_fraction: 0.75}
    - {name: moderate,  min_fraction: 0.5}
  min_region_length: 5      # steric-zipper spine floor (Sawaya et al., 2007)
  coverage_fraction: 0.8    # a tool counts toward a window only if it spans >=80% of it
  overlap_resolution_max: 0.3
  denominator: configured   # or "available" to rescale against tools that returned data
```

With eight predictors enabled, `unanimous` = 8/8, `strong` ≥ 6/8, `moderate`
≥ 4/8 — numerically identical to the original hardcoded `==8 / >=6 / >=4`
thresholds, but self-adjusting. Drop a tool and the floors rescale to 6/6, 5/6,
3/6 automatically; the legacy integer thresholds would have made `unanimous`
unreachable and misclassified the rest.

The `min_region_length` floor of five residues reflects the minimum
steric-zipper spine below which a contiguous cross-beta segment is not
structurally meaningful ([Sawaya et al., 2007, *Nature*](https://doi.org/10.1038/nature05695)).

## Adding a predictor

Two steps, no core edits:

```python
# 1. register a parser that returns a normalised track
from amyloscope import register_adapter
from amyloscope.io.adapters import _finalise   # or build the frame yourself

@register_adapter("mytool")
def parse_mytool(path):
    import pandas as pd
    df = pd.read_csv(path)                       # -> Number, Residue, Score (+ flags)
    return _finalise(df, path)
```

```yaml
# 2. reference it in the config
tools:
  - name: MyTool
    adapter: mytool
    path: data/mytool/{protein}.csv
    detection: {method: above, column: Score, threshold: 0.5}
```

This adapter registry is the seam that replaces the original
`if tool == "..."` parsing ladder. See
[`docs/adding-a-predictor.md`](docs/adding-a-predictor.md) for the normalised
track contract.

## Design

The pipeline is staged so that each layer has one responsibility and the
analysis (the numbers) is fully separated from rendering (the figures):

```
config (YAML)
   │
   ▼
io.adapters ──▶ io.loader ──▶ core.regions ──▶ core.consensus
                                                     │
                        ┌────────────────────────────┼────────────────────────────┐
                        ▼                             ▼                            ▼
              analysis.statistics          analysis.domains                  viz.*
```

- **`io`** — `adapters` (one parser per predictor, registered by key) and
  `loader` (resolve paths, parse tracks, infer length/sequence).
- **`core`** — `regions` (per-tool APR calling from a `DetectionStrategy`) and
  `consensus` (fractional tiering, coverage filtering, overlap resolution).
- **`analysis`** — `statistics` (tier/length/positional analysis) and `domains`
  (APR–domain overlap), both pure computation returning dataclasses.
- **`viz`** — renderers that consume those results; styling is isolated in
  `viz.style`.

## Scope and roadmap

This release covers the consensus, statistics, domain-overlap and visualisation
core. Two capabilities from the original ribosomal codebase are deferred to
future plugins and are **not** included here:

- **ACM (amyloid core motif) detection** — the per-window beta-arch core scoring.
- **TAPASS structural extraction and 3D mapping** — projecting APRs onto
  structures.

Both are planned as optional adapters that consume the same `ConsensusResult`,
so they will slot in without disturbing the core.

## Citation and licence

If `amyloscope` contributes to published work, please cite the repository.
Released under the MIT licence (see `pyproject.toml`).

The synthetic example inputs under `examples/ribosomal/data/` are **not** real
predictions; they exist only so the example and tests run deterministically.
