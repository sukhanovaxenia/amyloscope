# Changelog

All notable changes to **amyloscope** are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

A release that changes a number the previous version reported opens with a
**Results affected** section. Anyone who has run a prior version against the same
inputs should read that section before comparing outputs, because the difference
is a correction and not drift.

---

## [0.3.0] — 2026-08-04

Domain-overlap analysis rebuilt on residue sets and a permutation null;
consensus-versus-measurement correlation added; four faults fixed that produced
either an impossible number or no output at all.

### Results affected

| Output | Previous | Corrected | Cause |
|---|---|---|---|
| `domain_overlap.txt`, in-domain residues | 64 aa (148.8 %) for uS5, 30 aa (136.4 %) for eL27 | 43 aa (100 %), 22 aa (100 %) | residues counted once per overlapping domain rather than once |
| `domain_overlap.txt`, outside residues | −0 aa (−48.8 %) | 0 aa (0 %) | derived by subtraction from an inflated total |
| Domain enrichment | not computed; a fixed-threshold verdict was printed instead | RNA-binding 1.73-fold, one-sided permutation *p* = 0.0198 | no null model existed |
| Spearman, consensus regions vs external measurement | *p* = 0.051 (asymptotic) | *p* = 0.200 (exact permutation); floor 0.200 | the asymptotic *p* is not attainable at *n* = 4 with ties |
| Positional χ² | printed unqualified | printed with the expected count per bin | expectation of 0.9/bin is below the asymptotic validity threshold |

The in-domain percentages above exceeded 100 %, so any figure or statement
derived from them is wrong rather than imprecise. Reruns are required for
`domain_overlap.txt` and for anything quoting the domain fraction.

### Added

- `analysis/domains.py`: per-category enrichment against a relocation null in
  which whole consensus regions are moved to uniformly random positions with
  their lengths preserved and without mutual overlap, so each draw covers
  exactly the observed residue count. Whole regions rather than residues are
  relocated because consensus residues occur in contiguous runs; a residue-level
  binomial returns *p* < 0.0001 on data for which the region-level null returns
  0.02.
- `analysis/domains.py`: `primary_category` distinguishes the one comparison a
  study was designed to test — reported one-sided at a family size of one — from
  exploratory categories, reported two-sided with Holm correction. Without the
  distinction the primary result is one of several tests presented as the only
  one.
- `analysis/domains.py`: `DomainOverlapResult.null_hist` carries the permutation
  null as a per-category count histogram (tens of integers) rather than the raw
  draws (hundreds of thousands), so the figure can draw the distribution without
  recomputing it and cannot disagree with the report.
- `viz/overlap.py`: `plot_domain_overlap()`. Each category is drawn as its null
  distribution — box for the interquartile range, whiskers for the central 95 % —
  with the observed count as a marker on the same axis, so a category whose load
  is explained by its sequence coverage shows an observation inside its own box.
  `style="publication"` (default) omits the statistics column; `"diagnostic"`
  retains it.
- `viz/overlap.py`: `caption()` generates the figure legend from the same result
  object the panels were drawn from, so *n*, the null construction and the
  *p*-values in the text cannot drift from the figure.
- `viz/correlation.py`: `plot_measurement_correlation()` scatters a consensus
  summary against an external per-protein measurement. Three summaries are
  computed in parallel — called-region count, residues at the moderate tier,
  residues below it — and all three are shown, because the choice of summary is
  a free parameter and reporting only the favourable one is selection on the
  outcome. In this dataset they disagree in sign.
- `viz/correlation.py`: `exact_spearman_p()` enumerates the permutation null for
  *n* ≤ 8 and samples it above, returning the smallest *p* the design can
  produce alongside the observed one. The floor is taken from the null itself,
  not from *n*!, because ties collapse it: an untied *n* = 4 admits a floor near
  2/24, while two tied ranks raise it to 0.20.
- `config.py`: `MeasurementSet` on `PipelineConfig`, holding external
  measurements as data rather than as a rendering flag. Accepts a compact
  mapping, an explicit `values`/`label`/`source` block, or `from_file` reading a
  delimited table with `id_column`, `value_column`, `id_map` and an optional
  `require` list.
- `config.py`: file-backed measurements record the source path, modification
  time and SHA-256 on the result, so the report states which upstream run
  produced the figure and a stale transcription is detectable.
- `config.py`: `primary_domain_category`, validated against the categories the
  configured domains actually use.
- `predictor_audit.py`: standalone gate verifying that each predictor output
  contains the sequence of the protein it is filed under, by cross-checking the
  residue column between tools and against a reference FASTA. Outputs are
  collected by filename and nothing downstream re-reads the sequence, so a
  mislabelled file otherwise propagates undetected into the consensus and every
  figure. Non-zero exit on failure.

### Changed

- `analysis/domains.py`: in-domain and outside counts are computed on residue
  sets and are now complements by construction. Per-domain intersections are
  still reported but labelled as overlapping counts, not as a partition.
- `analysis/domains.py`: the fixed-threshold interpretation — below 0.4
  "depleted", above 0.6 "enriched" — is replaced by the enrichment test. The
  in-domain fraction is bounded below by how much of the sequence the annotation
  covers (79.9 % and 96.3 % for the two proteins with regions), so it cannot
  distinguish enrichment from coverage.
- `analysis/domains.py`: the report's H1 previously read "depleted within
  functional domains" while its conclusion asserted enrichment. H1 is now stated
  in terms of the primary category and matches what is tested.
- `viz/consensus.py`: `plot_positional_enrichment()` labels bin edges rather
  than centres. Centres at one decimal place produced "0.2" and "0.9" twice
  each, leaving two pairs of adjacent bars indistinguishable. The y axis is
  forced to integer ticks and the χ² annotation carries *n* and the expected
  count per bin.
- `viz/correlation.py`: statistics moved out of the data area into their own
  axis-off column. At `transAxes` (0.03, 0.97) the annotation covered the
  topmost point, and with four proteins one of them is always at the y limit.
- `viz/correlation.py`: bar values are right-aligned in a fixed column rather
  than anchored to the bar tip, where a negative ρ drew its text leftward into
  the category label on the same row.
- `viz/correlation.py`: explicit point sizes derived from
  `viz.resolved_base_font` replace relative size keywords, which resolve against
  the preset base font and rendered annotations at 14–17 pt on a poster run
  while the character count of a label stayed the same.
- `pipeline.py`: the domain-overlap result is retained in `run()` and passed to
  `_render_figures(..., overlap=None)`; `PipelineArtifacts.overlap` exposes it so
  a caller working in Python has the enrichment table without re-running the
  null.
- `pipeline.py`: `load_config()` builds the protein list before constructing
  `PipelineConfig`, so protein ids can be passed to the measurement builder for
  scope filtering.
- `viz/labels.py`: keys added for both new figures in English and Russian; the
  correlation strings carry explicit line breaks so they fit a narrow column.

### Fixed

- `analysis/domains.py`: `total_apr_aa`, `total_in_domain_aa` and
  `total_outside_domain_aa` called `len()` on a generator expression, raising
  `TypeError` and aborting `format_overlap_report()` at the GLOBAL section.
- `pipeline.py`: `plot_domain_overlap(result)` was called with the
  `ConsensusResult`, raising `AttributeError: 'ConsensusResult' object has no
  attribute 'enrichment'`. The figure now receives the analysis object, and the
  wrong type is rejected with a message naming the right one.
- `pipeline.py`: `plot_cross_protein_overview()` and
  `plot_all_proteins_agreement()` sat inside the per-protein loop at the same
  indentation as the per-protein blocks, so each was rendered once per protein
  and overwritten every time but the last — five renders of the two most
  expensive figures in the set. Both are hoisted to the panel-level section.
- `config.py`: external measurements were declared as a boolean on `VizConfig`
  and read from `PipelineConfig`, so the condition was false on every run and no
  correlation figure was written, without an error. Had the attribute path alone
  been corrected the call would have proceeded on a boolean and failed inside
  the figure.
- `config.py`: the guard rejecting the superseded `viz.measurements` key was
  applied to the top-level configuration mapping rather than to the `viz` block,
  so it ignored the stale key and rejected the correct top-level form it
  instructed users to adopt.
- `analysis/domains.py`: relocation-null placements are resampled until a
  protein's regions do not overlap one another. Allowing overlaps let a draw
  cover fewer residues than the observation it was compared against, biasing the
  null low and the enrichment high; with six 6–11 residue regions in 293
  positions a pair collides in roughly 4 % of attempts.

### Removed

- `VizConfig.measurement`. Measurements are data and live at the top level of
  the configuration.
- The threshold-based verdict from `_interpret_overlap()`.

### Migration

Configuration:

```yaml
# remove from the viz block
viz:
  measurements: true        # rejected with an explanatory error

# add at the top level, beside `hypothesis`
primary_domain_category: rna_binding

measurements:
  from_file: path/to/measurements.csv
  id_column: condition
  value_column: mobile_fraction_pct
  id_map: {display_name: protein_id}    # when the source is keyed differently
  require: [protein_id, ...]            # fail if the assay loses a condition
  label: "Measured value (unit)"
```

Callers:

```python
# before
overlap = compute_domain_overlap(result)
plot_domain_overlap(result)

# after
overlap = compute_domain_overlap(result)          # primary_category read from config
plot_domain_overlap(overlap)                      # takes the analysis object
plot_domain_overlap(overlap, style="diagnostic")  # with the statistics column
```

`compute_domain_overlap()` runs 20 000 placements by default; raise `draws` for
finer *p*-value resolution. Rerun any analysis whose `domain_overlap.txt`
predates this release.

---

## [0.2.0] - 2026-06-24

### Added
- **In-silico directed mutagenesis** (`amyloscope.mutate`). Designs sequence
  variants along three vectors — `activating`, `inhibiting`, and `neutral`
  (conservative controls plus `nonsense`/truncation) — under an `amyloid`
  (Chiti–Dobson) or `condensate` (sticker-and-spacer) grammar, both selectable
  per run. Targets are the union of consensus APRs (closed loop), explicit
  regions/positions, and a whole-sequence scan. The rule engine proposes
  candidates; the external predictor panel adjudicates them.
- **Multi-protein panels.** A run can mutate any number of proteins at once via
  a `proteins:` list; grammars/vectors/limits/weights are shared, while
  sequence, targets and protected positions are per protein. Per-protein FASTA
  and manifest outputs plus a combined manifest. `consensus_protein` defaults to
  each protein's own id so a shared consensus table serves the whole panel.
- CLI: `amyloscope mutate CONFIG.yaml` and
  `amyloscope mutate-compare WT.tsv MUT.tsv [-p ID]` (diff wild-type vs mutant
  consensus to read out a variant's effect).
- `examples/mutagenesis/` (single-protein GAPDH and a GAPDH + Abeta42 panel),
  `docs/mutagenesis.md`, and a mutagenesis test suite.

### Notes
- The single-protein config spelling (top-level `sequence:`/`targets:`) remains
  valid and is treated as a one-element panel — no breaking changes.

### Fixed
- `import amyloscope` (and `amyloscope.mutate`) no longer require matplotlib: the
  plotting stack is imported lazily, only when figures are rendered, so a base
  install without the `[viz]` extra works.

### Packaging
- Version is now single-sourced from `src/amyloscope/__init__.py` via Hatchling
  dynamic versioning (no more duplicate version strings).
- Added a tag-triggered `release` workflow that tests, builds the sdist + wheel,
  checks the tag matches the package version, and publishes a GitHub Release with
  the artifacts (optional PyPI Trusted Publishing block included, commented).

## [0.1.0]

### Added
- Consensus meta-scoring, statistics, domain-overlap and visualisation core.
- Eleven predictor adapters: AGGRESCAN, APPNN, FoldAmyloid, PASTA2, Waltz,
  AggreProt, CrossBeta (web + local), ArchCandy (web + local), TANGO.
