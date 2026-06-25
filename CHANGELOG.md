# Changelog

All notable changes to amyloscope are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project adheres to
[Semantic Versioning](https://semver.org/).

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
