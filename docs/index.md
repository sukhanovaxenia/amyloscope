# amyloscope

Consensus meta-scoring, domain mapping and visualisation of protein
amyloidogenicity predictions.

`amyloscope` is the **post-processing layer** downstream of the sequence-based
aggregation predictors. It does not predict aggregation; it ingests the
per-residue tracks those tools emit and turns them into a defensible,
cross-validated picture of where a protein is aggregation-prone:

1. **Call** per-predictor aggregation-prone regions (APRs) from each track.
2. **Reconcile** them into a fractional cross-predictor consensus.
3. **Map** the consensus onto a user-defined domain architecture.
4. **Report** statistics and render publication figures.

Everything is specified in a YAML configuration, so the same code analyses any
protein panel — the package is not tied to the ribosomal proteins it was first
written for.

## The argument for consensus

The published aggregation predictors encode mechanistically distinct
hypotheses about cross-beta assembly — hydrophobic-cluster burial, beta-pairing
free energy, position-specific amyloid matrices, beta-arch packing, and learned
per-residue propensity. Each carries characteristic false positives. A window
flagged independently by several *orthogonal* models is therefore a far more
credible intrinsic aggregation determinant than one called by any single tool.
`amyloscope` makes that consensus the central object of analysis.

The key design decision is to express agreement as a **fraction** of the
predictors actually run, not an absolute count. With eight predictors,
`unanimous` is 8/8 and `strong` is ≥ 6/8; drop two and the floors rescale to
6/6 and 5/6 automatically. Tier semantics are invariant to panel size — a
property an integer-threshold design cannot provide.

## Where to go next

- **[Configuration](configuration.md)** — the full YAML schema: protein panel,
  predictor panel, detection strategies, the consensus model, and rendering.
- **[Adding a predictor](adding-a-predictor.md)** — the normalised-track
  contract and the two-step registration that adds a tool without touching core
  code.
- **[API reference](api.md)** — rendered from the source docstrings.

## Install and run

```bash
pip install -e ".[viz]"
python examples/ribosomal/make_fixtures.py
amyloscope run examples/ribosomal/config.yaml
```

!!! note "Synthetic example data"
    The inputs under `examples/ribosomal/data/` are generated, not real
    predictions. They exist so the example and test suite run deterministically.
