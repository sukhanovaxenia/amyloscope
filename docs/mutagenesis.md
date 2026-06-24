# In-silico directed mutagenesis

The `amyloscope.mutate` module designs sequence variants intended to *raise*,
*lower*, or *leave unchanged* the aggregation or phase-separation propensity of a
protein, so the variants can be scored by the predictor panel and triaged for
the bench. It closes the loop the rest of the package opens: the consensus run
locates aggregation-prone regions (APRs); mutagenesis perturbs them in a defined
direction; re-scoring the mutants and diffing the consensus reads out the
effect.

!!! warning "What the engine does and does not claim"
    The rule engine **proposes** candidate mutations from sequence-level
    physicochemistry. It does not predict whether a variant will aggregate. The
    external predictor panel — and ultimately wet-lab measurement — is the
    arbiter. A single scale cannot capture the full grammar of aggregation, so
    the engine is a hypothesis generator that concentrates prediction and
    experimental effort on informative variants rather than a blind 19×N scan.

## The biophysical basis

Substitution effects are scored against the axes the literature has repeatedly
tied to aggregation rate. The amyloid grammar follows the Chiti–Dobson model
(Chiti et al. 2003, *Nature*; Chiti & Dobson 2006, *Annu. Rev. Biochem.*) and
Zyggregator (Pawar et al. 2005, *J. Mol. Biol.*): a transparent linear
combination of Kyte–Doolittle hydrophobicity and Chou–Fasman β-sheet propensity,
penalised by side-chain charge and by the β-breakers proline and glycine. The
condensate grammar scores aromatic and cation-π **sticker valence** from the
sticker-and-spacer model of low-complexity-domain phase separation (Wang et al.
2018, *Cell*; Martin et al. 2020, *Science*; Bremer et al. 2022, *Nat. Chem.*),
where tyrosine outweighs phenylalanine and arginine outweighs lysine.

Because both per-residue profiles carry the gatekeeper penalties explicitly,
scoring a substitution over a short **window** rather than at the single mutated
residue reproduces context effects without special-casing. Raising
hydrophobicity in an APR core lifts the window score; removing a flanking
charge or proline (a gatekeeper; Reumers et al. 2009; Rousseau & Serrano) lifts
it too, by removing a penalty. This is what lets all three vectors run on one
operation — score every alternative residue, then select by the sign and
magnitude the vector requires.

"Distant" sequence context is reported as charge patterning: net charge per
residue, fraction of charged residues, and the sequence charge decoration SCD
(Das & Pappu 2013, *PNAS*; Sawle & Ghosh 2015), which grows more negative as
like-charges segregate and is the standard sequence-only proxy for the
electrostatic context point scales miss. Structural (3D) context is *not*
modelled — from sequence alone, "distant" means non-local charge patterning, not
tertiary contacts.

## The three vectors

Each vector is the same scoring operation read in a different direction:

| Vector | Selects | Typical move |
| --- | --- | --- |
| `activating` | substitutions that most **raise** region propensity | strengthen a core (→V/I/F), or remove a flanking gatekeeper |
| `inhibiting` | substitutions that most **lower** it | install a β-breaker (→P/G) or charge (→D/E/K/R), strip a sticker |
| `neutral` | iso-physicochemical swaps with effect nearest **zero** | conservative control (V↔I, D↔E, S↔T) |

Keeping the three on one substrate is what makes the controls meaningful: a
"neutral" mutation is neutral *on the same axis* the directional arms move along.
The neutral arm has two modes (both generated): conservative iso-propensity
substitutions, and **nonsense/truncation** variants that ablate a region and
everything downstream — a stronger negative control for "is this region
necessary?". The conservative set is grammar-specific: Phe↔Tyr and Lys↔Arg are
near-silent for amyloid but change sticker/cation-π valence, so they are excluded
from the condensate conservative set (an aromatic therefore has no
condensate-conservative partner, by design).

## Targeting

Targets are the **union** of three sources, any combination of which may be set:

- **Consensus APRs** (`consensus_tsv`) — the closed loop. The module parses a
  `consensus_regions.tsv` from a prior `amyloscope run`, optionally filtered to
  one protein and a tier floor, and mutates within those regions.
- **Explicit regions / positions** (`regions`, `positions`) — hand-picked windows
  or single residues.
- **Whole-sequence scan** (`scan_whole_sequence`) — every position, each scored
  against a local window.

`protected_positions` are never mutated (e.g. catalytic residues). Multi-site
variants are built by directed enumeration over the highest-leverage positions,
not the full combinatorial product, so the search stays bounded by the
configured caps.

## Multiple proteins (panels)

A run can mutate any number of proteins at once. The grammars, vectors,
combinatorial limits and scoring weights are shared across the panel; the
sequence, targets and protected positions are declared per protein, since those
are inherently protein-specific. Each protein is generated, ranked and capped
independently (`max_total` is per protein), and a combined manifest ties them
together.

```yaml
name: panel
output_dir: mutants
modes: [amyloid, condensate]
vectors: [activating, inhibiting, neutral, nonsense]
mutation:
  orders: [1, 2]
  max_total: 400                 # per protein

proteins:
  - sequence: {id: GAPDH, fasta: gapdh.fasta}
    targets:
      consensus_tsv: panel_consensus_regions.tsv   # shared table
      consensus_min_tier: strong
      regions: [[40, 49]]
    protected_positions: [152, 156]
  - sequence: {id: Abeta42, fasta: abeta42.fasta}
    targets:
      consensus_tsv: panel_consensus_regions.tsv   # same file
      positions: [22]
```

A single consensus table usually contains every protein in the panel (one
`amyloscope run` over the whole set). When `consensus_protein` is omitted, it
**defaults to the protein's own id**, so the same `consensus_tsv` serves each
protein without repeating the filter. The single-protein spelling — a top-level
`sequence:`/`targets:` block instead of a `proteins:` list — remains valid and is
equivalent to a one-element panel. See `examples/mutagenesis/panel.yaml`.



```yaml
name: gapdh
output_dir: mutants

sequence:
  id: GAPDH
  fasta: gapdh.fasta            # or: sequence: MGKVK...

targets:                        # union of all sources set
  consensus_tsv: consensus_regions.tsv
  consensus_protein: GAPDH
  consensus_min_tier: strong
  regions: [[40, 49]]
  positions: [130]
  scan_whole_sequence: false

modes: [amyloid, condensate]    # grammars to apply, scored independently
vectors: [activating, inhibiting, neutral, nonsense]

mutation:
  orders: [1, 2]                # single and double mutants
  top_k_per_position: 2
  top_k_positions: 10           # leverage-ranked positions for multi-site
  max_per_order: 40             # cap per (mode, vector, order)
  max_total: 600
  protected_positions: [152, 156]
  truncate_at: region_start     # 'region_start' | 'position'

weights:                        # all optional; Chiti-Dobson-style linear model
  hydrophobicity: 1.0
  beta: 1.0
  charge_penalty: 0.7
  breaker_penalty: 1.2
  window: 7
  flank: 3
```

## Running

```bash
amyloscope mutate config.yaml
```

For each protein this writes two artifacts to `output_dir`, named by protein id:

- **`<id>_mutants.fasta`** — the wild type (first record) followed by every
  variant. Headers encode the edit, grammar, vector, and predicted load change,
  e.g. `>GAPDH_amyloid_activating_o1_0001|amyloid|activating|G134V|dLoad=+2.425`.
- **`<id>_manifest.tsv`** — one row per variant: `protein`, `mutant_id`, `mode`,
  `vector`, `order`, `mutation` (HGVS-like, e.g. `p.G134V`), `length`,
  `delta_load`, and the charge-patterning deltas `delta_NCPR`, `delta_FCR`,
  `delta_SCD`.

When the panel holds more than one protein, a combined `<name>_manifest.tsv`
spanning all of them is written as well. Within each grammar/vector/order group
the rows are ordered by `delta_load`, the intrinsic heuristic, so the most
impactful candidates sit at the top.

## Closing the loop

Score `<name>_mutants.fasta` with the external predictors, post-process the
wild-type and mutant outputs with `amyloscope run`, then diff the two consensus
tables:

```bash
amyloscope mutate-compare wt_consensus_regions.tsv mut_consensus_regions.tsv -p GAPDH
```

```
Consensus APR comparison (wild type -> mutant)
==============================================
APR residues:   58 -> 71 (+13)
Regions gained: 1  ['100-110']
Regions lost:   0  []
Regions kept:   5
Net effect:     aggregation-prone coverage increased.
```

This panel-level read-out is the quantity the intrinsic `delta_load` only
approximates, and it is the number to trust when ranking variants for synthesis.

## Limitations

- A single linear scale is insufficient; the predictor panel is the arbiter, and
  the engine only ranks candidates.
- Amyloid and condensate grammars are kept separate by design — the optimal move
  for one is often not the optimal move for the other.
- "Distant" context is non-local charge patterning, not 3D structure.
- The combinatorial space is sampled, not enumerated exhaustively; which variants
  survive depends on the leverage ranking and the configured caps.
