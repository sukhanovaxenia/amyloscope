"""In-silico directed mutagenesis for amyloidogenicity and condensate analysis.

Generates sequence variants along three directional vectors — activating
(pro-aggregation), inhibiting (anti-aggregation), and neutral (conservative
controls), plus optional nonsense/truncation variants — under an amyloid or
condensate grammar, targeting consensus APRs, explicit regions/positions, or a
whole-sequence scan. The resulting FASTA feeds the external predictor panel; the
``compare`` utility then diffs wild-type vs mutant consensus to read out the
effect.
"""

from __future__ import annotations

from .compare import ConsensusDiff, compare_consensus
from .config import (
    MutagenesisConfig,
    MutateConfigError,
    MutationSpec,
    ProteinJob,
    SequenceSpec,
    TargetSpec,
    load_mutate_config,
)
from .effect import (
    ChargePatterning,
    EffectWeights,
    aggregation_load,
    charge_patterning,
    profile,
)
from .engine import (
    MutantRecord,
    generate,
    generate_for_protein,
    resolve_targets,
)
from .io import MutagenesisArtifacts, run, run_from_file, write_fasta, write_manifest
from .rules import Substitution, propose

__all__ = [
    "ChargePatterning",
    "ConsensusDiff",
    "EffectWeights",
    "MutagenesisArtifacts",
    "MutagenesisConfig",
    "MutantRecord",
    "MutateConfigError",
    "MutationSpec",
    "ProteinJob",
    "SequenceSpec",
    "Substitution",
    "TargetSpec",
    "aggregation_load",
    "charge_patterning",
    "compare_consensus",
    "generate",
    "generate_for_protein",
    "load_mutate_config",
    "profile",
    "propose",
    "resolve_targets",
    "run",
    "run_from_file",
    "write_fasta",
    "write_manifest",
]
