"""Command-line interface.

    amyloscope run config.yaml [-o OUTPUT_DIR] [--no-figures]
    amyloscope validate config.yaml
    amyloscope adapters
    amyloscope mutate mutate.yaml
    amyloscope mutate-compare WT_consensus.tsv MUT_consensus.tsv [-p PROTEIN]
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import ConfigError, load_config, with_overrides
from .io.adapters import available_adapters


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run

    cfg = load_config(args.config)
    if args.output:
        cfg = with_overrides(cfg, output_dir=args.output)
    artifacts = run(cfg, make_figures=not args.no_figures)
    n_regions = len(artifacts.consensus.all_regions())
    print(f"Consensus regions: {n_regions}")
    print(f"Wrote {len(artifacts.written_files)} files to {cfg.output_dir}/")
    for path in artifacts.written_files:
        print(f"  {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"OK: '{cfg.name}' — {len(cfg.proteins)} proteins, "
          f"{len(cfg.enabled_tools)} enabled predictors")
    return 0


def _cmd_adapters(_args: argparse.Namespace) -> int:
    print("Registered adapters:")
    for key in available_adapters():
        print(f"  {key}")
    return 0


def _cmd_mutate(args: argparse.Namespace) -> int:
    from collections import Counter

    from .mutate import load_mutate_config
    from .mutate import run as run_mutate

    try:
        cfg = load_mutate_config(args.config)
    except ValueError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    artifacts = run_mutate(cfg)
    print(f"Panel: {len(cfg.proteins)} protein(s), "
          f"{len(artifacts.records)} variants total")
    per_protein = Counter(r.protein for r in artifacts.records)
    for job in cfg.proteins:
        pid = job.sequence.id
        print(f"  {pid} ({len(job.sequence.sequence)} aa): "
              f"{per_protein[pid]} variants")
    for path in artifacts.written_files:
        print(f"  {path}")
    return 0


def _cmd_mutate_compare(args: argparse.Namespace) -> int:
    from .mutate import compare_consensus

    diff = compare_consensus(args.wildtype, args.mutant, protein=args.protein)
    print(diff.report())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amyloscope", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the full pipeline from a YAML config")
    p_run.add_argument("config")
    p_run.add_argument("-o", "--output", help="override output_dir")
    p_run.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    p_run.set_defaults(func=_cmd_run)

    p_val = sub.add_parser("validate", help="validate a config without running")
    p_val.add_argument("config")
    p_val.set_defaults(func=_cmd_validate)

    p_ad = sub.add_parser("adapters", help="list registered predictor adapters")
    p_ad.set_defaults(func=_cmd_adapters)

    p_mut = sub.add_parser("mutate", help="generate directed mutant sequences")
    p_mut.add_argument("config")
    p_mut.set_defaults(func=_cmd_mutate)

    p_cmp = sub.add_parser(
        "mutate-compare", help="diff wild-type vs mutant consensus_regions.tsv"
    )
    p_cmp.add_argument("wildtype", help="wild-type consensus_regions.tsv")
    p_cmp.add_argument("mutant", help="mutant consensus_regions.tsv")
    p_cmp.add_argument("-p", "--protein", help="restrict to one protein id")
    p_cmp.set_defaults(func=_cmd_mutate_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
