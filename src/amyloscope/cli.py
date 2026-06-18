"""Command-line interface.

    amyloscope run config.yaml [-o OUTPUT_DIR] [--no-figures]
    amyloscope validate config.yaml
    amyloscope adapters
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
