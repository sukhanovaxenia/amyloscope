"""Command-line interface.

    amyloscope run config.yaml [-o OUTPUT_DIR] [--no-figures]
    amyloscope validate config.yaml
    amyloscope adapters
    amyloscope mutate mutate.yaml
    amyloscope mutate-compare WT_consensus.tsv MUT_consensus.tsv [-p PROTEIN]
    amyloscope ingest config.yaml --tool CrossBeta --protein RPL27 [--from DIR]
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


def _cmd_ingest(args: argparse.Namespace) -> int:
    """File a manually downloaded predictor output into the configured tree.

    Some predictors cannot be driven programmatically — Cross-Beta-Pred's form
    is CAPTCHA-gated, and its result reaches the user only through a "Download
    result" button that writes wherever the browser is configured to write,
    usually ~/Downloads. Leaving the analysis to reach across into that folder
    makes the pipeline depend on a location that has nothing to do with the
    project and that differs per machine and per browser.

    This moves the file once, into the exact path the config already declares
    for that tool and protein, so every later step reads from the project tree.
    The destination is not invented here: it is `tool.path_for(protein)`, the
    same resolution the loader uses, which means a successful ingest guarantees
    the loader will find it.
    """
    import shutil
    from pathlib import Path as _Path

    cfg = load_config(args.config)
    tools = {t.name.lower(): t for t in cfg.tools}
    tool = tools.get(args.tool.lower())
    if tool is None:
        print(
            f"unknown tool {args.tool!r}; config declares {sorted(t.name for t in cfg.tools)}",
            file=sys.stderr,
        )
        return 1
    known = {p.id for p in cfg.proteins}
    if args.protein not in known:
        print(f"unknown protein {args.protein!r}; config declares {sorted(known)}",
              file=sys.stderr)
        return 1

    if args.file:
        source = _Path(args.file).expanduser()
        if not source.is_file():
            print(f"no such file: {source}", file=sys.stderr)
            return 1
    else:
        search = _Path(args.source).expanduser()
        candidates = sorted(
            (p for p in search.glob(args.pattern) if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"no file matching {args.pattern!r} in {search}", file=sys.stderr)
            return 1
        source = candidates[0]
        # Named explicitly: picking "the newest download" silently is how the
        # wrong protein's result gets filed under the right protein's name.
        print(f"newest match: {source}")

    dest = _Path(tool.path_for(args.protein))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not args.force:
        print(f"refusing to overwrite {dest} (pass --force)", file=sys.stderr)
        return 1
    (shutil.copy2 if args.keep else shutil.move)(str(source), str(dest))
    print(f"{args.tool} / {args.protein}: {source.name} -> {dest}")
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

    p_in = sub.add_parser(
        "ingest",
        help="file a manually downloaded predictor output into the config's tree",
    )
    p_in.add_argument("config")
    p_in.add_argument("--tool", required=True, help="tool name as written in the config")
    p_in.add_argument("--protein", required=True, help="protein id as written in the config")
    p_in.add_argument("--file", help="the downloaded file (skips the search)")
    p_in.add_argument("--from", dest="source", default="~/Downloads",
                      help="directory to search when --file is omitted")
    p_in.add_argument("--pattern", default="*.json",
                      help="glob used inside --from (default: *.json)")
    p_in.add_argument("--keep", action="store_true",
                      help="copy instead of move, leaving the download in place")
    p_in.add_argument("--force", action="store_true", help="overwrite an existing file")
    p_in.set_defaults(func=_cmd_ingest)

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
