"""Single rebuild entry point for the project.

Run with the project interpreter (see config.yaml `python:`):

    python.exe run_all.py --help
    python.exe run_all.py --dry-run
    python.exe run_all.py --only data text
    python.exe run_all.py --from score

Each stage maps to a dotted target ``ftyields.<subpkg>.<module>:main``. A stage
`main` is called as ``main(cfg)`` where ``cfg`` is the dict from
``ftyields.config.load_config``. Stages whose module (or ``main``) does not exist
yet are reported as "not implemented yet" and skipped. The exit code is non-zero
if any implemented stage raises.

Only the standard library is imported at module top level so ``--help`` works
in any interpreter.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ordered stage -> "module:function" target.
STAGES: dict[str, str] = {
    "data": "ftyields.data.build:main",
    "text": "ftyields.text.build:main",
    "label": "ftyields.label.build:main",
    "score": "ftyields.model.score:main",
    "indices": "ftyields.analysis.indices:main",
    "analysis": "ftyields.analysis.run:main",
    "forecast": "ftyields.analysis.forecast:main",
    "figures": "ftyields.figures.build:main",
    "paper": "ftyields.figures.paper:main",
}


def _split(target: str) -> tuple[str, str]:
    module, _, func = target.partition(":")
    return module, func


def is_implemented(stage: str) -> bool:
    """True if the stage's module file exists (does not execute the module)."""
    module, _ = _split(STAGES[stage])
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def select_stages(only: list[str] | None, from_stage: str | None) -> list[str]:
    names = list(STAGES)
    if only:
        wanted = set(only)
        return [s for s in names if s in wanted]
    if from_stage:
        return names[names.index(from_stage):]
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild the FOMC tone & yields project, stage by stage.",
        epilog="Stages, in order: " + ", ".join(STAGES),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only", nargs="+", choices=list(STAGES), metavar="STAGE",
        help="run only these stages (still executed in pipeline order)",
    )
    group.add_argument(
        "--from", dest="from_stage", choices=list(STAGES), metavar="STAGE",
        help="run this stage and every later stage",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.add_argument("--config", metavar="PATH", default=None,
                        help="path to config.yaml (default: repo-root config.yaml)")
    return parser


def run_stage(stage: str, cfg: dict) -> None:
    module_name, func_name = _split(STAGES[stage])
    module = importlib.import_module(module_name)
    func = getattr(module, func_name, None)
    if func is None:
        print(f"stage {stage}: not implemented yet")
        return
    print(f"stage {stage}: running {STAGES[stage]}")
    func(cfg)
    print(f"stage {stage}: done")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = select_stages(args.only, args.from_stage)

    if args.dry_run:
        print("Plan:")
        for stage in stages:
            status = "ready" if is_implemented(stage) else "pending (not implemented yet)"
            print(f"  {stage:<9} {STAGES[stage]:<34} {status}")
        return 0

    cfg = None
    failed: list[str] = []
    for stage in stages:
        if not is_implemented(stage):
            print(f"stage {stage}: not implemented yet")
            continue
        try:
            if cfg is None:
                from ftyields.config import load_config

                cfg = load_config(args.config)
            run_stage(stage, cfg)
        except Exception:
            traceback.print_exc()
            print(f"stage {stage}: FAILED")
            failed.append(stage)
            break  # later stages depend on earlier ones
    if failed:
        print("failed stages: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
