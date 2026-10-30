#!/usr/bin/env python
"""Run the BrainCAP pipeline against the repository testing configuration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = ("concatenate_bolds", "prep", "clustering", "post")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the testing runner."""
    testing_dir = Path(__file__).resolve().parent
    braincap_dir = testing_dir.parent / "brainCAP"

    parser = argparse.ArgumentParser(
        description="Run BrainCAP using testing/testing_param.yaml."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=testing_dir / "testing_param.yaml",
        help="Path to a BrainCAP YAML configuration.",
    )
    parser.add_argument(
        "--braincap-dir",
        type=Path,
        default=braincap_dir,
        help="Directory containing braincap.py and pipeline scripts.",
    )
    parser.add_argument(
        "--steps",
        default=",".join(STEPS),
        help="Comma-separated pipeline steps to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without running the pipeline.",
    )
    return parser


def parse_steps(value: str) -> list[str]:
    """Validate and normalize a comma-separated step list."""
    steps = [step.strip() for step in value.split(",") if step.strip()]
    invalid = sorted(set(steps) - set(STEPS))
    if invalid:
        raise ValueError(
            f"Unknown step(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(STEPS)}."
        )
    if not steps:
        raise ValueError("At least one pipeline step must be specified.")
    return steps


def main() -> int:
    """Validate inputs and run the BrainCAP orchestrator."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        steps = parse_steps(args.steps)
    except ValueError as error:
        parser.error(str(error))

    config = args.config.resolve()
    braincap_dir = args.braincap_dir.resolve()
    orchestrator = braincap_dir / "braincap.py"

    if not config.is_file():
        parser.error(f"Configuration file does not exist: {config}")
    if not orchestrator.is_file():
        parser.error(f"BrainCAP orchestrator does not exist: {orchestrator}")

    command = [
        sys.executable,
        str(orchestrator),
        f"--config={config}",
        f"--steps={','.join(steps)}",
        f"--dryrun={'yes' if args.dry_run else 'no'}",
    ]

    print("Running:", " ".join(f'"{part}"' if " " in part else part for part in command))
    completed = subprocess.run(command, cwd=braincap_dir, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
