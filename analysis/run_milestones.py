"""Single entry point for rebuilding Matchmakers Milestones 2 and 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from .m2_eda import build_milestone_2
from .m3_actor_baseline import build_milestone_3
from .milestone_common import AUDIT_V5, RAW_DATA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--milestone",
        choices=("2", "3", "all"),
        default="all",
        help="Milestone to rebuild; default: all.",
    )
    parser.add_argument("--input", type=Path, default=RAW_DATA)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_V5)
    parser.add_argument(
        "--force", action="store_true", help="Replace existing milestone artifacts."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written: list[Path] = []
    if args.milestone in {"2", "all"}:
        written.extend(
            build_milestone_2(
                input_path=args.input,
                audit_dir=args.audit_dir,
                force=args.force,
            )
        )
    if args.milestone in {"3", "all"}:
        written.extend(
            build_milestone_3(
                input_path=args.input,
                audit_dir=args.audit_dir,
                force=args.force,
            )
        )
    print(f"wrote {len(written)} milestone artifacts")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()

