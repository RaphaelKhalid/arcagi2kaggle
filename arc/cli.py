"""Command-line interface for the ARC Prize 2026 harness.

Usage:
    python -m arc.cli validate <submission.json> <challenges.json>
    python -m arc.cli score <submission.json> <challenges.json> <solutions.json>
    python -m arc.cli folds [--challenges <challenges.json>] [--json]

Exit codes: 0 on success (validation passed / scored / folds printed),
1 when validation finds issues, 2 on usage or file errors.
"""

from __future__ import annotations

import argparse
import json
import sys

from arc.folds import NUM_FOLDS, SHADOW_FOLD, compute_folds, fold_members
from arc.score import score_submission_files
from arc.tasks import load_challenges
from arc.validate import validate_submission_files

DEFAULT_TRAINING_CHALLENGES = "data/raw/arc-agi_training_challenges.json"
MAX_ISSUES_SHOWN = 50


def _cmd_validate(args: argparse.Namespace) -> int:
    issues = validate_submission_files(args.submission, args.challenges)
    if not issues:
        print(f"PASS: {args.submission} is a valid submission "
              f"for {args.challenges}")
        return 0
    print(f"FAIL: {len(issues)} issue(s) found:")
    for issue in issues[:MAX_ISSUES_SHOWN]:
        print(f"  {issue}")
    if len(issues) > MAX_ISSUES_SHOWN:
        print(f"  ... and {len(issues) - MAX_ISSUES_SHOWN} more")
    return 1


def _cmd_score(args: argparse.Namespace) -> int:
    report = score_submission_files(
        args.submission, args.challenges, args.solutions
    )
    print(report.summary())
    if args.per_task:
        print("\nper-task breakdown (correct/total):")
        for task_id, task in sorted(report.per_task.items()):
            print(f"  {task_id}: {task.correct}/{task.total}")
    return 0


def _cmd_folds(args: argparse.Namespace) -> int:
    challenges = load_challenges(args.challenges)
    assignment = compute_folds(challenges.keys())
    if args.json:
        json.dump(assignment, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    members = fold_members(assignment)
    print(f"{NUM_FOLDS}-fold split of {len(assignment)} tasks "
          f"from {args.challenges}")
    for fold in range(NUM_FOLDS):
        tag = "  <-- SHADOW (held out between major milestones)" \
            if fold == SHADOW_FOLD else ""
        ids = members[fold]
        preview = ", ".join(ids[:4]) + (", ..." if len(ids) > 4 else "")
        print(f"  fold {fold}: {len(ids):4d} tasks  [{preview}]{tag}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m arc.cli",
        description="Local evaluation harness for ARC Prize 2026.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="validate a submission file")
    p_val.add_argument("submission", help="path to submission.json")
    p_val.add_argument("challenges", help="path to the challenges JSON")
    p_val.set_defaults(func=_cmd_validate)

    p_score = sub.add_parser("score", help="score a submission file")
    p_score.add_argument("submission", help="path to submission.json")
    p_score.add_argument("challenges", help="path to the challenges JSON")
    p_score.add_argument("solutions", help="path to the solutions JSON")
    p_score.add_argument(
        "--per-task", action="store_true",
        help="also print a per-task correct/total breakdown",
    )
    p_score.set_defaults(func=_cmd_score)

    p_folds = sub.add_parser(
        "folds", help="print the deterministic 5-fold training split"
    )
    p_folds.add_argument(
        "--challenges", default=DEFAULT_TRAINING_CHALLENGES,
        help=f"challenges JSON to split (default: {DEFAULT_TRAINING_CHALLENGES})",
    )
    p_folds.add_argument(
        "--json", action="store_true",
        help="emit the full task_id -> fold mapping as JSON",
    )
    p_folds.set_defaults(func=_cmd_folds)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
