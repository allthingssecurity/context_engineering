"""Command-line interface.

Run as a module::

    python -m context_engineering.cli list-skills
    python -m context_engineering.cli list-rules
    python -m context_engineering.cli build-context --task "What is the refund policy?" --domain rag_doc_qa
    python -m context_engineering.cli run-experiments --out results.json
    python -m context_engineering.cli compare-skills --domain rag_doc_qa
    python -m context_engineering.cli inspect-trace --result results.json --task-id rag_task_refund_policy
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from .compare_skills import compare, print_table
from .loaders import build_engine
from .run_experiments import main as run_experiments_main


def _default_examples_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "examples")


def cmd_list_skills(args: argparse.Namespace) -> int:
    engine = build_engine(args.examples)
    for skill in engine.skills.list_skills():
        print(f"{skill.id:<22} types={skill.task_types}  ops={skill.operators}")
    return 0


def cmd_list_rules(args: argparse.Namespace) -> int:
    engine = build_engine(args.examples)
    for rule in sorted(engine.rules.list_rules(), key=lambda r: r.id):
        print(
            f"{rule.id:<38} [{rule.status}] p={rule.priority} "
            f"scopes={rule.scopes} ({rule.enforcement_type})"
        )
    return 0


def cmd_build_context(args: argparse.Namespace) -> int:
    engine = build_engine(args.examples)
    pkg = engine.build_context(
        task=args.task, domain_hint=args.domain, skill_id=args.skill
    )
    print(f"Selected skill: {pkg.skill_id}\n")
    print("Selected rules:")
    for r in pkg.selected_rules:
        print(f"  - {r.id}")
    print("\nSelected resources:")
    for i in pkg.selected_items:
        print(f"  - {i.resource_id} (score {i.score:.3f}) : {i.reason_selected}")
    print("\n" + pkg.formatted_context)
    print(f"[token estimate: {pkg.token_estimate} / budget {pkg.budget_tokens}]")
    if pkg.warnings:
        print(f"[warnings: {pkg.warnings}]")
    return 0


def cmd_run_experiments(args: argparse.Namespace) -> int:
    argv = ["--examples", args.examples, "--out", args.out]
    return run_experiments_main(argv)


def cmd_compare_skills(args: argparse.Namespace) -> int:
    rows = compare(args.examples, args.domain)
    print_table(args.domain, rows)
    return 0


def cmd_inspect_trace(args: argparse.Namespace) -> int:
    with open(args.result, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    match = next((r for r in data if r["task_id"] == args.task_id), None)
    if not match:
        print(f"task-id '{args.task_id}' not found in {args.result}")
        return 1
    print(f"TRACE for task '{args.task_id}' (skill {match['skill_id']}):\n")
    for event in match["context_package"]["trace"]:
        name = event.get("event", "?")
        details = {k: v for k, v in event.items() if k != "event"}
        print(f"  * {name}: {details}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-eng")
    parser.add_argument(
        "--examples", default=_default_examples_dir(), help="examples directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-skills").set_defaults(func=cmd_list_skills)
    sub.add_parser("list-rules").set_defaults(func=cmd_list_rules)

    p_build = sub.add_parser("build-context")
    p_build.add_argument("--task", required=True)
    p_build.add_argument("--domain", default=None)
    p_build.add_argument("--skill", default=None)
    p_build.set_defaults(func=cmd_build_context)

    p_run = sub.add_parser("run-experiments")
    p_run.add_argument("--out", default="results.json")
    p_run.set_defaults(func=cmd_run_experiments)

    p_cmp = sub.add_parser("compare-skills")
    p_cmp.add_argument("--domain", default="rag_doc_qa")
    p_cmp.set_defaults(func=cmd_compare_skills)

    p_trace = sub.add_parser("inspect-trace")
    p_trace.add_argument("--result", required=True)
    p_trace.add_argument("--task-id", required=True)
    p_trace.set_defaults(func=cmd_inspect_trace)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
