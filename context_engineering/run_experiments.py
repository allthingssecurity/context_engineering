"""Experiment runner: build context for every example task and grade it.

Usage::

    python -m context_engineering.run_experiments --examples <dir> --out results.json

For each task it:
  * builds the context package (running the skill's pipeline F_s),
  * runs the skill's evaluators against the task's gold spec,
  * prints a readable report, and
  * writes all results to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List

from .engine import ContextEngine
from .loaders import build_engine, list_domains, load_domain_bundle
from .models import ContextPackage, EvaluationResult, ExperimentResult


def _default_examples_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "examples")


def run_task(
    engine: ContextEngine,
    domain: str,
    task: Dict[str, Any],
    gold: Dict[str, Dict[str, Any]],
) -> ExperimentResult:
    """Build context for a single task and evaluate it."""
    spec = gold.get(task["task_id"], {})
    package = engine.build_context(
        task=task["task"],
        domain_hint=domain,
        skill_id=task.get("skill_id"),
    )
    results = engine.evaluate(package, spec)
    aggregate = (
        sum(r.score for r in results) / len(results) if results else 0.0
    )
    return ExperimentResult(
        task_id=task["task_id"],
        domain=domain,
        skill_id=package.skill_id,
        context_package=package,
        evaluation_results=results,
        aggregate_score=round(aggregate, 4),
    )


def run_all(examples_dir: str) -> List[ExperimentResult]:
    """Run every task across every domain."""
    domains = list_domains(examples_dir)
    engine = build_engine(examples_dir, domains)
    all_results: List[ExperimentResult] = []
    for domain in domains:
        tasks, gold = load_domain_bundle(examples_dir, domain)
        for task in tasks:
            all_results.append(run_task(engine, domain, task, gold))
    return all_results


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
def _fmt_eval(r: EvaluationResult) -> str:
    mark = "PASS" if r.passed else "FAIL"
    return f"    - {r.evaluator_name}: {mark} (score={r.score})"


def print_report(results: List[ExperimentResult]) -> None:
    """Print a human-readable per-task report."""
    for res in results:
        pkg = res.context_package
        print("=" * 72)
        print(f"TASK: {res.task_id}   [{res.domain}]")
        print(f"  task text     : {pkg.task}")
        print(f"  selected skill: {res.skill_id}")
        print(f"  selected rules: {[r.id for r in pkg.selected_rules]}")
        print(
            "  selected res. : "
            + str(
                [
                    (i.resource_id, round(i.score, 3))
                    for i in pkg.selected_items
                ]
            )
        )
        print(f"  token estimate: {pkg.token_estimate} / budget {pkg.budget_tokens}")
        if pkg.warnings:
            print(f"  warnings      : {pkg.warnings}")
        print("  evaluations:")
        for r in res.evaluation_results:
            print(_fmt_eval(r))
        print(f"  AGGREGATE SCORE: {res.aggregate_score}")
    print("=" * 72)
    if results:
        overall = sum(r.aggregate_score for r in results) / len(results)
        n_pass = sum(
            1
            for r in results
            if all(e.passed for e in r.evaluation_results)
        )
        print(
            f"OVERALL: {len(results)} tasks, {n_pass} fully-passing, "
            f"mean aggregate score = {round(overall, 4)}"
        )


# --------------------------------------------------------------------------- #
# Serialization                                                                #
# --------------------------------------------------------------------------- #
def _package_to_dict(pkg: ContextPackage) -> Dict[str, Any]:
    return {
        "task": pkg.task,
        "skill_id": pkg.skill_id,
        "selected_rules": [r.id for r in pkg.selected_rules],
        "selected_items": [
            {
                "resource_id": i.resource_id,
                "title": i.title,
                "score": i.score,
                "reason_selected": i.reason_selected,
                "metadata": i.metadata,
            }
            for i in pkg.selected_items
        ],
        "formatted_context": pkg.formatted_context,
        "token_estimate": pkg.token_estimate,
        "budget_tokens": pkg.budget_tokens,
        "trace": pkg.trace,
        "warnings": pkg.warnings,
    }


def results_to_dict(results: List[ExperimentResult]) -> List[Dict[str, Any]]:
    out = []
    for res in results:
        out.append(
            {
                "task_id": res.task_id,
                "domain": res.domain,
                "skill_id": res.skill_id,
                "aggregate_score": res.aggregate_score,
                "evaluation_results": [asdict(e) for e in res.evaluation_results],
                "context_package": _package_to_dict(res.context_package),
            }
        )
    return out


def save_results(results: List[ExperimentResult], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results_to_dict(results), fh, indent=2)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run context-engineering experiments")
    parser.add_argument("--examples", default=_default_examples_dir())
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args(argv)

    results = run_all(args.examples)
    print_report(results)
    save_results(results, args.out)
    print(f"\nSaved {len(results)} results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
