"""Meta experiment: compare several skills on the same domain's tasks.

This demonstrates the inner/outer loop distinction:
  * inner loop: for one task, which context (skill) scored best?
  * outer loop: across all tasks, which skill generalized best?

Usage::

    python -m context_engineering.compare_skills --domain rag_doc_qa
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

from .engine import ContextEngine
from .loaders import build_engine, load_domain_bundle
from .models import EvaluationResult


def _default_examples_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "examples")


def _eval_by_name(results: List[EvaluationResult]) -> Dict[str, EvaluationResult]:
    return {r.evaluator_name: r for r in results}


def compare(examples_dir: str, domain: str) -> List[Dict[str, Any]]:
    """Run every skill scoped to ``domain`` against that domain's tasks."""
    engine = build_engine(examples_dir, [domain])
    tasks, gold = load_domain_bundle(examples_dir, domain)

    skills = [s for s in engine.skills.list_skills() if domain in s.task_types]
    rows: List[Dict[str, Any]] = []

    for skill in skills:
        aggregates: List[float] = []
        budget_pass = 0
        gold_recall: List[float] = []
        citation_pass = 0
        citation_total = 0

        for task in tasks:
            spec = gold.get(task["task_id"], {})
            pkg = engine.build_context(
                task=task["task"], domain_hint=domain, skill_id=skill.id
            )
            results = engine.evaluate(pkg, spec)
            by_name = _eval_by_name(results)
            if results:
                aggregates.append(sum(r.score for r in results) / len(results))
            if "budget_evaluator" in by_name and by_name["budget_evaluator"].passed:
                budget_pass += 1
            if "gold_resource_recall_evaluator" in by_name:
                gold_recall.append(by_name["gold_resource_recall_evaluator"].score)
            if "citation_evaluator" in by_name:
                citation_total += 1
                if by_name["citation_evaluator"].passed:
                    citation_pass += 1

        n = len(tasks) or 1
        rows.append(
            {
                "skill_id": skill.id,
                "avg_score": round(sum(aggregates) / len(aggregates), 4)
                if aggregates
                else 0.0,
                "budget_pass_rate": round(budget_pass / n, 4),
                "gold_recall": round(sum(gold_recall) / len(gold_recall), 4)
                if gold_recall
                else 0.0,
                "citation_pass_rate": round(citation_pass / citation_total, 4)
                if citation_total
                else 0.0,
            }
        )
    rows.sort(key=lambda r: -r["avg_score"])
    return rows


def print_table(domain: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\nSkill comparison for domain: {domain}\n")
    header = f"{'skill_id':<22} {'avg_score':>10} {'budget_pass':>12} {'gold_recall':>12} {'citation_pass':>14}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['skill_id']:<22} {r['avg_score']:>10} {r['budget_pass_rate']:>12} "
            f"{r['gold_recall']:>12} {r['citation_pass_rate']:>14}"
        )
    if rows:
        print(f"\nBest skill (outer loop): {rows[0]['skill_id']}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare skills on one domain")
    parser.add_argument("--domain", default="rag_doc_qa")
    parser.add_argument("--examples", default=_default_examples_dir())
    args = parser.parse_args(argv)

    rows = compare(args.examples, args.domain)
    print_table(args.domain, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
