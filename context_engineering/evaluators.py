"""Evaluators — grade a :class:`ContextPackage` against a task's gold spec.

Each evaluator has the signature
``(package: ContextPackage, spec: dict) -> EvaluationResult`` and is pure /
deterministic.  ``spec`` is the per-task gold entry, e.g.::

    {
      "gold_resources": ["refund_policy_2026", "faq_refunds"],
      "required_rules": ["rag.require_citations"],
      "stale_resources": ["refund_policy_2024"],
      "must_contain": ["equation"],
    }
"""
from __future__ import annotations

from typing import Any, Dict

from .models import ContextPackage, EvaluationResult
from .registries import EvaluatorRegistry


def _selected_ids(package: ContextPackage) -> set:
    return {item.resource_id for item in package.selected_items}


def budget_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Pass iff the token estimate is within the package's budget ``B``."""
    budget = package.budget_tokens or spec.get("budget_tokens", 0)
    passed = budget <= 0 or package.token_estimate <= budget
    return EvaluationResult(
        evaluator_name="budget_evaluator",
        passed=passed,
        score=1.0 if passed else 0.0,
        details={"token_estimate": package.token_estimate, "budget": budget},
    )


def required_rule_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Pass iff every required rule id is present in the selected rules."""
    required = set(spec.get("required_rules", []))
    present = {r.id for r in package.selected_rules}
    missing = sorted(required - present)
    passed = not missing
    score = 1.0 if not required else (len(required & present) / len(required))
    return EvaluationResult(
        evaluator_name="required_rule_evaluator",
        passed=passed,
        score=round(score, 4),
        details={"required": sorted(required), "missing": missing},
    )


def gold_resource_recall_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Fraction of expected gold resources actually included."""
    gold = set(spec.get("gold_resources", []))
    selected = _selected_ids(package)
    hit = gold & selected
    score = 1.0 if not gold else len(hit) / len(gold)
    passed = gold.issubset(selected)
    return EvaluationResult(
        evaluator_name="gold_resource_recall_evaluator",
        passed=passed,
        score=round(score, 4),
        details={
            "gold": sorted(gold),
            "selected": sorted(selected),
            "missing": sorted(gold - selected),
        },
    )


def stale_resource_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Pass iff stale resources are excluded (or explicitly marked stale)."""
    stale = set(spec.get("stale_resources", []))
    selected = _selected_ids(package)
    text = package.formatted_context.lower()
    violations = []
    for sid in stale:
        if sid in selected:
            # Included — only acceptable if clearly marked stale in the text.
            marked = f"{sid.lower()}" in text and "stale" in text
            if not marked:
                violations.append(sid)
    passed = not violations
    score = 1.0 if not stale else (len(stale) - len(violations)) / len(stale)
    return EvaluationResult(
        evaluator_name="stale_resource_evaluator",
        passed=passed,
        score=round(score, 4),
        details={"stale": sorted(stale), "violations": violations},
    )


def citation_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Pass iff every selected item carries a citation marker in the context."""
    text = package.formatted_context
    if not package.selected_items:
        return EvaluationResult(
            evaluator_name="citation_evaluator",
            passed=False,
            score=0.0,
            details={"reason": "no items to cite"},
        )
    cited = 0
    for item in package.selected_items:
        marker = f"[{item.resource_id}"
        if marker in text:
            cited += 1
    score = cited / len(package.selected_items)
    passed = score == 1.0
    return EvaluationResult(
        evaluator_name="citation_evaluator",
        passed=passed,
        score=round(score, 4),
        details={"cited": cited, "total": len(package.selected_items)},
    )


def coding_context_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Check the source file, failing test and stack trace are all present.

    Reads the expectation from ``spec['coding']`` with keys
    ``source_file``, ``failing_test`` and ``stack_trace`` (resource ids), and
    from ``spec.get('unrelated_resources')`` for the exclusion check.
    """
    coding = spec.get("coding", {})
    selected = _selected_ids(package)
    checks = {
        "source_file": coding.get("source_file") in selected
        if coding.get("source_file")
        else True,
        "failing_test": coding.get("failing_test") in selected
        if coding.get("failing_test")
        else True,
        "stack_trace": coding.get("stack_trace") in selected
        if coding.get("stack_trace")
        else True,
    }
    unrelated = set(spec.get("unrelated_resources", []))
    excluded_ok = not (unrelated & selected)
    checks["unrelated_excluded"] = excluded_ok

    passed_count = sum(1 for v in checks.values() if v)
    score = passed_count / len(checks)
    return EvaluationResult(
        evaluator_name="coding_context_evaluator",
        passed=all(checks.values()),
        score=round(score, 4),
        details={"checks": checks, "unrelated_included": sorted(unrelated & selected)},
    )


def paper_context_evaluator(package: ContextPackage, spec: Dict[str, Any]) -> EvaluationResult:
    """Check equation, method section, symbol rule and limitations are present."""
    selected = _selected_ids(package)
    text = package.formatted_context.lower()
    paper = spec.get("paper", {})

    has_equation = any(
        item.metadata.get("equations") for item in package.selected_items
    )
    checks = {
        "equation_included": has_equation,
        "method_included": (paper.get("method") in selected)
        if paper.get("method")
        else ("method" in text),
        "limitations_included": (paper.get("limitations") in selected)
        if paper.get("limitations")
        else ("limitation" in text),
        "symbol_rule_present": any(
            "symbol" in r.description.lower() or "symbol" in r.id.lower()
            for r in package.selected_rules
        ),
    }
    passed_count = sum(1 for v in checks.values() if v)
    score = passed_count / len(checks)
    return EvaluationResult(
        evaluator_name="paper_context_evaluator",
        passed=all(checks.values()),
        score=round(score, 4),
        details={"checks": checks},
    )


def register_default_evaluators(registry: EvaluatorRegistry) -> EvaluatorRegistry:
    """Register every built-in evaluator into ``registry`` and return it."""
    registry.register_evaluator("budget_evaluator", budget_evaluator)
    registry.register_evaluator("required_rule_evaluator", required_rule_evaluator)
    registry.register_evaluator(
        "gold_resource_recall_evaluator", gold_resource_recall_evaluator
    )
    registry.register_evaluator("stale_resource_evaluator", stale_resource_evaluator)
    registry.register_evaluator("citation_evaluator", citation_evaluator)
    registry.register_evaluator("coding_context_evaluator", coding_context_evaluator)
    registry.register_evaluator("paper_context_evaluator", paper_context_evaluator)
    return registry
