"""Unit tests for evaluator behavior."""
from context_engineering.evaluators import (
    budget_evaluator,
    citation_evaluator,
    gold_resource_recall_evaluator,
    required_rule_evaluator,
    stale_resource_evaluator,
)
from context_engineering.models import ContextItem, ContextPackage, Rule


def _pkg(**kwargs):
    base = dict(task="t", skill_id="s")
    base.update(kwargs)
    return ContextPackage(**base)


def test_budget_evaluator_pass_and_fail():
    pkg = _pkg(token_estimate=100, budget_tokens=200)
    assert budget_evaluator(pkg, {}).passed
    pkg2 = _pkg(token_estimate=300, budget_tokens=200)
    assert not budget_evaluator(pkg2, {}).passed


def test_required_rule_evaluator():
    pkg = _pkg(selected_rules=[Rule(id="r1", description=""), Rule(id="r2", description="")])
    ok = required_rule_evaluator(pkg, {"required_rules": ["r1"]})
    assert ok.passed and ok.score == 1.0
    bad = required_rule_evaluator(pkg, {"required_rules": ["r1", "r3"]})
    assert not bad.passed
    assert bad.details["missing"] == ["r3"]


def test_gold_resource_recall_evaluator():
    pkg = _pkg(
        selected_items=[
            ContextItem(resource_id="a", title="a", content=""),
            ContextItem(resource_id="b", title="b", content=""),
        ]
    )
    full = gold_resource_recall_evaluator(pkg, {"gold_resources": ["a", "b"]})
    assert full.passed and full.score == 1.0
    partial = gold_resource_recall_evaluator(pkg, {"gold_resources": ["a", "c"]})
    assert not partial.passed and partial.score == 0.5


def test_stale_resource_evaluator_excluded_passes():
    pkg = _pkg(
        selected_items=[ContextItem(resource_id="new", title="new", content="")],
        formatted_context="only new content here",
    )
    res = stale_resource_evaluator(pkg, {"stale_resources": ["old"]})
    assert res.passed


def test_stale_resource_evaluator_included_unmarked_fails():
    pkg = _pkg(
        selected_items=[ContextItem(resource_id="old", title="old", content="x")],
        formatted_context="[old] content with no marker",
    )
    res = stale_resource_evaluator(pkg, {"stale_resources": ["old"]})
    assert not res.passed
    assert res.details["violations"] == ["old"]


def test_citation_evaluator():
    pkg = _pkg(
        selected_items=[ContextItem(resource_id="a", title="a", content="")],
        formatted_context="[a §1] evidence",
    )
    assert citation_evaluator(pkg, {}).passed
    pkg2 = _pkg(
        selected_items=[ContextItem(resource_id="a", title="a", content="")],
        formatted_context="no citation marker",
    )
    assert not citation_evaluator(pkg2, {}).passed


def test_evaluators_run_via_engine(engine):
    pkg = engine.build_context("What is the refund policy?", domain_hint="rag_doc_qa")
    spec = {
        "gold_resources": ["refund_policy_2026", "faq_refunds"],
        "required_rules": ["rag.require_citations"],
        "stale_resources": ["refund_policy_2024"],
    }
    results = engine.evaluate(pkg, spec)
    names = {r.evaluator_name for r in results}
    assert "citation_evaluator" in names
    assert all(r.passed for r in results)
