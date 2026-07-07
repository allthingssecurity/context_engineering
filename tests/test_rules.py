"""Unit tests for rule selection and conflict resolution."""
from context_engineering.models import Rule
from context_engineering.registries import RuleRegistry


def test_select_rules_by_scope(engine):
    rules = engine.rules.select_rules(["rag"])
    ids = {r.id for r in rules}
    assert "rag.require_citations" in ids
    assert "rag.prefer_latest_version" in ids
    # A coding rule must not leak into the rag scope.
    assert "code.preserve_public_api" not in ids


def test_select_rules_filters_status():
    reg = RuleRegistry()
    reg.register_rule(Rule(id="a", description="active", scopes=["x"], status="active"))
    reg.register_rule(
        Rule(id="b", description="experimental", scopes=["x"], status="experimental")
    )
    active = reg.select_rules(["x"], status="active")
    assert {r.id for r in active} == {"a"}
    any_status = reg.select_rules(["x"], status=None)
    assert {r.id for r in any_status} == {"a", "b"}


def test_rules_ordered_by_priority():
    reg = RuleRegistry()
    reg.register_rule(Rule(id="low", description="", scopes=["s"], priority=10))
    reg.register_rule(Rule(id="high", description="", scopes=["s"], priority=90))
    ordered = reg.select_rules(["s"])
    assert [r.id for r in ordered] == ["high", "low"]


def test_conflict_resolution_keeps_highest_priority():
    reg = RuleRegistry()
    reg.register_rule(
        Rule(
            id="r_lenient",
            description="allow",
            scopes=["s"],
            priority=10,
            metadata={"conflict_group": "deploy"},
        )
    )
    reg.register_rule(
        Rule(
            id="r_strict",
            description="deny",
            scopes=["s"],
            priority=80,
            metadata={"conflict_group": "deploy"},
        )
    )
    selected = reg.select_rules(["s"])
    ids = {r.id for r in selected}
    assert ids == {"r_strict"}


def test_conflict_resolution_passes_through_ungrouped():
    reg = RuleRegistry()
    reg.register_rule(Rule(id="a", description="", scopes=["s"], priority=5))
    reg.register_rule(Rule(id="b", description="", scopes=["s"], priority=6))
    selected = reg.resolve_conflicts(reg.list_rules())
    assert {r.id for r in selected} == {"a", "b"}
