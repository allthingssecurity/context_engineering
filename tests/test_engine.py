"""Unit tests for budget compression, formatting, and full context builds."""
from context_engineering.models import (
    ContextItem,
    PipelineState,
    Skill,
    estimate_tokens,
)
from context_engineering.operators import (
    budget_compressor_operator,
    context_formatter_operator,
)


def _bare_state(skill, budget):
    return PipelineState(task="t", domain="d", skill=skill, budget=budget)


def test_budget_compressor_reduces_tokens():
    skill = Skill(id="s", description="")
    long_text = " ".join(f"Sentence number {i}." for i in range(200))
    state = _bare_state(skill, budget=20)
    state.selected_items = [
        ContextItem(resource_id="r1", title="r1", content=long_text)
    ]
    before = estimate_tokens(long_text)
    state = budget_compressor_operator(state)
    after = sum(estimate_tokens(i.content) for i in state.selected_items)
    assert after < before


def test_budget_compressor_drops_when_cannot_shrink():
    skill = Skill(id="s", description="")
    state = _bare_state(skill, budget=1)
    state.selected_items = [
        ContextItem(resource_id="a", title="a", content="one two three four five six."),
        ContextItem(resource_id="b", title="b", content="seven eight nine ten eleven."),
    ]
    state = budget_compressor_operator(state)
    total = sum(estimate_tokens(i.content) for i in state.selected_items)
    assert total <= 1 or len(state.selected_items) < 2
    assert any("dropped" in w for w in state.warnings)


def test_context_formatter_includes_task_rules_evidence():
    skill = Skill(id="s", description="")
    state = _bare_state(skill, budget=1000)
    state.selected_items = [
        ContextItem(resource_id="r1", title="Doc 1", content="Some evidence here.")
    ]
    state = context_formatter_operator(state)
    text = state.formatted_context
    assert "=== TASK ===" in text
    assert "=== RULES ===" in text
    assert "=== EVIDENCE ===" in text
    assert "[r1]" in text
    assert state.token_estimate == estimate_tokens(text)


def test_context_formatter_flags_missing_evidence():
    skill = Skill(id="s", description="")
    state = _bare_state(skill, budget=1000)
    state = context_formatter_operator(state)
    assert "Evidence is missing" in state.formatted_context
    assert any("no evidence" in w for w in state.warnings)


def test_build_context_produces_full_package(engine):
    pkg = engine.build_context(
        "What is the refund policy?", domain_hint="rag_doc_qa"
    )
    assert pkg.skill_id == "rag_doc_qa_v2"
    assert pkg.selected_rules  # R_s populated
    assert pkg.selected_items  # evidence selected
    assert pkg.formatted_context
    assert pkg.token_estimate > 0
    assert pkg.budget_tokens == 700
    # trace records the key pipeline events
    events = {e["event"] for e in pkg.trace}
    assert "skill_selected" in events
    assert "rules_selected" in events
    assert "retrieved_candidates" in events
    assert "formatted" in events


def test_build_context_respects_explicit_budget(engine):
    pkg = engine.build_context(
        "What is the refund policy?", domain_hint="rag_doc_qa", budget=50
    )
    assert pkg.budget_tokens == 50


def test_build_context_with_explicit_skill(engine):
    pkg = engine.build_context(
        "What is the refund policy?",
        domain_hint="rag_doc_qa",
        skill_id="rag_doc_qa_v1",
    )
    assert pkg.skill_id == "rag_doc_qa_v1"
