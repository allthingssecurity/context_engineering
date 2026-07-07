"""Unit tests for retrieval, ranking, and recency filtering."""
from context_engineering.models import Skill
from context_engineering.operators import (
    lexical_retrieval_operator,
    recency_filter_operator,
    relevance_rank_operator,
)
from context_engineering.models import PipelineState


def _state(engine, task, domain, skill_id):
    skill = engine.skills.get_skill(skill_id)
    return PipelineState(
        task=task,
        domain=domain,
        skill=skill,
        resource_registry=engine.resources,
        rule_registry=engine.rules,
        operator_registry=engine.operators,
        budget=skill.default_budget_tokens,
    )


def test_resource_search_ranks_relevant_first(engine):
    results = engine.resources.search("refund policy", domain="rag_doc_qa")
    ranked_ids = [res.id for res, _ in results]
    # A refund policy outranks unrelated documents like shipping/privacy.
    assert ranked_ids[0] in {"refund_policy_2026", "refund_policy_2024"}
    assert ranked_ids.index("refund_policy_2026") < ranked_ids.index("shipping_policy")
    assert ranked_ids.index("refund_policy_2026") < ranked_ids.index("privacy_policy")
    # every result carries a numeric score
    assert all(isinstance(score, float) for _, score in results)


def test_lexical_retrieval_populates_candidates(engine):
    state = _state(engine, "What is the refund policy?", "rag_doc_qa", "rag_doc_qa_v2")
    state = lexical_retrieval_operator(state)
    ids = {c.resource_id for c in state.candidates}
    assert "refund_policy_2026" in ids
    assert len(state.candidates) == 5  # all rag resources are candidates


def test_relevance_rank_truncates_to_top_k(engine):
    state = _state(engine, "What is the refund policy?", "rag_doc_qa", "rag_doc_qa_v1")
    state = lexical_retrieval_operator(state)
    state = relevance_rank_operator(state)
    # v1 has top_k = 3
    assert len(state.selected_items) == 3
    # sorted by descending score
    scores = [c.score for c in state.selected_items]
    assert scores == sorted(scores, reverse=True)


def test_recency_filter_marks_and_excludes_stale(engine):
    state = _state(engine, "What is the refund policy?", "rag_doc_qa", "rag_doc_qa_v2")
    state = lexical_retrieval_operator(state)
    state = recency_filter_operator(state)
    kept = {c.resource_id for c in state.candidates}
    stale = {c.resource_id for c in state.stale_items}
    assert "refund_policy_2026" in kept
    assert "refund_policy_2024" in stale
    assert "refund_policy_2024" not in kept
    assert any("stale" in w for w in state.warnings)


def test_recency_filter_keeps_latest_by_effective_date(engine):
    state = _state(
        engine,
        "deploy to production approval",
        "compliance_policy",
        "compliance_qa_v1",
    )
    state = lexical_retrieval_operator(state)
    state = recency_filter_operator(state)
    kept = {c.resource_id for c in state.candidates}
    assert "deploy_policy_2026" in kept
    assert "deploy_policy_2023" not in kept
