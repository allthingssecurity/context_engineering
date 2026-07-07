"""Unit tests for the production pipeline layer."""
import pytest

from context_engineering.pipeline.base import (
    Operator,
    OperatorRegistry,
    Pipeline,
    build_pipeline,
)
from context_engineering.pipeline.metrics import mrr, ndcg_at_k, recall_at_k
from context_engineering.pipeline.registry import PIPELINES, default_registry
from context_engineering.pipeline.state import ContextBuildState


def _state(engine, task, domain, user_context=None):
    return ContextBuildState(
        task=task,
        domain=domain,
        user_context=user_context or {},
        resources=engine.resources,
        rules=engine.rules,
        skills=engine.skills,
    )


def _run(engine, spec, task, domain, user_context=None):
    pipeline = build_pipeline(default_registry(), spec)
    return pipeline.run(_state(engine, task, domain, user_context))


# --------------------------------------------------------------------------- #
# Interface + registry                                                        #
# --------------------------------------------------------------------------- #
def test_operator_run_records_metrics(engine):
    state = _run(engine, ["task_router", "skill_planner"], "What is the refund policy?", "rag_doc_qa")
    assert len(state.metrics) == 2
    assert {m.operator for m in state.metrics} == {"task_router", "skill_planner"}
    assert all(m.error is None for m in state.metrics)


def test_operator_never_silently_fails(engine):
    class Boom(Operator):
        name = "boom"
        version = "0.0"

        def _run(self, state):
            raise ValueError("kaboom")

    reg = default_registry()
    reg.register("boom", Boom)
    pipeline = Pipeline([reg.create("task_router"), reg.create("boom"), reg.create("skill_planner")])
    state = pipeline.run(_state(engine, "What is the refund policy?", "rag_doc_qa"))
    # error captured, pipeline continued to the next operator
    assert any("kaboom" in e for e in state.errors)
    assert state.metrics[-1].operator == "skill_planner"
    assert any(m.error for m in state.metrics)


def test_registry_is_pluggable(engine):
    class TagOperator(Operator):
        name = "tagger"
        version = "1.0"

        def _run(self, state):
            state.user_context["tagged"] = True
            return state

    reg = default_registry()
    reg.register("tagger", TagOperator)
    assert "tagger" in reg.available()
    pipeline = build_pipeline(reg, ["task_router", "tagger"])
    state = pipeline.run(_state(engine, "x", "rag_doc_qa"))
    assert state.user_context.get("tagged") is True


# --------------------------------------------------------------------------- #
# Retrieval / ranking operators                                               #
# --------------------------------------------------------------------------- #
def test_hybrid_retriever_dense_adds_signal(engine):
    spec = ["task_router", "skill_planner", "query_rewriter",
            ("hybrid_retriever", {"sparse": True, "dense": True})]
    state = _run(engine, spec, "What is the refund policy?", "rag_doc_qa")
    assert state.candidates
    assert any(c.dense_score != 0.0 for c in state.candidates)
    assert state.retrieved_order  # populated for recall metrics


def test_reranker_sets_scores_and_orders(engine):
    spec = ["task_router", "skill_planner", "query_rewriter",
            "hybrid_retriever", "cross_encoder_reranker"]
    state = _run(engine, spec, "Fix refund bug for UPI payment.", "coding_bugfix")
    assert all(hasattr(c, "rerank_score") for c in state.candidates)
    scores = [c.score for c in state.candidates]
    assert scores == sorted(scores, reverse=True)


def test_permission_filter_blocks_confidential_without_clearance(engine):
    spec = ["task_router", "skill_planner", "scope_resolver", "query_rewriter",
            "hybrid_retriever", "permission_filter"]
    task = "Can a developer deploy to production without approval?"
    without = _run(engine, spec, task, "compliance_policy")
    ids = {c.resource_id for c in without.candidates}
    assert "deploy_runbook_confidential" not in ids

    with_clear = _run(engine, spec, task, "compliance_policy",
                      user_context={"clearances": ["sre_oncall"]})
    ids2 = {c.resource_id for c in with_clear.candidates}
    assert "deploy_runbook_confidential" in ids2


def test_conflict_detector_marks_and_drops_stale(engine):
    spec = ["task_router", "skill_planner", "scope_resolver", "query_rewriter",
            "hybrid_retriever", "permission_filter", "conflict_detector"]
    state = _run(engine, spec, "What is the refund policy?", "rag_doc_qa")
    ids = {c.resource_id for c in state.candidates}
    assert "refund_policy_2024" not in ids
    assert state.conflicts
    assert any(cf["superseded"] == "refund_policy_2024" for cf in state.conflicts)


# --------------------------------------------------------------------------- #
# Assembly + validation                                                       #
# --------------------------------------------------------------------------- #
def test_xml_formatter_has_citations_and_contract(engine):
    state = _run(engine, PIPELINES["full_production"], "What is the refund policy?", "rag_doc_qa")
    text = state.formatted_context
    assert "<task>" in text and "<output_contract>" in text
    assert '<source id="refund_policy_2026"' in text
    assert state.token_estimate > 0


def test_validator_flags_secret_leak(engine):
    # removing the permission filter lets the confidential secret through
    state = _run(engine, PIPELINES["full_minus_permission"],
                 "Can a developer deploy to production without approval?",
                 "compliance_policy")
    if "deploy_runbook_confidential" in {c.resource_id for c in state.selected_items}:
        assert state.validation["checks"]["no_secrets"] is False
        assert any("SECURITY" in e for e in state.errors)


def test_validator_passes_on_clean_full_pipeline(engine):
    state = _run(engine, PIPELINES["full_production"], "What is the refund policy?", "rag_doc_qa")
    assert state.validation["passed"] is True


# --------------------------------------------------------------------------- #
# Metric functions                                                            #
# --------------------------------------------------------------------------- #
def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "d"}, 3) == 0.5
    assert recall_at_k(["a", "b"], {"a", "b"}, 5) == 1.0


def test_mrr():
    assert mrr(["x", "y", "gold"], {"gold"}) == pytest.approx(1 / 3)
    assert mrr(["gold"], {"gold"}) == 1.0
    assert mrr(["a", "b"], {"gold"}) == 0.0


def test_ndcg_perfect_and_imperfect():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 5) == pytest.approx(1.0)
    worse = ndcg_at_k(["x", "a"], {"a"}, 5)
    better = ndcg_at_k(["a", "x"], {"a"}, 5)
    assert better > worse
