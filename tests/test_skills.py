"""Unit tests for skill selection."""
from context_engineering.operators import classify_task


def test_select_skill_by_domain_hint_prefers_default(engine):
    skill = engine.skills.select_skill("anything", domain_hint="rag_doc_qa")
    # v2 is flagged as the default skill for the rag domain.
    assert skill.id == "rag_doc_qa_v2"


def test_select_skill_falls_back_to_classification(engine):
    skill = engine.select_skill("Fix refund bug for UPI payment.")
    assert skill.id == "coding_bugfix_v1"


def test_classify_task_domains():
    assert classify_task("What is the refund policy?") == "rag_doc_qa"
    assert classify_task("Fix refund bug for UPI payment.") == "coding_bugfix"
    assert classify_task("Explain the equation and method in this paper.") == "paper_explanation"
    assert (
        classify_task("Can a developer deploy to production without approval?")
        == "compliance_policy"
    )


def test_get_skill_unknown_raises(engine):
    import pytest

    with pytest.raises(KeyError):
        engine.skills.get_skill("does_not_exist")
