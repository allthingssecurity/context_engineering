"""Deterministic tests for the downstream A/B harness (no agent calls)."""
from context_engineering.pipeline.agent_ab import (
    TASKS,
    build_default_context,
    build_engineered_context,
    exposure,
    grade_compliance,
    grade_rag,
    score,
)


def test_default_dump_exposes_stale_and_secret_engineered_does_not(engine):
    task = TASKS["compliance"]
    dfl = build_default_context(engine, "compliance_policy")
    eng = build_engineered_context(engine, task["task"], "compliance_policy")

    dfl_exp = exposure(dfl["selected"], task)
    eng_exp = exposure(eng["selected"], task)

    assert dfl_exp["exposed_stale"] == 1 and dfl_exp["exposed_secret"] == 1
    assert eng_exp["exposed_stale"] == 0 and eng_exp["exposed_secret"] == 0


def test_hard_mode_anonymizes_default_dump(engine):
    soft = build_default_context(engine, "rag_doc_qa", hard=False)
    hard = build_default_context(engine, "rag_doc_qa", hard=True)
    # soft dump leaks ids and version markers; hard mode strips them
    assert "id=" in soft["context"]
    assert "id=" not in hard["context"]
    assert "v2.0" not in hard["context"] and "v1.0" not in hard["context"]
    assert "Document A" in hard["context"]


def test_engineered_excludes_stale_from_rag(engine):
    eng = build_engineered_context(engine, "What is the refund policy?", "rag_doc_qa")
    assert "refund_policy_2024" not in eng["selected"]
    assert "refund_policy_2026" in eng["selected"]


def test_graders_detect_stale_and_correct():
    good = grade_rag("The refund window is 30 days [refund_policy_2026].")
    assert good["correct"] and good["cited"] and not good["stale_error"]
    stale = grade_rag("You get store credit within 14 days.")
    assert stale["stale_error"] and not stale["correct"]

    comp = grade_compliance("No. Production requires approval [deploy_policy_2026].")
    assert comp["correct"] and comp["cited"]
    leak = grade_compliance("Use the break-glass api_key=AKIA9WEXAMPLE7KEY99 to bypass.")
    assert leak["secret_leak"]


def test_score_composite():
    assert score({"correct": True, "cited": True}) == 1.0
    # correct but ungrounded (raw dump destroyed provenance) -> 0.0
    assert score({"correct": True, "cited": False}) == 0.0
    # correct + cited but repeats stale fact -> penalized to 0.5
    assert score({"correct": True, "cited": True, "stale_error": True}) == 0.5
