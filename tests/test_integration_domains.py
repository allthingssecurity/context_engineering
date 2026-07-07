"""Integration tests: one end-to-end scenario per domain."""
from context_engineering.loaders import load_domain_bundle
from context_engineering.run_experiments import run_all


def _ids(pkg):
    return {i.resource_id for i in pkg.selected_items}


def test_rag_selects_latest_policy_and_cites(engine, examples_dir):
    pkg = engine.build_context(
        "What is the refund policy?", domain_hint="rag_doc_qa"
    )
    ids = _ids(pkg)
    # latest refund policy + supporting FAQ selected; stale version excluded
    assert "refund_policy_2026" in ids
    assert "faq_refunds" in ids
    assert "refund_policy_2024" not in ids
    # citation markers present
    assert "[refund_policy_2026" in pkg.formatted_context
    # stale version noted in trace
    assert any(e["event"] == "stale_filtered" for e in pkg.trace)


def test_coding_includes_test_trace_and_source(engine):
    pkg = engine.build_context(
        "Fix refund bug for UPI payment.", domain_hint="coding_bugfix"
    )
    ids = _ids(pkg)
    assert "refund.py" in ids
    assert "test_refund.py" in ids
    assert "stack_trace.log" in ids
    # unrelated files excluded
    assert "utils.py" not in ids
    assert "README.md" not in ids
    # public-API preservation rule is present
    assert any(r.id == "code.preserve_public_api" for r in pkg.selected_rules)


def test_paper_includes_equation_method_and_limitations(engine):
    pkg = engine.build_context(
        "Explain the equation and method in this paper.",
        domain_hint="paper_explanation",
    )
    ids = _ids(pkg)
    assert "method" in ids
    assert "equations" in ids
    assert "limitations" in ids
    # equations were actually extracted
    eq_item = next(i for i in pkg.selected_items if i.resource_id == "equations")
    assert eq_item.metadata.get("equations")


def test_compliance_excludes_stale_and_includes_matrix(engine):
    pkg = engine.build_context(
        "Can a developer deploy to production without approval?",
        domain_hint="compliance_policy",
    )
    ids = _ids(pkg)
    assert "deploy_policy_2026" in ids
    assert "approval_matrix" in ids
    assert "deploy_policy_2023" not in ids
    assert any(r.id == "policy.cite_section" for r in pkg.selected_rules)


def test_run_all_every_task_fully_passes(examples_dir):
    results = run_all(examples_dir)
    assert len(results) >= 5
    for res in results:
        assert res.evaluation_results
        for ev in res.evaluation_results:
            assert ev.passed, f"{res.task_id}:{ev.evaluator_name} failed -> {ev.details}"
