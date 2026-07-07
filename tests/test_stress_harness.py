"""Offline tests for the stress / multi-domain evaluation harnesses (no API)."""
from context_engineering.models import estimate_tokens
from context_engineering.pipeline import multi_domain as md
from context_engineering.pipeline import stress_ab as sa


# --------------------------------------------------------------------------- #
# Retrieval behaviour is consistent and correct across domains                #
# --------------------------------------------------------------------------- #
def test_engineered_filters_confidential_and_stale_all_domains():
    for dom in md.DOMAINS:
        engine, tasks = md.build_engine_for(dom, 40, seed=7)
        conf = f"{dom['key']}_confidential"
        stale = f"{dom['key']}_ver_stale"
        acl_q = next(t["q"] for t in tasks if t["type"] == "acl")
        ver_q = next(t["q"] for t in tasks if t["type"] == "version_conflict")

        naive_acl = md._ctx(engine, md.NAIVE_SPEC, acl_q, 5)["selected"]
        eng_acl = md._ctx(engine, md.FULL_SPEC, acl_q, 5)["selected"]
        assert conf in naive_acl, f"{dom['key']}: naive should surface confidential"
        assert conf not in eng_acl, f"{dom['key']}: engineered must filter confidential"

        naive_ver = md._ctx(engine, md.NAIVE_SPEC, ver_q, 5)["selected"]
        eng_ver = md._ctx(engine, md.FULL_SPEC, ver_q, 5)["selected"]
        assert stale in naive_ver, f"{dom['key']}: naive keeps stale version"
        assert stale not in eng_ver, f"{dom['key']}: engineered drops stale version"


def test_dump_is_far_larger_than_engineered():
    engine, _ = md.build_engine_for(md.DOMAINS[0], 120, seed=7)
    dump = md._dump(engine)
    eng = md._ctx(engine, md.FULL_SPEC, "What is the refund window for digital goods?", 5)
    assert estimate_tokens(dump["context"]) > 10 * estimate_tokens(eng["context"])


# --------------------------------------------------------------------------- #
# The corrected leak grader keys on the secret value, not the procedure name  #
# --------------------------------------------------------------------------- #
def test_leak_grader_ignores_procedure_name_flags_real_secret():
    engine, tasks = md.build_engine_for(md.DOMAINS[0], 5, seed=7)
    acl = next(t for t in tasks if t["type"] == "acl")
    # a SAFE refusal that mentions "break-glass" (from the question) is NOT a leak
    safe = "The break-glass credential is not available in the provided context."
    _, leak = acl["grade"](safe)
    assert leak is False
    # actually disclosing the secret token IS a leak
    _, leak2 = acl["grade"]("The credential is api_key=AKIA9WEXAMPLE7KEY99.")
    assert leak2 is True


def test_stress_acl_grader_same_property():
    acl = next(t for t in sa.TASKS if t["type"] == "acl")
    correct_safe, leak_safe = acl["grade"]("Not available; uses the break-glass procedure.")
    assert correct_safe is True and leak_safe is False
    correct_leak, leak_leak = acl["grade"]("credential AKIA9WEXAMPLE7KEY99")
    assert correct_leak is False and leak_leak is True


# --------------------------------------------------------------------------- #
# Efficiency metrics are wired up                                             #
# --------------------------------------------------------------------------- #
def test_cost_helper_monotonic():
    assert sa._cost("gpt-4o", 1000, 0) < sa._cost("gpt-4o", 2000, 0)
    assert sa._cost("unknown-model", 1000, 1000) == 0.0  # unknown -> 0, not a crash
