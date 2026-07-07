"""Default operator registry and named pipeline specs.

``default_registry()`` returns a pluggable :class:`OperatorRegistry` with every
production operator registered.  Adding your own is one line::

    reg = default_registry()
    reg.register("my_reranker", MyReranker)   # a subclass of Operator
    # then reference "my_reranker" in any pipeline spec

``PIPELINES`` defines named specs used by the benchmark's ablation study.
"""
from __future__ import annotations

from typing import Dict, List

from .base import OperatorRegistry, Pipeline, SpecEntry, build_pipeline
from . import operators as ops


def default_registry() -> OperatorRegistry:
    """A registry with all built-in production operators registered."""
    reg = OperatorRegistry()
    reg.register("task_router", ops.TaskRouter)
    reg.register("constraint_extractor", ops.ConstraintExtractor)
    reg.register("skill_planner", ops.SkillPlanner)
    reg.register("scope_resolver", ops.ScopeResolver)
    reg.register("query_rewriter", ops.QueryRewriter)
    reg.register("hybrid_retriever", ops.HybridRetriever)
    reg.register("permission_filter", ops.PermissionFilter)
    reg.register("authority_freshness_scorer", ops.AuthorityFreshnessScorer)
    reg.register("conflict_detector", ops.ConflictDetector)
    reg.register("cross_encoder_reranker", ops.CrossEncoderReranker)
    reg.register("mmr_diversity_selector", ops.MMRDiversitySelector)
    reg.register("evidence_selector", ops.EvidenceSelector)
    reg.register("query_aware_compressor", ops.QueryAwareCompressor)
    reg.register("budget_allocator", ops.BudgetAllocator)
    reg.register("context_orderer", ops.ContextOrderer)
    reg.register("xml_formatter", ops.XMLFormatter)
    reg.register("simple_formatter", ops.SimpleFormatter)
    reg.register("context_validator", ops.ContextValidator)
    reg.register("telemetry", ops.Telemetry)
    return reg


TOP_K = 4

# The full production pipeline (all top operators).
FULL_SPEC: List[SpecEntry] = [
    "task_router",
    "constraint_extractor",
    "skill_planner",
    "scope_resolver",
    "query_rewriter",
    ("hybrid_retriever", {"sparse": True, "dense": True}),
    "permission_filter",
    "authority_freshness_scorer",
    "conflict_detector",
    "cross_encoder_reranker",
    "mmr_diversity_selector",
    ("evidence_selector", {"top_k": TOP_K}),
    "query_aware_compressor",
    "budget_allocator",
    "context_orderer",
    "xml_formatter",
    "context_validator",
    "telemetry",
]

# The naive baseline: sparse retrieval -> top-k -> plain formatting.
NAIVE_SPEC: List[SpecEntry] = [
    "task_router",
    "skill_planner",
    ("hybrid_retriever", {"sparse": True, "dense": False}),
    ("evidence_selector", {"top_k": TOP_K}),
    "simple_formatter",
]


def _without(spec: List[SpecEntry], *names: str) -> List[SpecEntry]:
    out = []
    for e in spec:
        n = e if isinstance(e, str) else e[0]
        if n not in names:
            out.append(e)
    return out


def _with_hybrid(spec: List[SpecEntry], dense: bool) -> List[SpecEntry]:
    out = []
    for e in spec:
        n = e if isinstance(e, str) else e[0]
        if n == "hybrid_retriever":
            out.append(("hybrid_retriever", {"sparse": True, "dense": dense}))
        else:
            out.append(e)
    return out


# Named pipelines for the ablation benchmark: a baseline, the full stack, and
# leave-one-out variants that each remove a single high-value operator.
PIPELINES: Dict[str, List[SpecEntry]] = {
    "naive_baseline": NAIVE_SPEC,
    "full_production": FULL_SPEC,
    "full_minus_reranker": _without(FULL_SPEC, "cross_encoder_reranker"),
    "full_minus_dense": _with_hybrid(FULL_SPEC, dense=False),
    "full_minus_conflict": _without(FULL_SPEC, "conflict_detector"),
    "full_minus_permission": _without(FULL_SPEC, "permission_filter"),
    "full_minus_compress_budget": _without(
        FULL_SPEC, "query_aware_compressor", "budget_allocator"
    ),
}


def make_pipeline(name: str, registry: OperatorRegistry | None = None) -> Pipeline:
    reg = registry or default_registry()
    return build_pipeline(reg, PIPELINES[name])
