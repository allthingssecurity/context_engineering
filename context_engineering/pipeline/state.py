"""Production pipeline state and value objects.

The whole production pipeline threads a single :class:`ContextBuildState`
through a list of :class:`~context_engineering.pipeline.base.Operator` instances.
Every operator reads the state, mutates it, appends a trace event, and emits an
:class:`OperatorMetrics` record — never silently failing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Query:
    """A single retrieval query (a rewriter may produce several)."""

    text: str
    kind: str = "original"  # original | keyword | synonym | entity | hyde | symbol
    weight: float = 1.0


@dataclass
class CandidateResource:
    """A retrieved candidate, carrying every score a stage assigns it."""

    resource_id: str
    title: str
    content: str
    type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sparse_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    authority_score: float = 0.0
    rerank_score: float = 0.0
    score: float = 0.0  # current working score used for ordering
    stale: bool = False
    reasons: List[str] = field(default_factory=list)

    def add_reason(self, reason: str) -> None:
        self.reasons.append(reason)

    def copy(self) -> "CandidateResource":
        return _clone(self)


def _clone(c: "CandidateResource") -> "CandidateResource":
    return CandidateResource(
        resource_id=c.resource_id,
        title=c.title,
        content=c.content,
        type=c.type,
        metadata=dict(c.metadata),
        sparse_score=c.sparse_score,
        dense_score=c.dense_score,
        fused_score=c.fused_score,
        authority_score=c.authority_score,
        rerank_score=c.rerank_score,
        score=c.score,
        stale=c.stale,
        reasons=list(c.reasons),
    )


@dataclass
class TraceEvent:
    operator: str
    version: str
    event: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorMetrics:
    operator: str
    version: str
    duration_ms: float
    candidates_in: int
    candidates_out: int
    error: Optional[str] = None


@dataclass
class Budget:
    """Token budget ``B`` with optional per-section quotas."""

    total_tokens: int
    section_quotas: Dict[str, int] = field(default_factory=dict)
    spent: int = 0


@dataclass
class ContextBuildState:
    """Mutable state threaded through the production operator pipeline."""

    task: str
    domain: Optional[str] = None
    # U — user/session context: preferences, tenant, clearances, acl.
    user_context: Dict[str, Any] = field(default_factory=dict)
    # Registries exposing K_s / rules / skills.
    resources: Any = None
    rules: Any = None
    skills: Any = None
    # Routing / planning outputs.
    route: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    selected_skill: Any = None
    selected_rules: List[Any] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)
    queries: List[Query] = field(default_factory=list)
    # Retrieval / ranking working sets.
    candidates: List[CandidateResource] = field(default_factory=list)
    retrieved_order: List[str] = field(default_factory=list)  # post-retrieval ids
    ranked_order: List[str] = field(default_factory=list)  # post-rerank ids (pre-truncate)
    selected_items: List[CandidateResource] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    # Outputs / observability.
    formatted_context: str = ""
    token_estimate: int = 0
    validation: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    trace: List[TraceEvent] = field(default_factory=list)
    metrics: List[OperatorMetrics] = field(default_factory=list)
    budget: Optional[Budget] = None

    def log(self, operator: str, version: str, event: str, **details: Any) -> None:
        self.trace.append(TraceEvent(operator, version, event, details))
