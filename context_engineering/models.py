"""Core data models for the context-engineering framework.

These dataclasses are the vocabulary the whole system speaks in:

    c = F_s(x, K_s, R_s, U, B)

where ``x`` is a task, ``K_s`` are static resources, ``R_s`` are rules,
``U`` are user/session preferences, ``B`` is a token budget, ``F_s`` is a
skill's operator pipeline, and ``c`` is the final :class:`ContextPackage`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Token estimation                                                            #
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """Return a deterministic, dependency-free token estimate.

    Uses the common ``~4 characters per token`` heuristic.  It is *not* a real
    tokenizer, but it is stable, monotonic in length, and good enough to drive
    budget decisions in tests and demos.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


# --------------------------------------------------------------------------- #
# Static domain objects                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    """A single context-construction rule (part of ``R_s``)."""

    id: str
    description: str
    scopes: List[str] = field(default_factory=list)
    priority: int = 0
    source: str = "platform_default"
    enforcement_type: str = "prompt_instruction"
    status: str = "active"
    conflict_policy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """A reusable context-construction *recipe* (``s``).

    A skill is deliberately **not** an answer.  It names the operator pipeline
    ``F_s`` to run, the rule/resource scopes it draws from, a formatting
    template, a default token budget ``B``, and the evaluators used to grade
    the resulting context.
    """

    id: str
    description: str
    task_types: List[str] = field(default_factory=list)
    rule_scopes: List[str] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    operators: List[str] = field(default_factory=list)
    context_template: str = "default"
    default_budget_tokens: int = 1000
    evaluators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Resource:
    """A static knowledge item (part of ``K_s``)."""

    id: str
    domain: str
    type: str
    title: str
    content: str
    version: Optional[str] = None
    effective_date: Optional[str] = None  # ISO date string, e.g. "2026-01-01"
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dynamic / produced objects                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class ContextItem:
    """A candidate/selected piece of evidence flowing through the pipeline."""

    resource_id: str
    title: str
    content: str
    reason_selected: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPackage:
    """The final context ``c`` handed to an LLM, plus its provenance."""

    task: str
    skill_id: str
    selected_rules: List[Rule] = field(default_factory=list)
    selected_items: List[ContextItem] = field(default_factory=list)
    formatted_context: str = ""
    token_estimate: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    budget_tokens: int = 0  # the budget B this package was built against


@dataclass
class EvaluationResult:
    """Outcome of a single evaluator run against a :class:`ContextPackage`."""

    evaluator_name: str
    passed: bool
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """A task + its built context + its evaluation scores."""

    task_id: str
    domain: str
    skill_id: str
    context_package: ContextPackage
    evaluation_results: List[EvaluationResult] = field(default_factory=list)
    aggregate_score: float = 0.0


# --------------------------------------------------------------------------- #
# Pipeline state                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineState:
    """Mutable state threaded through the operator pipeline ``F_s``.

    Operators read from and write to this object.  It carries everything an
    operator might need: the task ``x``, the skill ``s``, the registries that
    expose ``K_s`` and the rule pool, user prefs ``U``, budget ``B``, and the
    growing set of selected rules/items plus a running trace.
    """

    task: str
    domain: str
    skill: Skill
    # Registries (typed loosely to avoid import cycles).
    resource_registry: Any = None
    rule_registry: Any = None
    operator_registry: Any = None
    # Preferences U and budget B.
    user_prefs: Dict[str, Any] = field(default_factory=dict)
    budget: int = 1000
    # Working sets.
    selected_rules: List[Rule] = field(default_factory=list)
    candidates: List[ContextItem] = field(default_factory=list)
    selected_items: List[ContextItem] = field(default_factory=list)
    stale_items: List[ContextItem] = field(default_factory=list)
    # Outputs.
    formatted_context: str = ""
    token_estimate: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def log(self, event: str, **details: Any) -> None:
        """Append a structured trace event.

        Traceability is a first-class goal: the point is not only to *produce*
        context but to make it obvious *why* each piece was produced.
        """
        self.trace.append({"event": event, **details})


# A dynamic operator F is any callable ``state -> state``.
Operator = Callable[[PipelineState], PipelineState]
