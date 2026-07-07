"""Production-grade context pipeline layer.

A stable, pluggable operator interface and the full stage pipeline:

    task → route → constrain → plan → scope → rewrite → retrieve(hybrid) →
    permission-filter → authority/freshness → conflict → rerank → diversify →
    select → compress → budget → order → format → validate → telemetry

Every stage is an :class:`~context_engineering.pipeline.base.Operator` with a
stable ``name``/``version``/``run`` contract, registered in a pluggable
:class:`~context_engineering.pipeline.base.OperatorRegistry`.  Local, offline,
deterministic defaults ship for every stage, with clearly-marked seams for
vendor models (rerankers, embeddings, LLM routers).
"""
from .base import Operator, OperatorRegistry, Pipeline, build_pipeline
from .registry import PIPELINES, default_registry, make_pipeline
from .state import (
    Budget,
    CandidateResource,
    ContextBuildState,
    OperatorMetrics,
    Query,
    TraceEvent,
)

__all__ = [
    "Operator",
    "OperatorRegistry",
    "Pipeline",
    "build_pipeline",
    "PIPELINES",
    "default_registry",
    "make_pipeline",
    "Budget",
    "CandidateResource",
    "ContextBuildState",
    "OperatorMetrics",
    "Query",
    "TraceEvent",
]
