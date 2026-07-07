"""Context-engineering framework.

Implements ``c = F_s(x, K_s, R_s, U, B)``:
  x   = task
  K_s = static resources (ResourceRegistry)
  R_s = selected rules (RuleRegistry)
  U   = user/session preferences
  B   = token budget
  F_s = a skill's ordered operator pipeline (OperatorRegistry)
  c   = the final ContextPackage
"""
from .engine import ContextEngine
from .models import (
    ContextItem,
    ContextPackage,
    EvaluationResult,
    ExperimentResult,
    PipelineState,
    Resource,
    Rule,
    Skill,
    estimate_tokens,
)
from .registries import (
    EvaluatorRegistry,
    OperatorRegistry,
    ResourceRegistry,
    RuleRegistry,
    SkillRegistry,
)

__all__ = [
    "ContextEngine",
    "ContextItem",
    "ContextPackage",
    "EvaluationResult",
    "ExperimentResult",
    "PipelineState",
    "Resource",
    "Rule",
    "Skill",
    "estimate_tokens",
    "EvaluatorRegistry",
    "OperatorRegistry",
    "ResourceRegistry",
    "RuleRegistry",
    "SkillRegistry",
]

__version__ = "0.1.0"
