"""The context-engineering engine.

The engine is the concrete implementation of

    c = F_s(x, K_s, R_s, U, B)

Given a task ``x`` (and optional domain hint / skill / budget), it:
  1. selects a skill ``s`` (the recipe),
  2. builds a :class:`PipelineState` exposing ``K_s``/rules/prefs/budget,
  3. runs the skill's operator pipeline ``F_s`` in order, and
  4. packages the result as a :class:`ContextPackage` ``c`` with full trace.

The engine holds the registries but contains no domain logic itself — all of
that lives in operators, rules, skills and resources.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .evaluators import register_default_evaluators
from .models import (
    ContextPackage,
    EvaluationResult,
    PipelineState,
    Skill,
)
from .operators import classify_task, register_default_operators
from .registries import (
    EvaluatorRegistry,
    OperatorRegistry,
    ResourceRegistry,
    RuleRegistry,
    SkillRegistry,
)


class ContextEngine:
    """Orchestrates skill selection and pipeline execution."""

    def __init__(
        self,
        skill_registry: Optional[SkillRegistry] = None,
        rule_registry: Optional[RuleRegistry] = None,
        resource_registry: Optional[ResourceRegistry] = None,
        operator_registry: Optional[OperatorRegistry] = None,
        evaluator_registry: Optional[EvaluatorRegistry] = None,
    ) -> None:
        self.skills = skill_registry or SkillRegistry()
        self.rules = rule_registry or RuleRegistry()
        self.resources = resource_registry or ResourceRegistry()
        self.operators = operator_registry or register_default_operators(
            OperatorRegistry()
        )
        self.evaluators = evaluator_registry or register_default_evaluators(
            EvaluatorRegistry()
        )

    # ------------------------------------------------------------------ #
    # Skill selection                                                    #
    # ------------------------------------------------------------------ #
    def select_skill(self, task: str, domain_hint: Optional[str] = None) -> Skill:
        """Select a skill for the task; fall back to keyword classification."""
        if not domain_hint:
            domain_hint = classify_task(task)
        return self.skills.select_skill(task, domain_hint=domain_hint)

    # ------------------------------------------------------------------ #
    # Context construction  (c = F_s(x, K_s, R_s, U, B))                 #
    # ------------------------------------------------------------------ #
    def build_context(
        self,
        task: str,
        domain_hint: Optional[str] = None,
        skill_id: Optional[str] = None,
        user_prefs: Optional[Dict] = None,
        budget: Optional[int] = None,
    ) -> ContextPackage:
        """Build the final context package ``c`` for ``task``."""
        domain = domain_hint or classify_task(task)

        if skill_id:
            skill = self.skills.get_skill(skill_id)
        else:
            skill = self.skills.select_skill(task, domain_hint=domain)

        # Budget B: explicit arg > user pref > skill default.
        prefs = user_prefs or {}
        effective_budget = (
            budget
            if budget is not None
            else prefs.get("budget_tokens", skill.default_budget_tokens)
        )

        state = PipelineState(
            task=task,
            domain=domain,
            skill=skill,
            resource_registry=self.resources,
            rule_registry=self.rules,
            operator_registry=self.operators,
            user_prefs=prefs,
            budget=effective_budget,
        )
        state.log(
            "skill_selected",
            skill_id=skill.id,
            reason=f"domain='{domain}' -> skill '{skill.id}'",
            budget=effective_budget,
        )

        # Run the pipeline F_s.
        for op_name in skill.operators:
            if not self.operators.has(op_name):
                state.warnings.append(f"unknown operator skipped: {op_name}")
                state.log("operator_missing", operator=op_name)
                continue
            state = self.operators.run_operator(op_name, state)

        return self._to_package(state)

    @staticmethod
    def _to_package(state: PipelineState) -> ContextPackage:
        """Materialize a :class:`ContextPackage` from final pipeline state."""
        return ContextPackage(
            task=state.task,
            skill_id=state.skill.id,
            selected_rules=state.selected_rules,
            selected_items=state.selected_items,
            formatted_context=state.formatted_context,
            token_estimate=state.token_estimate,
            trace=state.trace,
            warnings=state.warnings,
            budget_tokens=state.budget,
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                         #
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        package: ContextPackage,
        spec: Dict,
        evaluator_names: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Run the skill's evaluators (or an override list) against ``package``."""
        if evaluator_names is None:
            skill = self.skills.get_skill(package.skill_id)
            evaluator_names = skill.evaluators
        return self.evaluators.evaluate(package, spec, evaluator_names)
