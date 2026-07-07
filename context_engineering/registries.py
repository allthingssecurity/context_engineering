"""Registries: the pluggable stores for skills, rules, resources, operators
and evaluators.

Everything the engine needs is looked up through one of these registries, so
adding a new domain/skill/operator/evaluator never requires touching the
engine itself.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ContextPackage,
    EvaluationResult,
    PipelineState,
    Resource,
    Rule,
    Skill,
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used by lexical retrieval."""
    return _WORD_RE.findall(text.lower())


# --------------------------------------------------------------------------- #
# Skill registry                                                              #
# --------------------------------------------------------------------------- #
class SkillRegistry:
    """Stores skills and selects one for a task/domain."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register_skill(self, skill: Skill) -> None:
        self._skills[skill.id] = skill

    def get_skill(self, skill_id: str) -> Skill:
        if skill_id not in self._skills:
            raise KeyError(f"unknown skill: {skill_id}")
        return self._skills[skill_id]

    def list_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def select_skill(
        self, task: str, domain_hint: Optional[str] = None
    ) -> Skill:
        """Pick the best skill for a task.

        Selection order:
          1. If ``domain_hint`` is given, restrict to skills whose
             ``task_types`` include it.
          2. Prefer the skill flagged ``metadata["default"] == True``.
          3. Otherwise fall back to a keyword score against the task.
        """
        candidates = self.list_skills()
        if domain_hint:
            scoped = [s for s in candidates if domain_hint in s.task_types]
            if scoped:
                candidates = scoped

        # Prefer an explicitly-marked default skill within the candidate set.
        defaults = [s for s in candidates if s.metadata.get("default")]
        if defaults:
            return defaults[0]

        # Keyword fallback: score task tokens against skill task_types/id/desc.
        task_tokens = set(tokenize(task))

        def score(skill: Skill) -> int:
            hay = " ".join(
                [skill.id, skill.description, *skill.task_types]
            )
            return len(task_tokens & set(tokenize(hay)))

        if not candidates:
            raise ValueError("no skills registered")
        return max(candidates, key=score)


# --------------------------------------------------------------------------- #
# Rule registry                                                               #
# --------------------------------------------------------------------------- #
class RuleRegistry:
    """Stores rules and selects/resolves them by scope."""

    def __init__(self) -> None:
        self._rules: Dict[str, Rule] = {}

    def register_rule(self, rule: Rule) -> None:
        self._rules[rule.id] = rule

    def get_rule(self, rule_id: str) -> Rule:
        return self._rules[rule_id]

    def list_rules(self) -> List[Rule]:
        return list(self._rules.values())

    def select_rules(
        self, scopes: List[str], status: str = "active"
    ) -> List[Rule]:
        """Return rules whose scopes intersect ``scopes`` and match ``status``.

        ``status=None`` selects any status.  Results are ordered by descending
        priority for stable, meaningful output.
        """
        scope_set = set(scopes)
        selected = [
            r
            for r in self._rules.values()
            if scope_set & set(r.scopes)
            and (status is None or r.status == status)
        ]
        selected = self.resolve_conflicts(selected)
        return sorted(selected, key=lambda r: (-r.priority, r.id))

    def resolve_conflicts(self, rules: List[Rule]) -> List[Rule]:
        """Resolve conflicts within ``metadata['conflict_group']`` groups.

        Within a conflict group the highest-priority rule wins; ties break on
        rule id for determinism.  Rules with no conflict group pass through.
        """
        by_group: Dict[str, List[Rule]] = {}
        passthrough: List[Rule] = []
        for r in rules:
            group = r.metadata.get("conflict_group")
            if group is None:
                passthrough.append(r)
            else:
                by_group.setdefault(group, []).append(r)

        winners: List[Rule] = list(passthrough)
        for group, members in by_group.items():
            winner = max(members, key=lambda r: (r.priority, r.id))
            winners.append(winner)
        return winners


# --------------------------------------------------------------------------- #
# Resource registry                                                           #
# --------------------------------------------------------------------------- #
class ResourceRegistry:
    """Stores resources and provides lexical search (the ``K_s`` store)."""

    def __init__(self) -> None:
        self._resources: Dict[str, Resource] = {}

    def add_resource(self, resource: Resource) -> None:
        self._resources[resource.id] = resource

    def get_resource(self, resource_id: str) -> Resource:
        return self._resources[resource_id]

    def list_resources(self, domain: Optional[str] = None) -> List[Resource]:
        items = list(self._resources.values())
        if domain:
            items = [r for r in items if r.domain == domain]
        return items

    def _idf(self, corpus: List[Resource]) -> Dict[str, float]:
        """Inverse document frequency over a corpus of resources."""
        n = len(corpus) or 1
        df: Dict[str, int] = {}
        for res in corpus:
            for term in set(tokenize(res.title + " " + res.content)):
                df[term] = df.get(term, 0) + 1
        return {
            term: math.log(1 + n / (1 + freq)) for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        domain: Optional[str] = None,
        type: Optional[str] = None,
    ) -> List[tuple[Resource, float]]:
        """Lexical (TF-IDF-style) search over resources.

        Returns ``(resource, score)`` pairs sorted by descending score.  This
        is a local, deterministic stand-in for a vector store — no API needed.
        """
        corpus = self.list_resources(domain)
        if type:
            corpus = [r for r in corpus if r.type == type]
        idf = self._idf(corpus)
        q_terms = tokenize(query)

        scored: List[tuple[Resource, float]] = []
        for res in corpus:
            doc_tokens = tokenize(res.title + " " + res.content)
            if not doc_tokens:
                scored.append((res, 0.0))
                continue
            tf: Dict[str, int] = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1
            length = len(doc_tokens)
            score = 0.0
            for term in q_terms:
                if term in tf:
                    score += (tf[term] / length) * idf.get(term, 0.0)
            # Title matches are worth a small boost.
            title_tokens = set(tokenize(res.title))
            score += 0.1 * len(set(q_terms) & title_tokens)
            scored.append((res, round(score, 6)))

        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored


# --------------------------------------------------------------------------- #
# Operator registry                                                           #
# --------------------------------------------------------------------------- #
class OperatorRegistry:
    """Stores dynamic operators (the building blocks of ``F_s``)."""

    def __init__(self) -> None:
        self._operators: Dict[str, Callable[[PipelineState], PipelineState]] = {}

    def register_operator(
        self, name: str, fn: Callable[[PipelineState], PipelineState]
    ) -> None:
        self._operators[name] = fn

    def has(self, name: str) -> bool:
        return name in self._operators

    def run_operator(self, name: str, state: PipelineState) -> PipelineState:
        if name not in self._operators:
            raise KeyError(f"unknown operator: {name}")
        return self._operators[name](state)

    def list_operators(self) -> List[str]:
        return list(self._operators.keys())


# --------------------------------------------------------------------------- #
# Evaluator registry                                                          #
# --------------------------------------------------------------------------- #
Evaluator = Callable[[ContextPackage, Dict[str, Any]], EvaluationResult]


class EvaluatorRegistry:
    """Stores evaluators and runs them against a context package."""

    def __init__(self) -> None:
        self._evaluators: Dict[str, Evaluator] = {}

    def register_evaluator(self, name: str, fn: Evaluator) -> None:
        self._evaluators[name] = fn

    def get(self, name: str) -> Evaluator:
        return self._evaluators[name]

    def list_evaluators(self) -> List[str]:
        return list(self._evaluators.keys())

    def evaluate(
        self,
        context_package: ContextPackage,
        spec: Dict[str, Any],
        evaluator_names: Optional[List[str]] = None,
    ) -> List[EvaluationResult]:
        """Run the named evaluators (or all) against ``context_package``."""
        names = evaluator_names or list(self._evaluators.keys())
        results: List[EvaluationResult] = []
        for name in names:
            if name not in self._evaluators:
                continue
            results.append(self._evaluators[name](context_package, spec))
        return results
