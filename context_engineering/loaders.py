"""Load example data (resources/rules/skills/tasks/gold) from JSON on disk.

Each domain folder under ``examples/`` contains:
  resources.json, rules.json, skills.json, tasks.json, gold.json

These loaders build a fully-wired :class:`ContextEngine` from that data.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from .engine import ContextEngine
from .models import Resource, Rule, Skill


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_domains(examples_dir: str) -> List[str]:
    """Return domain folder names that contain a skills.json file."""
    domains = []
    for name in sorted(os.listdir(examples_dir)):
        folder = os.path.join(examples_dir, name)
        if os.path.isdir(folder) and os.path.exists(
            os.path.join(folder, "skills.json")
        ):
            domains.append(name)
    return domains


def load_rules(path: str) -> List[Rule]:
    return [Rule(**r) for r in _read_json(path)]


def load_skills(path: str) -> List[Skill]:
    return [Skill(**s) for s in _read_json(path)]


def load_resources(path: str) -> List[Resource]:
    return [Resource(**r) for r in _read_json(path)]


def load_tasks(path: str) -> List[Dict[str, Any]]:
    return _read_json(path)


def load_gold(path: str) -> Dict[str, Dict[str, Any]]:
    """Return a mapping of ``task_id -> gold spec``."""
    data = _read_json(path)
    if isinstance(data, list):
        return {entry["task_id"]: entry for entry in data}
    return data


def build_engine(examples_dir: str, domains: List[str] | None = None) -> ContextEngine:
    """Build a single engine populated with every requested domain's data."""
    engine = ContextEngine()
    for domain in domains or list_domains(examples_dir):
        folder = os.path.join(examples_dir, domain)
        for rule in load_rules(os.path.join(folder, "rules.json")):
            engine.rules.register_rule(rule)
        for skill in load_skills(os.path.join(folder, "skills.json")):
            engine.skills.register_skill(skill)
        for res in load_resources(os.path.join(folder, "resources.json")):
            engine.resources.add_resource(res)
    return engine


def load_domain_bundle(
    examples_dir: str, domain: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return ``(tasks, gold)`` for a single domain."""
    folder = os.path.join(examples_dir, domain)
    tasks = load_tasks(os.path.join(folder, "tasks.json"))
    gold = load_gold(os.path.join(folder, "gold.json"))
    return tasks, gold
